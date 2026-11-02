"""Modelling table: the EDA's integrated shipment-level dataset plus ML-specific leakage fixes.

The EDA integration (`cargomind_eda.features.build_shipment_features`) is reused unchanged, so
every as-of-cutoff join is the one already validated in EDA notebook 09. This module only adds:

1. Row filter: the 295 shipments whose outcome was never observed in raw are dropped
   (their labels were filled during cleaning, EDA §2).
2. Point-in-time history recomputed on observed shipments only, so filled outcomes cannot
   leak into `hist_*` features, with a minimum of `HIST_MIN_N` prior arrivals.
3. A schedule-based ETA for the 194 observed shipments whose `expected_arrival_ts` was
   reconstructed from the actual arrival during cleaning (EDA §6, "needs a decision").
   ETA-derived features use departure + planned route time for those rows instead.
4. A few encodings (hub id with an explicit DIRECT level, ordinal priority, capped holiday distance).

`feature_catalog()` records, for every column of the integrated table, whether it is used and why not.
"""

import numpy as np
import pandas as pd

from . import config

from cargomind_eda import features as eda_features, io as eda_io, lineage as eda_lineage  # noqa: E402

HIST_MIN_N = 10  # history below 10 prior arrivals is too noisy to be a rate; left NaN

# ---------------------------------------------------------------------------------------------
# Feature sets. One column per EDA redundancy cluster (EDA §7); see feature_catalog() for exclusions.
# ---------------------------------------------------------------------------------------------
CATEGORICAL = ["origin_port_id", "destination_port_id", "hub_port_id", "route_type", "carrier", "cargo_type", "shipping_mode"]

FEATURE_GROUPS = {
    "route": ["origin_port_id", "destination_port_id", "hub_port_id", "route_type", "route_distance_km", "planned_transit_hours"],
    "carrier": ["carrier"],
    "shipment_attributes": ["shipping_mode", "cargo_type", "priority_rank", "container_teu"],
    "calendar": ["dep_hour_local", "dep_dow_utc", "hol_o_days_to_nearest_cap14", "hol_o_within_2d", "hol_h_days_in_voyage_sched"],
    "origin_congestion": [
        "o_port_utilization_pct", "o_util_max24", "o_util_change24", "o_vessel_time_in_port_hours",
        "o_arrivals_sum24", "o_labour_disrupted_share24", "o_customs_mean24", "o_equip_min24",
    ],
    "container_dwell": ["o_container_dwell_hours"],
    "anchor_queue": ["o_anchor_mean24"],
    "weather": ["wx_o_gust_max24", "wx_o_precip_sum24", "wx_o_thunderstorm_hours24", "wx_o_pressure_change24"],
    "disruptions": [
        "dis_o_active_any", "dis_o_active_weather", "dis_o_active_strike", "dis_o_active_congestion",
        "dis_o_active_equipment_failure", "dis_o_active_cyber", "dis_o_active_hours_elapsed", "dis_o_started_72h_n",
    ],
    "destination_state": ["d_util_mean24", "d_anchor_mean24", "dis_d_active_any"],
    "hub_state": ["h_util_mean24", "h_anchor_mean24", "h_container_dwell_hours", "wx_h_gust_max24", "dis_h_active_any"],
    "historical_delay_rates": [
        "hist_carrier_pct_delayed", "hist_carrier_mean_delay_h", "hist_origin_port_id_pct_delayed",
        "hist_destination_port_id_pct_delayed", "hist_route_id_pct_delayed",
    ],
}
FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
FEATURE_TO_GROUP = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}

# Identity features: tell the model *which* port/carrier, not *what state* it is in.
IDENTITY_FEATURES = ["origin_port_id", "destination_port_id", "hub_port_id", "carrier"]

# Operational state at the origin (and destination) at the cutoff, for the anomaly detectors.
# Continuous metrics are standardised per origin port inside the anomaly pipelines (see preprocessing).
ANOMALY_PORT_RELATIVE = [
    "o_port_utilization_pct", "o_util_max24", "o_util_change24", "o_anchor_mean24", "o_container_dwell_hours",
    "o_vessel_time_in_port_hours", "o_arrivals_sum24", "wx_o_gust_max24", "wx_o_precip_sum24", "wx_o_pressure_change24",
]
ANOMALY_RAW = ["o_labour_disrupted_share24", "wx_o_thunderstorm_hours24", "dis_o_active_any", "dis_o_active_hours_elapsed", "dis_o_started_72h_n"]
ANOMALY_FEATURES = ANOMALY_PORT_RELATIVE + ANOMALY_RAW


def _schedule_eta(s):
    """expected_arrival_ts, except where it was reconstructed from the actual arrival in cleaning.
    There: departure + planned route hours x the mode's median planned/route ratio (schedule-only)."""
    ok = ~s["filled_expected_arrival_ts"]
    ratio = s.loc[ok].groupby("shipping_mode")["planned_vs_route_ratio"].median()
    hours = s["planned_transit_hours"] * s["shipping_mode"].map(ratio)
    eta = s["expected_arrival_ts"].copy()
    eta[~ok] = s.loc[~ok, "departure_ts"] + pd.to_timedelta(hours[~ok], unit="h")
    return eta


def _hub_holidays_in_voyage(s, eta, tables):
    """Same definition as the EDA's hol_h_days_in_voyage, but bounded by the schedule ETA."""
    meta = tables["port_metadata"]
    tz = meta.set_index("port_id")["local_timezone"].to_dict()
    hd = eda_features._holiday_dates(tables["public_holidays"], meta).rename(columns={"port_id": "hub_port_id"})
    arr_local = eda_features._local_date(eta, s["destination_port_id"], tz)
    m = pd.DataFrame({"shipment_id": s["shipment_id"], "hub_port_id": s["transshipment_port_id"],
                      "dep": s["dep_local_date"], "arr": arr_local}).dropna(subset=["hub_port_id"]).merge(hd, on="hub_port_id")
    m = m[(m["holiday_date"] >= m["dep"]) & (m["holiday_date"] <= m["arr"])]
    out = s["shipment_id"].map(m.groupby("shipment_id").size())
    return out.where(s["transshipment_port_id"].isna(), out.fillna(0))


def build_modelling_table(lag_hours=config.LAG_HOURS):
    """Return (table, log). `table` has one row per *observed* shipment, sorted by departure."""
    tables = eda_io.load_all()
    s = eda_features.build_shipment_features(lag_hours=lag_hours, tables=tables)
    flags = eda_lineage.shipment_fill_flags()
    s = s.merge(flags, on="shipment_id", how="left", validate="1:1")
    log = {"lag_hours": lag_hours, "n_shipments_integrated": int(len(s))}

    unobserved = s["outcome_not_observed_in_raw"]
    log["n_dropped_unobserved_outcome"] = int(unobserved.sum())
    s = s[~unobserved].sort_values(["departure_ts", "shipment_id"]).reset_index(drop=True)
    log["n_observed"] = int(len(s))

    # (2) history on observed shipments only (replaces the EDA's hist_* columns)
    s = s.drop(columns=[c for c in s.columns if c.startswith("hist_")])
    for key in ["route_id", "carrier", "origin_port_id", "destination_port_id"]:
        s = pd.concat([s, eda_features._historical_outcomes(s, key, min_n=HIST_MIN_N)], axis=1)

    # (3) schedule ETA for reconstructed expected arrivals
    eta = _schedule_eta(s)
    log["n_reconstructed_eta_replaced"] = int(s["filled_expected_arrival_ts"].sum())
    s["hol_h_days_in_voyage_sched"] = _hub_holidays_in_voyage(s, eta, tables)

    # (4) encodings
    s["hub_port_id"] = s["transshipment_port_id"].fillna("DIRECT")
    s["priority_rank"] = s["shipment_priority"].cat.codes.astype(float).replace(-1, np.nan)  # low 0 < medium 1 < high 2
    s["hol_o_days_to_nearest_cap14"] = s["hol_o_days_to_nearest"].clip(upper=14)
    # elapsed hours of an active origin event: 0 when no event is active (the *_any flag carries "no event")
    s["dis_o_active_hours_elapsed"] = s["dis_o_active_hours_elapsed"].fillna(0)

    s["y"] = s[config.TARGET].cat.codes.astype(int)
    assert s["y"].between(0, 2).all()
    assert s["shipment_id"].is_unique
    return s, log


# ---------------------------------------------------------------------------------------------
# Catalogue: why every column of the integrated table is or is not a model input
# ---------------------------------------------------------------------------------------------
_EXCLUSION_RULES = [
    (lambda c: c in ("delay_status", "y"), "target"),
    (lambda c: c in ("actual_delay_hours", "actual_arrival_ts"), "outcome: reproduces the label exactly (leakage)"),
    (lambda c: c.startswith("LEAK_"), "future information after the cutoff (leakage)"),
    (lambda c: c.startswith(("filled_", "raw_label", "outcome_not")), "cleaning-lineage flag: row filtering / sensitivity only"),
    (lambda c: c in ("shipment_id", "data_source", "is_synthetic"), "identifier / constant"),
    (lambda c: c in ("cutoff_ts", "exp_arr_hour", "dep_plus24", "traffic_day", "dep_local_date", "exp_arr_local_date", "hol_calendar_covers_exp_arr"), "join helper / coverage flag"),
    (lambda c: c in ("departure_ts", "dep_day_index"), "absolute time: encodes the period, will not generalise"),
    (lambda c: c == "expected_arrival_ts", "used only via schedule features; 194 values reconstructed from actual arrival"),
    (lambda c: c in ("planned_duration_h", "planned_vs_route_ratio"), "derived from expected_arrival_ts (reconstructed for 194 rows); planned_transit_hours + shipping_mode carry the same schedule information"),
    (lambda c: c in ("route_id", "od_pair"), "90-level identifier; route = origin + destination + hub (EDA R² 0.91)"),
    (lambda c: c == "transshipment_port_id", "re-encoded as hub_port_id with explicit DIRECT"),
    (lambda c: c == "shipment_priority", "re-encoded as ordinal priority_rank"),
    (lambda c: c.endswith("_n_arrived"), "count of prior arrivals grows with calendar time: a hidden time index"),
    (lambda c: c.startswith("hist_") and c.endswith("_mean_delay_h"), "redundant with the matching pct_delayed history (kept for carrier only)"),
    (lambda c: c.startswith("tr_"), "exact daily aggregate of telemetry (EDA §7)"),
    (lambda c: "visibility" in c, "visibility at USLAX/CNSHA/LKCMB was filled during cleaning, not measured"),
    (lambda c: c.startswith("dis_") and ("max_sev" in c or "capacity_pct" in c), "unknown whether severity/capacity are known at event onset (EDA open decision 3)"),
    (lambda c: c.startswith("hol_d_"), "destination holiday at ETA: calendar ends 26 Feb, most test-period ETAs are uncovered (spurious 'no holiday')"),
    (lambda c: c == "hol_h_days_in_voyage", "replaced by hol_h_days_in_voyage_sched (schedule ETA)"),
    (lambda c: c == "hol_o_days_to_nearest", "replaced by capped version hol_o_days_to_nearest_cap14"),
    (lambda c: c.endswith(("_region", "_capacity_teu_per_hour")), "port constant: fully determined by the port id already encoded"),
    (lambda c: c in ("dep_hour_utc",), "redundant with dep_hour_local"),
    (lambda c: c.endswith(("_labour_status", "_labour_strike", "_labour_disrupted")), "labour snapshot: kept as 24 h share at origin only"),
    (lambda c: c.endswith(("_active_n",)), "redundant with *_active_any (ρ > 0.99, EDA §7)"),
    (lambda c: c.startswith("dis_o_active_") or c.startswith("dis_o_started"), "covered by the selected origin disruption columns"),
    (lambda c: c.startswith(("dis_d_", "dis_h_")), "destination/hub disruption detail: only *_active_any kept (weak signal, EDA §4.1e)"),
    (lambda c: c.startswith(("o_", "wx_o_")), "redundant member of an origin redundancy cluster (EDA §7) or no association"),
    (lambda c: c.startswith(("d_", "wx_d_")), "destination state at departure is weak (|δ| ≤ 0.11); only anchor/util 24 h kept"),
    (lambda c: c.startswith(("h_", "wx_h_")), "hub state: only util/anchor/dwell/gust kept"),
    (lambda c: c == "route_departures_prev7d", "no meaningful association (EDA) and depends on schedule completeness"),
]


def feature_catalog(table):
    rows = []
    for c in table.columns:
        if c in FEATURES:
            rows.append((c, "used", FEATURE_TO_GROUP[c], "selected (EDA §5 recommendation)"))
            continue
        reason = next((r for rule, r in _EXCLUSION_RULES if rule(c)), "not selected")
        rows.append((c, "excluded", "", reason))
    return pd.DataFrame(rows, columns=["column", "status", "group", "reason"]).set_index("column")
