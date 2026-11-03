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
# =============================================================================
# WHAT DOES THIS FILE DO?
# =============================================================================
# In machine learning, the features you give your model are like the ingredients
# in a recipe. If you give the model poor ingredients (or worse, ingredients from
# the future!), the predictions will fail in real life.
#
# This file prepares the official "Modelling Table" — the clean dataset where each
# row is one cargo shipment, and each column is a carefully chosen, leakage-free
# piece of information known BEFORE the ship leaves the port.
#
# WHAT IS DATA LEAKAGE?
# Imagine taking a history test, but somebody accidentally wrote the answers on the
# back of the test paper. You would score 100%! But on the real exam without the cheat
# sheet, you would fail.
# In ML, "leakage" happens when data from the FUTURE (e.g., actual arrival time or
# post-departure weather) sneaks into the training inputs.
# This file aggressively prevents leakage so our model works reliably in the real world.
# =============================================================================

# --- Third-party library imports ---
import numpy as np  # For numerical operations and NaN representations
import pandas as pd # For manipulating tabular data tables (DataFrames)

# --- Internal project imports ---
from . import config

# Functions from the EDA (Exploratory Data Analysis) package:
#   features: builds raw joined features
#   io: loads data tables from disk
#   lineage: tracks data cleaning flags (e.g. which values were filled)
from cargomind_eda import features as eda_features, io as eda_io, lineage as eda_lineage  # noqa: E402

# Minimum number of historical arrivals needed before calculating a historical delay rate.
# If a route or carrier only had 2 past shipments, a 50% delay rate is too noisy to be meaningful.
HIST_MIN_N = 10  # history below 10 prior arrivals is too noisy to be a rate; left NaN

# ---------------------------------------------------------------------------------------------
# CATEGORICAL FEATURES: Columns that contain text labels / categories instead of numbers
# ---------------------------------------------------------------------------------------------
CATEGORICAL = [
    "origin_port_id",      # Starting port (e.g., USLAX, CNSHA)
    "destination_port_id", # Ending port
    "hub_port_id",         # Intermediate transfer port (or 'DIRECT' if no stop)
    "route_type",          # 'direct' or 'transshipment'
    "carrier",             # Shipping company (e.g., Maersk, MSC, Evergreen)
    "cargo_type",          # Type of cargo (e.g., electronics, perishables, dry bulk)
    "shipping_mode",       # 'sea' or 'air'
]

# ---------------------------------------------------------------------------------------------
# FEATURE GROUPS: All 51 features organized into logical business concepts
# ---------------------------------------------------------------------------------------------
FEATURE_GROUPS = {
    # 1. Route characteristics: Where is the shipment going, and how long should it take?
    "route": [
        "origin_port_id", "destination_port_id", "hub_port_id",
        "route_type", "route_distance_km", "planned_transit_hours"
    ],

    # 2. Carrier: Who is transporting the cargo?
    "carrier": ["carrier"],

    # 3. Shipment attributes: What is inside, and what is its priority?
    "shipment_attributes": [
        "shipping_mode", "cargo_type", "priority_rank", "container_teu"
    ],

    # 4. Calendar and timing: When does it depart, and are there nearby holidays?
    "calendar": [
        "dep_hour_local", "dep_dow_utc", "hol_o_days_to_nearest_cap14",
        "hol_o_within_2d", "hol_h_days_in_voyage_sched"
    ],

    # 5. Origin port congestion: How busy is the port right before departure?
    "origin_congestion": [
        "o_port_utilization_pct", "o_util_max24", "o_util_change24",
        "o_vessel_time_in_port_hours", "o_arrivals_sum24",
        "o_labour_disrupted_share24", "o_customs_mean24", "o_equip_min24",
    ],

    # 6. Container dwell time: How many hours containers have been waiting at origin
    "container_dwell": ["o_container_dwell_hours"],

    # 7. Anchor queue: Average waiting time of ships anchored outside the origin port
    "anchor_queue": ["o_anchor_mean24"],

    # 8. Weather at origin: Wind gusts, rain, thunderstorms, and atmospheric pressure drops
    "weather": [
        "wx_o_gust_max24", "wx_o_precip_sum24",
        "wx_o_thunderstorm_hours24", "wx_o_pressure_change24"
    ],

    # 9. Disruptions: Strikes, equipment failures, weather alerts active at the origin
    "disruptions": [
        "dis_o_active_any", "dis_o_active_weather", "dis_o_active_strike",
        "dis_o_active_congestion", "dis_o_active_equipment_failure",
        "dis_o_active_cyber", "dis_o_active_hours_elapsed", "dis_o_started_72h_n",
    ],

    # 10. Destination port state: Congestion and disruption status at destination
    "destination_state": ["d_util_mean24", "d_anchor_mean24", "dis_d_active_any"],

    # 11. Intermediate hub state: Congestion and weather at the transfer hub
    "hub_state": [
        "h_util_mean24", "h_anchor_mean24", "h_container_dwell_hours",
        "wx_h_gust_max24", "dis_h_active_any"
    ],

    # 12. Historical track record: Past delay percentage for this carrier, route, and ports
    "historical_delay_rates": [
        "hist_carrier_pct_delayed", "hist_carrier_mean_delay_h",
        "hist_origin_port_id_pct_delayed", "hist_destination_port_id_pct_delayed",
        "hist_route_id_pct_delayed",
    ],
}

# The complete list of all 51 features combined
FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]

# All features that are numeric (continuous measurements, counts, hours, percentages)
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

# Lookup dictionary mapping feature name -> group name (e.g. 'wx_o_gust_max24' -> 'weather')
FEATURE_TO_GROUP = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}

# Identity features: Tells the model *who* (which port/carrier), not *what state* it's in.
# Used during ablation experiments to test if the model memorises port IDs.
IDENTITY_FEATURES = ["origin_port_id", "destination_port_id", "hub_port_id", "carrier"]

# Features used by the Anomaly Detectors (Isolation Forest and One-Class SVM).
# These focus purely on whether port operating and weather conditions are abnormal.
ANOMALY_PORT_RELATIVE = [
    "o_port_utilization_pct", "o_util_max24", "o_util_change24", "o_anchor_mean24", "o_container_dwell_hours",
    "o_vessel_time_in_port_hours", "o_arrivals_sum24", "wx_o_gust_max24", "wx_o_precip_sum24", "wx_o_pressure_change24",
]
ANOMALY_RAW = ["o_labour_disrupted_share24", "wx_o_thunderstorm_hours24", "dis_o_active_any", "dis_o_active_hours_elapsed", "dis_o_started_72h_n"]
ANOMALY_FEATURES = ANOMALY_PORT_RELATIVE + ANOMALY_RAW


def _schedule_eta(s):
    """Reconstruct an honest, planned ETA without leaking actual arrival times.

    In the raw dataset, 194 shipments were missing their expected arrival time,
    and a data cleaning script had filled them in by looking at when they ACTUALLY arrived.
    That is data leakage!
    This helper replaces those filled values with a schedule-only estimation:
      Departure Time + (Planned Transit Hours * Typical Transit Ratio)
    """
    ok = ~s["filled_expected_arrival_ts"]
    ratio = s.loc[ok].groupby("shipping_mode")["planned_vs_route_ratio"].median()
    hours = s["planned_transit_hours"] * s["shipping_mode"].map(ratio)
    eta = s["expected_arrival_ts"].copy()
    eta[~ok] = s.loc[~ok, "departure_ts"] + pd.to_timedelta(hours[~ok], unit="h")
    return eta


def _hub_holidays_in_voyage(s, eta, tables):
    """Count how many public holidays occur at the intermediate hub port during the voyage.

    Holidays at transshipment hubs cause port closures and crane standstills, leading to delays.
    We count holidays strictly between departure date and estimated arrival date.
    """
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
    """Build and return the master modelling table and diagnostic log.

    Steps performed:
      1. Load all tables and build initial features as of the cutoff (departure - lag_hours).
      2. Drop the 295 shipments whose outcome was synthetic/unobserved.
      3. Recompute historical delay track records using ONLY observed past shipments (min 10 arrivals).
      4. Fix ETA leakage using schedule-based transit calculations.
      5. Encode categorical variables and map target class (0=On-Time, 1=At-Risk, 2=Critical).
    """
    tables = eda_io.load_all()
    s = eda_features.build_shipment_features(lag_hours=lag_hours, tables=tables)
    flags = eda_lineage.shipment_fill_flags()
    s = s.merge(flags, on="shipment_id", how="left", validate="1:1")
    log = {"lag_hours": lag_hours, "n_shipments_integrated": int(len(s))}

    # Step 1: Filter out unobserved outcomes (synthetic or filled target labels)
    unobserved = s["outcome_not_observed_in_raw"]
    log["n_dropped_unobserved_outcome"] = int(unobserved.sum())
    s = s[~unobserved].sort_values(["departure_ts", "shipment_id"]).reset_index(drop=True)
    log["n_observed"] = int(len(s))

    # Step 2: Recompute historical delay statistics strictly on observed past shipments
    # (Removes any possibility of synthetic values polluting historical delay averages)
    s = s.drop(columns=[c for c in s.columns if c.startswith("hist_")])
    for key in ["route_id", "carrier", "origin_port_id", "destination_port_id"]:
        s = pd.concat([s, eda_features._historical_outcomes(s, key, min_n=HIST_MIN_N)], axis=1)

    # Step 3: Replace leaked ETAs with pure schedule-based ETAs
    eta = _schedule_eta(s)
    log["n_reconstructed_eta_replaced"] = int(s["filled_expected_arrival_ts"].sum())
    s["hol_h_days_in_voyage_sched"] = _hub_holidays_in_voyage(s, eta, tables)

    # Step 4: Value encodings and boundary clippings
    # Replace empty transshipment hub with an explicit 'DIRECT' category
    s["hub_port_id"] = s["transshipment_port_id"].fillna("DIRECT")

    # Map priority ranks: low -> 0.0, medium -> 1.0, high -> 2.0
    s["priority_rank"] = s["shipment_priority"].cat.codes.astype(float).replace(-1, np.nan)

    # Cap holiday distance at 14 days (a holiday 30 days away has identical effect as 14 days)
    s["hol_o_days_to_nearest_cap14"] = s["hol_o_days_to_nearest"].clip(upper=14)

    # Elapsed disruption hours: 0 when no disruption is currently active
    s["dis_o_active_hours_elapsed"] = s["dis_o_active_hours_elapsed"].fillna(0)

    # Target variable 'y': 0 for On-Time, 1 for At-Risk, 2 for Critical
    s["y"] = s[config.TARGET].cat.codes.astype(int)
    assert s["y"].between(0, 2).all(), "Target classes must strictly be 0, 1, or 2"
    assert s["shipment_id"].is_unique, "Each shipment row must have a unique ID"
    return s, log


# ---------------------------------------------------------------------------------------------
# FEATURE CATALOGUE AUDIT: Clear rules documenting why every column is included or excluded
# ---------------------------------------------------------------------------------------------
_EXCLUSION_RULES = [
    # The actual outcome we are trying to predict
    (lambda c: c in ("delay_status", "y"), "target"),

    # Direct leakage: these numbers literally reveal the answer
    (lambda c: c in ("actual_delay_hours", "actual_arrival_ts"), "outcome: reproduces the label exactly (leakage)"),

    # Post-departure events (happened after ship departed)
    (lambda c: c.startswith("LEAK_"), "future information after the cutoff (leakage)"),

    # Internal pipeline cleaning flags (used for audit only)
    (lambda c: c.startswith(("filled_", "raw_label", "outcome_not")), "cleaning-lineage flag: row filtering / sensitivity only"),

    # IDs that carry no predictive generalisation
    (lambda c: c in ("shipment_id", "data_source", "is_synthetic"), "identifier / constant"),

    # Timestamps used only for joining tables
    (lambda c: c in ("cutoff_ts", "exp_arr_hour", "dep_plus24", "traffic_day", "dep_local_date", "exp_arr_local_date", "hol_calendar_covers_exp_arr"), "join helper / coverage flag"),

    # Absolute dates (would cause model to memorise January vs February instead of learning general patterns)
    (lambda c: c in ("departure_ts", "dep_day_index"), "absolute time: encodes the period, will not generalise"),

    # Replaced by cleaner schedule-derived versions
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
    """Produces an auditable report documenting every single column in the table.

    For each column, specifies:
      - Status: 'used' or 'excluded'
      - Group: Which functional feature group it belongs to (e.g. weather, route)
      - Reason: Why it was selected or the explicit justification for excluding it.
    """
    rows = []
    for c in table.columns:
        if c in FEATURES:
            rows.append((c, "used", FEATURE_TO_GROUP[c], "selected (EDA §5 recommendation)"))
            continue
        reason = next((r for rule, r in _EXCLUSION_RULES if rule(c)), "not selected")
        rows.append((c, "excluded", "", reason))
    return pd.DataFrame(rows, columns=["column", "status", "group", "reason"]).set_index("column")
