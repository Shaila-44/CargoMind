"""Leakage-aware integration of all tables into one row per shipment.

Prediction-time convention used for EDA
---------------------------------------
The project does not define when a prediction is made (there is no booking timestamp).
For EDA we use the *departure timestamp* as the reference point and only use operational
observations whose hourly timestamp is <= ``departure_ts - lag_hours`` (default 1 h, so the
hour that ends at departure is not used if ``ts_hour`` marks the start of the hour).
This is an analysis convention, NOT a resolved modelling decision; see notebook 09.

Column prefixes
---------------
o_ / d_ / h_   : origin / destination / transshipment-hub conditions as of the cutoff
wx_o_ / wx_d_ / wx_h_ : weather as of the cutoff
dis_o_ / ...   : disruption events already started and still active at the cutoff
hol_           : calendar (holiday) features - known in advance
tr_o_ / tr_d_  : port traffic for the last *complete* UTC day before departure
hist_          : historical outcomes of shipments that had ARRIVED before the cutoff
LEAK_          : future / outcome information. Built only to measure leakage risk; never a feature.
"""
import numpy as np
import pandas as pd

from . import io

TEL_SNAPSHOT = [
    "throughput_teu",
    "port_utilization_pct",
    "container_dwell_hours",
    "vessel_time_in_port_hours",
    "vessel_arrivals",
    "vessel_departures",
    "vessels_at_anchor",
    "customs_hold_hours",
    "equipment_availability_pct",
]
WX_SNAPSHOT = [
    "temperature_2m_c",
    "relative_humidity_2m_pct",
    "precipitation_mm",
    "wind_speed_10m_kmh",
    "wind_gusts_10m_kmh",
    "surface_pressure_hpa",
    "visibility_m",
    "weather_code",
]
SEV_RANK = {"low": 1, "medium": 2, "high": 3}


def _port_hourly_features(tel, wx):
    """Per port-hour snapshot + trailing-24h aggregates (window ends at and includes the hour)."""
    t = tel.sort_values(["port_id", "ts_hour"]).set_index("ts_hour")
    t["labour_disrupted"] = (t["labour_status"] != "normal").astype(float)
    t["labour_strike"] = (t["labour_status"] == "strike").astype(float)
    g = t.groupby("port_id")
    roll = pd.DataFrame(
        {
            "util_mean24": g["port_utilization_pct"].rolling("24h").mean(),
            "util_max24": g["port_utilization_pct"].rolling("24h").max(),
            "anchor_mean24": g["vessels_at_anchor"].rolling("24h").mean(),
            "anchor_max24": g["vessels_at_anchor"].rolling("24h").max(),
            "dwell_mean24": g["container_dwell_hours"].rolling("24h").mean(),
            "customs_mean24": g["customs_hold_hours"].rolling("24h").mean(),
            "equip_min24": g["equipment_availability_pct"].rolling("24h").min(),
            "labour_disrupted_share24": g["labour_disrupted"].rolling("24h").mean(),
            "arrivals_sum24": g["vessel_arrivals"].rolling("24h").sum(),
            "util_change24": g["port_utilization_pct"].diff(24).values,
        }
    ).reset_index()
    snap = t.reset_index()[["port_id", "ts_hour", "labour_status", "labour_disrupted", "labour_strike"] + TEL_SNAPSHOT]
    port_h = snap.merge(roll, on=["port_id", "ts_hour"], validate="1:1")

    w = wx.sort_values(["port_id", "ts_hour"]).set_index("ts_hour")
    w["is_thunderstorm"] = (w["weather_code"] >= 95).astype(float)
    w["is_rain_any"] = (w["weather_code"] >= 61).astype(float)
    gw = w.groupby("port_id")
    wroll = pd.DataFrame(
        {
            "gust_max24": gw["wind_gusts_10m_kmh"].rolling("24h").max(),
            "wind_mean24": gw["wind_speed_10m_kmh"].rolling("24h").mean(),
            "precip_sum24": gw["precipitation_mm"].rolling("24h").sum(),
            "visibility_min24": gw["visibility_m"].rolling("24h").min(),
            "thunderstorm_hours24": gw["is_thunderstorm"].rolling("24h").sum(),
            "pressure_change24": gw["surface_pressure_hpa"].diff(24).values,
        }
    ).reset_index()
    wsnap = w.reset_index()[["port_id", "ts_hour"] + WX_SNAPSHOT]
    wx_h = wsnap.merge(wroll, on=["port_id", "ts_hour"], validate="1:1")
    return port_h, wx_h


def _attach(base, table, port_col, time_col, prefix):
    """Exact (port, hour) left join. Returns base with prefixed columns; never duplicates rows."""
    right = table.rename(columns={"port_id": port_col, "ts_hour": time_col})
    right = right.rename(columns={c: f"{prefix}{c}" for c in right.columns if c not in (port_col, time_col)})
    out = base.merge(right, on=[port_col, time_col], how="left", validate="m:1")
    assert len(out) == len(base)
    return out


def _disruption_asof(ship, dis, port_col, cutoff_col, prefix):
    """Events at `port_col` that started at or before the cutoff and had not ended by it."""
    keys = ship[["shipment_id", port_col, cutoff_col]].dropna(subset=[port_col])
    m = keys.merge(dis, left_on=port_col, right_on="port_id", how="inner")
    act = m[(m["event_start"] <= m[cutoff_col]) & (m["event_end"] > m[cutoff_col])]
    recent = m[(m["event_start"] <= m[cutoff_col]) & (m["event_start"] > m[cutoff_col] - pd.Timedelta("72h"))]
    agg = act.groupby("shipment_id").agg(
        **{
            f"{prefix}active_n": ("event_id", "size"),
            f"{prefix}active_max_sev": ("sev_rank", "max"),
            f"{prefix}active_capacity_pct": ("affected_capacity_pct", "max"),
        }
    )
    elapsed = (act[cutoff_col] - act["event_start"]).dt.total_seconds() / 3600
    agg[f"{prefix}active_hours_elapsed"] = elapsed.groupby(act["shipment_id"]).max()
    # LEAK: remaining duration requires event_end, which is not known while an event is ongoing
    remaining = (act["event_end"] - act[cutoff_col]).dt.total_seconds() / 3600
    agg[f"LEAK_{prefix}active_hours_remaining"] = remaining.groupby(act["shipment_id"]).max()
    for et in ["weather", "strike", "congestion", "equipment_failure", "cyber"]:
        agg[f"{prefix}active_{et}"] = act[act["event_type"] == et].groupby("shipment_id").size()
    agg[f"{prefix}started_72h_n"] = recent.groupby("shipment_id").size()
    out = ship[["shipment_id"]].merge(agg, left_on="shipment_id", right_index=True, how="left")
    has_port = ship[port_col].notna().values
    for c in out.columns[1:]:
        if c.endswith(("_n", "_max_sev", "_weather", "_strike", "_congestion", "_failure", "_cyber")):
            out.loc[has_port, c] = out.loc[has_port, c].fillna(0)
    out[f"{prefix}active_any"] = np.where(has_port, (out[f"{prefix}active_n"] > 0).astype(float), np.nan)
    return out.drop(columns="shipment_id")


def _disruption_future(ship, dis, port_col, start_col, end_col, prefix):
    """LEAK: events that START after the cutoff but overlap the planned voyage window."""
    keys = ship[["shipment_id", port_col, "cutoff_ts", start_col, end_col]].dropna(subset=[port_col])
    m = keys.merge(dis, left_on=port_col, right_on="port_id", how="inner")
    fut = m[(m["event_start"] > m["cutoff_ts"]) & (m["event_start"] < m[end_col]) & (m["event_end"] > m[start_col])]
    agg = fut.groupby("shipment_id").agg(
        **{f"LEAK_{prefix}future_n": ("event_id", "size"), f"LEAK_{prefix}future_max_sev": ("sev_rank", "max")}
    )
    out = ship[["shipment_id"]].merge(agg, left_on="shipment_id", right_index=True, how="left").drop(columns="shipment_id")
    has = ship[port_col].notna().values
    out.loc[has] = out.loc[has].fillna(0)
    return out


def _holiday_dates(hol, meta):
    """Expand to (port_id, local holiday date). National holidays (blank subdivision) apply to all
    ports in the country; subdivision holidays apply only to ports in that subdivision."""
    rows = []
    for _, p in meta.iterrows():
        h = hol[(hol["country_code"] == p["country_code"]) & (hol["subdivision_code"].isna() | (hol["subdivision_code"] == p["subdivision_code"]))]
        for d in h["holiday_date"].unique():
            rows.append((p["port_id"], pd.Timestamp(d)))
    return pd.DataFrame(rows, columns=["port_id", "holiday_date"]).drop_duplicates()


def _days_to_nearest_holiday(ports, local_dates, hdates):
    out = np.full(len(ports), np.nan)
    by_port = {k: np.sort(v.values.astype("datetime64[D]")) for k, v in hdates.groupby("port_id")["holiday_date"]}
    ld = local_dates.values.astype("datetime64[D]")
    for i, (p, d) in enumerate(zip(ports, ld)):
        arr = by_port.get(p)
        if arr is None or pd.isna(p):
            continue
        out[i] = np.min(np.abs((arr - d).astype(int)))
    return out


def _local_date(ts, ports, tzmap):
    out = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    for p, tz in tzmap.items():
        m = ports == p
        if m.any():
            out[m] = ts[m].dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
    return out


def _local_hour(ts, ports, tzmap):
    out = pd.Series(np.nan, index=ts.index)
    for p, tz in tzmap.items():
        m = ports == p
        if m.any():
            out[m] = ts[m].dt.tz_convert(tz).dt.hour
    return out


def _historical_outcomes(ship, key, cutoff_col="cutoff_ts", min_n=1):
    """Mean delay / share delayed among shipments in the same `key` group whose actual
    arrival happened strictly before this shipment's cutoff (point-in-time correct)."""
    res_mean = np.full(len(ship), np.nan)
    res_rate = np.full(len(ship), np.nan)
    res_n = np.zeros(len(ship))
    for _, g in ship.groupby(key, observed=True):
        arr = g.sort_values("actual_arrival_ts")
        t = arr["actual_arrival_ts"].values
        csum = np.concatenate([[0], np.cumsum(arr["actual_delay_hours"].values)])
        cdel = np.concatenate([[0], np.cumsum((arr["actual_delay_hours"].values > 12).astype(float))])
        k = np.searchsorted(t, g[cutoff_col].values, side="left")
        idx = ship.index.get_indexer(g.index)
        res_n[idx] = k
        with np.errstate(invalid="ignore", divide="ignore"):
            res_mean[idx] = np.where(k >= min_n, csum[k] / k, np.nan)
            res_rate[idx] = np.where(k >= min_n, cdel[k] / k, np.nan)
    name = key if isinstance(key, str) else "_".join(key)
    return pd.DataFrame(
        {f"hist_{name}_n_arrived": res_n, f"hist_{name}_mean_delay_h": res_mean, f"hist_{name}_pct_delayed": res_rate * 100},
        index=ship.index,
    )


def build_shipment_features(lag_hours=1, tables=None):
    T = tables or io.load_all()
    s = T["shipment_records"].copy()
    tel, wx, net, meta = T["port_hourly_telemetry"], T["weather_hourly"], T["port_network"], T["port_metadata"]
    dis = T["disruption_events"].copy()
    dis["sev_rank"] = dis["severity"].astype(str).map(SEV_RANK)
    hol, traffic = T["public_holidays"], T["port_traffic_history"]

    n0 = len(s)
    s["cutoff_ts"] = s["departure_ts"] - pd.Timedelta(hours=lag_hours)

    # --- route / network (static, safe) ---
    s = s.merge(
        net[["route_id", "route_type", "transshipment_port_id", "route_distance_km", "planned_transit_hours"]],
        on="route_id", how="left", validate="m:1",
    )
    s["planned_duration_h"] = (s["expected_arrival_ts"] - s["departure_ts"]).dt.total_seconds() / 3600
    s["planned_vs_route_ratio"] = s["planned_duration_h"] / s["planned_transit_hours"]
    s["od_pair"] = s["origin_port_id"] + "→" + s["destination_port_id"]
    cap = meta.set_index("port_id")["capacity_teu_per_hour"]
    for role, col in [("o", "origin_port_id"), ("d", "destination_port_id"), ("h", "transshipment_port_id")]:
        s[f"{role}_capacity_teu_per_hour"] = s[col].map(cap)
        s[f"{role}_region"] = s[col].map(meta.set_index("port_id")["region"])

    # --- calendar (safe) ---
    tz = meta.set_index("port_id")["local_timezone"].to_dict()
    s["dep_hour_utc"] = s["departure_ts"].dt.hour
    s["dep_hour_local"] = _local_hour(s["departure_ts"], s["origin_port_id"], tz)
    s["dep_dow_utc"] = s["departure_ts"].dt.dayofweek
    s["dep_day_index"] = (s["departure_ts"] - s["departure_ts"].min()).dt.days

    # --- port telemetry & weather as of cutoff ---
    port_h, wx_h = _port_hourly_features(tel, wx)
    for role, col in [("o_", "origin_port_id"), ("d_", "destination_port_id"), ("h_", "transshipment_port_id")]:
        s = _attach(s, port_h, col, "cutoff_ts", role)
        s = _attach(s, wx_h, col, "cutoff_ts", "wx_" + role)

    # --- LEAK: destination conditions at the (future) expected arrival hour ---
    s["exp_arr_hour"] = s["expected_arrival_ts"].dt.floor("h")
    fut = port_h[["port_id", "ts_hour", "port_utilization_pct", "vessels_at_anchor", "anchor_max24", "util_mean24"]]
    s = _attach(s, fut, "destination_port_id", "exp_arr_hour", "LEAK_d_at_arrival_")
    fw = wx_h[["port_id", "ts_hour", "gust_max24"]]
    s = _attach(s, fw, "destination_port_id", "exp_arr_hour", "LEAK_wx_d_at_arrival_")
    # LEAK: origin conditions over the 24 h AFTER departure
    s["dep_plus24"] = s["departure_ts"] + pd.Timedelta("24h")
    s = _attach(s, port_h[["port_id", "ts_hour", "util_mean24", "anchor_max24"]], "origin_port_id", "dep_plus24", "LEAK_o_next24_")
    s = _attach(s, wx_h[["port_id", "ts_hour", "gust_max24"]], "origin_port_id", "dep_plus24", "LEAK_wx_o_next24_")

    # --- disruptions ---
    for role, col in [("dis_o_", "origin_port_id"), ("dis_d_", "destination_port_id"), ("dis_h_", "transshipment_port_id")]:
        s = pd.concat([s, _disruption_asof(s, dis, col, "cutoff_ts", role)], axis=1)
        s = pd.concat([s, _disruption_future(s, dis, col, "departure_ts", "expected_arrival_ts", role)], axis=1)

    # --- holidays (calendar: known in advance) ---
    hd = _holiday_dates(hol, meta)
    s["dep_local_date"] = _local_date(s["departure_ts"], s["origin_port_id"], tz)
    s["exp_arr_local_date"] = _local_date(s["expected_arrival_ts"], s["destination_port_id"], tz)
    s["hol_o_days_to_nearest"] = _days_to_nearest_holiday(s["origin_port_id"], s["dep_local_date"], hd)
    s["hol_d_days_to_nearest_at_exp_arr"] = _days_to_nearest_holiday(s["destination_port_id"], s["exp_arr_local_date"], hd)
    s["hol_o_within_2d"] = (s["hol_o_days_to_nearest"] <= 2).astype(int)
    s["hol_d_within_2d_of_exp_arr"] = (s["hol_d_days_to_nearest_at_exp_arr"] <= 2).astype(int)
    # holidays at the hub during the planned voyage window (calendar, safe)
    hub_h = hd.rename(columns={"port_id": "transshipment_port_id"})
    m = s[["shipment_id", "transshipment_port_id", "dep_local_date", "exp_arr_local_date"]].dropna(subset=["transshipment_port_id"]).merge(hub_h, on="transshipment_port_id")
    m = m[(m["holiday_date"] >= m["dep_local_date"]) & (m["holiday_date"] <= m["exp_arr_local_date"])]
    s["hol_h_days_in_voyage"] = s["shipment_id"].map(m.groupby("shipment_id").size())
    s.loc[s["transshipment_port_id"].notna(), "hol_h_days_in_voyage"] = s.loc[s["transshipment_port_id"].notna(), "hol_h_days_in_voyage"].fillna(0)
    s["hol_calendar_covers_exp_arr"] = s["exp_arr_local_date"] <= hol["holiday_date"].max()

    # --- traffic: last complete UTC day strictly before the departure day ---
    s["traffic_day"] = s["departure_ts"].dt.tz_localize(None).dt.normalize() - pd.Timedelta("1D")
    tr = traffic[["port_id", "period_start", "vessel_calls", "container_vessel_calls", "import_volume_t", "export_volume_t", "traffic_deviation_pct", "avg_vessel_stay_hours"]]
    for role, col in [("tr_o_", "origin_port_id"), ("tr_d_", "destination_port_id")]:
        r = tr.rename(columns={"port_id": col, "period_start": "traffic_day"})
        r = r.rename(columns={c: role + c for c in r.columns if c not in (col, "traffic_day")})
        s = s.merge(r, on=[col, "traffic_day"], how="left", validate="m:1")

    # --- historical outcomes (point-in-time) ---
    for key in ["route_id", "carrier", "origin_port_id", "destination_port_id"]:
        s = pd.concat([s, _historical_outcomes(s, key)], axis=1)

    # --- trailing route frequency (departures on route in previous 7 days; schedule info) ---
    s = s.sort_values("departure_ts").reset_index(drop=True)
    cnt = np.zeros(len(s))
    for _, g in s.groupby("route_id"):
        t = g["departure_ts"].values
        lo = np.searchsorted(t, t - np.timedelta64(7, "D"), side="left")
        hi = np.searchsorted(t, t, side="left")
        cnt[g.index] = hi - lo
    s["route_departures_prev7d"] = cnt

    assert len(s) == n0, "integration changed the number of shipments"
    assert s["shipment_id"].is_unique
    return s
