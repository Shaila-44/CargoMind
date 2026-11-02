"""DL data-readiness measurements on Dataset/Processed (read-only). Every function returns a table
and writes it under dl/data_audit/tables/."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import OUT, PPT_SIGNALS, PROCESSED, TARGET, WINDOW  # noqa: E402

TABLES = ["port_hourly_telemetry", "weather_hourly", "port_metadata", "port_network", "disruption_events",
          "public_holidays", "port_traffic_history", "shipment_records"]
KEYS = {"port_hourly_telemetry": ["port_id", "ts_hour"], "weather_hourly": ["port_id", "ts_hour"], "port_metadata": ["port_id"],
        "port_network": ["route_id"], "disruption_events": ["event_id"], "public_holidays": ["country_code", "subdivision_code", "holiday_date"],
        "port_traffic_history": ["port_id", "period_start"], "shipment_records": ["shipment_id"]}
TS_COLS = {"port_hourly_telemetry": ["ts_hour"], "weather_hourly": ["ts_hour"], "disruption_events": ["event_start", "event_end"],
           "shipment_records": ["departure_ts", "expected_arrival_ts", "actual_arrival_ts"], "port_traffic_history": ["period_start"],
           "public_holidays": ["holiday_date"]}


def save(df, name, index=False):
    df.to_csv(OUT["tables"] / f"{name}.csv", index=index)
    return df


def load():
    return {t: pd.read_csv(PROCESSED / f"{t}.csv") for t in TABLES}


# ------------------------------------------------------------------------------------------
# A. Schema and completeness
# ------------------------------------------------------------------------------------------
def schema(tables):
    ports = set(tables["port_metadata"]["port_id"])
    rows, cols = [], []
    for name, df in tables.items():
        key = KEYS[name]
        pid = [c for c in df.columns if c.endswith("port_id")]
        bad_port = sum(int((~df[c].dropna().isin(ports)).sum()) for c in pid)
        tz = []
        for c in TS_COLS.get(name, []):
            s = df[c].dropna().astype(str)
            share = s.str.endswith("+00:00").mean()
            tz.append(f"{c}: {share:.0%} '+00:00'" if "ts" in c or "event" in c else f"{c}: date")
        rows.append({"table": name, "rows": len(df), "columns": df.shape[1], "missing_cells": int(df.isna().sum().sum()),
                     "exact_duplicate_rows": int(df.duplicated().sum()), "primary_key": "+".join(key),
                     "duplicate_keys": int(df.duplicated(key).sum()), "unknown_port_ids": bad_port, "timestamp_tz": "; ".join(tz)})
        for c in df.columns:
            s = df[c]
            r = {"table": name, "column": c, "dtype": str(s.dtype), "missing": int(s.isna().sum()), "n_unique": int(s.nunique())}
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                q = s.quantile([0, 0.01, 0.5, 0.99, 1]).values
                r.update({"min": q[0], "p01": q[1], "p50": q[2], "p99": q[3], "max": q[4], "std": s.std()})
            else:
                vc = s.astype(str).value_counts()
                r["top_values"] = "; ".join(f"{k} ({v})" for k, v in vc.head(5).items())
            cols.append(r)
    return save(pd.DataFrame(rows), "schema_tables"), save(pd.DataFrame(cols), "schema_columns")


# ------------------------------------------------------------------------------------------
# B. Temporal integrity per port
# ------------------------------------------------------------------------------------------
def timeline(df, name):
    out = []
    raw_order = df.copy()
    raw_order["ts"] = pd.to_datetime(raw_order["ts_hour"], utc=True)
    tz_ok = raw_order["ts_hour"].astype(str).str.endswith("+00:00")
    for p, g in raw_order.groupby("port_id", sort=True):
        ts = g["ts"]
        d = ts.sort_values().diff().dropna()
        full = pd.date_range(ts.min(), ts.max(), freq="h")
        missing = full.difference(ts)
        gaps = d[d > pd.Timedelta(hours=1)]
        out.append({"table": name, "port_id": p, "first_hour": ts.min(), "last_hour": ts.max(), "observations": len(ts),
                    "expected_hours": len(full), "missing_hours": len(missing), "duplicate_timestamps": int(ts.duplicated().sum()),
                    "file_order_monotonic": bool(ts.is_monotonic_increasing), "irregular_intervals": int((d != pd.Timedelta(hours=1)).sum()),
                    "largest_gap_h": float(gaps.max() / pd.Timedelta(hours=1)) if len(gaps) else 1.0,
                    "non_hour_aligned": int((ts.dt.minute != 0).sum() + (ts.dt.second != 0).sum()),
                    "all_utc_+00:00": bool(tz_ok[g.index].all())})
    return pd.DataFrame(out)


def temporal(tables):
    t = pd.concat([timeline(tables["port_hourly_telemetry"], "port_hourly_telemetry"), timeline(tables["weather_hourly"], "weather_hourly")])
    a = tables["port_hourly_telemetry"].set_index(["port_id", "ts_hour"]).index
    b = tables["weather_hourly"].set_index(["port_id", "ts_hour"]).index
    join = {"telemetry_keys": len(a), "weather_keys": len(b), "keys_in_both": int(a.isin(b).sum()),
            "telemetry_only": int((~a.isin(b)).sum()), "weather_only": int((~b.isin(a)).sum()),
            "common_hour_grid_for_all_ports": bool(t.groupby("table")[["first_hour", "last_hour"]].nunique().max().max() == 1)}
    save(t, "temporal_coverage_by_port")
    return t, join


# ------------------------------------------------------------------------------------------
# Hourly panel (telemetry + weather + calendar + disruption state known at t)
# ------------------------------------------------------------------------------------------
def panel(tables):
    tel = tables["port_hourly_telemetry"].copy()
    wx = tables["weather_hourly"].drop(columns=["data_source"]).copy()
    meta = tables["port_metadata"].set_index("port_id")
    df = tel.merge(wx, on=["port_id", "ts_hour"], how="left", validate="one_to_one")
    df["ts"] = pd.to_datetime(df["ts_hour"], utc=True)
    df = df.sort_values(["port_id", "ts"]).reset_index(drop=True)
    local = [ts.tz_convert(meta.loc[p, "local_timezone"]) for p, ts in zip(df["port_id"], df["ts"])]
    df["local_hour"] = [x.hour for x in local]
    df["local_date"] = [x.date() for x in local]
    df["dow"] = df["ts"].dt.dayofweek
    ev = tables["disruption_events"].copy()
    ev["s"], ev["e"] = pd.to_datetime(ev["event_start"], utc=True), pd.to_datetime(ev["event_end"], utc=True)
    df["disruption_active"] = 0
    df["disruption_hours_elapsed"] = 0.0
    for _, e in ev.iterrows():
        m = (df["port_id"] == e["port_id"]) & (df["ts"] >= e["s"]) & (df["ts"] < e["e"])
        df.loc[m, "disruption_active"] = 1
        df.loc[m, "disruption_hours_elapsed"] = np.maximum(df.loc[m, "disruption_hours_elapsed"],
                                                           (df.loc[m, "ts"] - e["s"]) / pd.Timedelta(hours=1))
    hol = tables["public_holidays"]
    hol_dates = {}
    for p in meta.index:
        cc, sub = meta.loc[p, "country_code"], meta.loc[p, "subdivision_code"]
        h = hol[(hol["country_code"] == cc) & (hol["subdivision_code"].isna() | (hol["subdivision_code"] == sub))]
        dates = set()
        for _, r in h.iterrows():
            d0 = pd.Timestamp(r["holiday_date"]).date()
            for k in range(int(r["holiday_period_days"]) if pd.notna(r["holiday_period_days"]) else 1):
                dates.add(d0 + pd.Timedelta(days=k))
        hol_dates[p] = dates
    df["holiday_local"] = [int(d in hol_dates[p]) for p, d in zip(df["port_id"], df["local_date"])]
    for lab in ["normal", "reduced", "strike"]:
        df[f"labour_{lab}"] = (df["labour_status"] == lab).astype(int)
    df["y_next"] = df.groupby("port_id")[TARGET].shift(-1)
    df["dy_next"] = df["y_next"] - df[TARGET]
    return df


# ------------------------------------------------------------------------------------------
# C. Target
# ------------------------------------------------------------------------------------------
def acf(x, lags):
    x = x - x.mean()
    v = (x * x).mean()
    return [float((x[:-k] * x[k:]).mean() / v) for k in lags]


def target(df, tel_status):
    df = df.copy()
    df["dwell_status"] = tel_status[TARGET].values
    rows = []
    lags = [1, 2, 3, 6, 12, 24, 48]
    for p, g in list(df.groupby("port_id")) + [("ALL", df)]:
        y = g[TARGET].values
        d = g.groupby("port_id")[TARGET].diff().dropna().values
        q = np.percentile(y, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        med_d = np.median(np.abs(d - np.median(d)))
        r = {"port_id": p, "n": len(y), "reconstructed": int((g["dwell_status"] == "reconstructed").sum()),
             "converted": int((g["dwell_status"] == "converted").sum()), "mean": y.mean(), "std": y.std(),
             **{f"q{k}": v for k, v in zip([0, 1, 5, 25, 50, 75, 95, 99, 100], q)},
             "negative": int((y < 0).sum()), "zero": int((y == 0).sum()),
             "hourly_change_std": d.std(), "hourly_change_abs_p95": np.percentile(np.abs(d), 95),
             "jumps_gt_6_robust_sd": int((np.abs(d - np.median(d)) > 6 * 1.4826 * med_d).sum()),
             "persistence_mae": float(np.abs(d).mean()), "persistence_rmse": float(np.sqrt((d ** 2).mean())),
             "level_std_over_change_std": y.std() / d.std()}
        if p != "ALL":
            r.update({f"acf_lag{k}": v for k, v in zip(lags, acf(y, lags))})
            r.update({f"acf_change_lag{k}": v for k, v in zip([1, 2, 24], acf(d, [1, 2, 24]))})
            daily = g.set_index("ts")[TARGET].resample("D").mean()
            r["daily_mean_trend_h_per_week"] = float(np.polyfit(np.arange(len(daily)), daily.values, 1)[0] * 7)
            r["daily_mean_min"], r["daily_mean_max"] = daily.min(), daily.max()
            # irreducible-noise estimate: for a smooth level plus white noise, var(diff) ~ 2 sigma^2
            r["white_noise_sd_upper_bound"] = d.std() / np.sqrt(2)
        rows.append(r)
    out = save(pd.DataFrame(rows), "target_profile_by_port")
    # dwell vs vessel time in port: distinct variables?
    within = df.groupby("port_id")[[TARGET, "vessel_time_in_port_hours"]].corr().xs(TARGET, level=1)["vessel_time_in_port_hours"]
    save(within.rename("corr_dwell_vs_vessel_time").reset_index(), "target_vs_vessel_time")
    return out, df


# ------------------------------------------------------------------------------------------
# D. Feature suitability
# ------------------------------------------------------------------------------------------
CANDIDATES = {
    # PPT signals
    "container_dwell_hours": ("PPT", "telemetry", "h"), "throughput_teu": ("PPT", "telemetry", "TEU/h"),
    "port_utilization_pct": ("PPT", "telemetry", "% of capacity"), "customs_hold_hours": ("PPT", "telemetry", "h"),
    "temperature_2m_c": ("PPT", "weather", "°C"), "wind_speed_10m_kmh": ("PPT", "weather", "km/h"),
    "labour_status": ("PPT", "telemetry", "category"), "equipment_availability_pct": ("PPT", "telemetry", "%"),
    # alternatives found in the data
    "vessels_at_anchor": ("alt", "telemetry", "vessels"), "vessel_time_in_port_hours": ("alt", "telemetry", "h"),
    "vessel_arrivals": ("alt", "telemetry", "vessels/h"), "vessel_departures": ("alt", "telemetry", "vessels/h"),
    "wind_gusts_10m_kmh": ("alt", "weather", "km/h"), "precipitation_mm": ("alt", "weather", "mm/h"),
    "relative_humidity_2m_pct": ("alt", "weather", "%"), "surface_pressure_hpa": ("alt", "weather", "hPa"),
    "visibility_m": ("alt", "weather", "m"), "disruption_active": ("alt", "disruption_events", "0/1"),
    "disruption_hours_elapsed": ("alt", "disruption_events", "h"), "holiday_local": ("alt", "public_holidays", "0/1"),
    "local_hour": ("alt", "metadata timezone", "hour"),
}


def _within(df, col):
    return df[col] - df.groupby("port_id")[col].transform("mean")


def features(df, statuses):
    rows = []
    d = df.dropna(subset=["y_next"])
    for col, (origin, source, unit) in CANDIDATES.items():
        cols = [f"labour_{k}" for k in ("reduced", "strike")] if col == "labour_status" else [col]
        for c in cols:
            x = d[c].astype(float)
            st = statuses.get(col)
            r = {"feature": c, "origin": origin, "source": source, "unit": unit,
                 "missing": int(df[c].isna().sum()), "ports_with_data": int(df.groupby("port_id")[c].apply(lambda s: s.notna().any()).sum()),
                 "reconstructed_share": float((st == "reconstructed").mean()) if st is not None else np.nan,
                 "min": x.min(), "median": x.median(), "max": x.max(), "std": x.std(),
                 "within_port_std_share": float(_within(d, c).std() / x.std()) if x.std() > 0 else np.nan,
                 "lag1_autocorr_within_port": float(d.groupby("port_id")[c].apply(lambda s: s.autocorr(1)).mean()),
                 "corr_with_next_dwell_pooled": x.corr(d["y_next"]),
                 "corr_with_next_dwell_within_port": _within(d, c).corr(_within(d, "y_next")),
                 "corr_with_next_hour_change_within_port": _within(d, c).corr(d["dy_next"])}
            # partial correlation with next-hour dwell given current dwell (within port)
            yw, xw, cw = _within(d, "y_next"), _within(d, c), _within(d, TARGET)
            if c != TARGET:
                ry = yw - np.polyfit(cw, yw, 1)[0] * cw
                rx = xw - np.polyfit(cw, xw, 1)[0] * cw
                r["partial_corr_given_current_dwell"] = float(np.corrcoef(rx, ry)[0, 1])
            # strongest lagged within-port correlation with the next-hour change, lags 0..24
            best = (0.0, 0)
            for lag in range(0, 25):
                xl = d.groupby("port_id")[c].shift(lag)
                ok = xl.notna()
                v = _within(d.assign(_x=xl)[ok], "_x").corr(d["dy_next"][ok]) if ok.sum() > 100 else np.nan
                if np.isfinite(v) and abs(v) > abs(best[0]):
                    best = (float(v), lag)
            r["max_abs_lagged_corr_with_change"], r["at_lag_h"] = best
            rows.append(r)
    out = save(pd.DataFrame(rows), "feature_suitability")
    redundancy = d[["throughput_teu", "port_utilization_pct"]].join(df[["port_id"]])
    util_check = []
    meta = pd.read_csv(PROCESSED / "port_metadata.csv").set_index("port_id")["capacity_teu_per_hour"]
    for p, g in redundancy.groupby("port_id"):
        util_check.append({"port_id": p, "max_abs_util_minus_throughput_over_capacity_pp":
                           float((g["port_utilization_pct"] - g["throughput_teu"] / meta[p] * 100).abs().max()),
                           "share_hours_at_100pct": float((g["port_utilization_pct"] >= 99.99).mean())})
    save(pd.DataFrame(util_check), "utilization_vs_throughput")
    corr = d[[c for c in CANDIDATES if c != "labour_status"] + ["labour_reduced", "labour_strike"]].corr()
    save(corr, "feature_correlation_matrix", index=True)
    return out


# ------------------------------------------------------------------------------------------
# E. Joins and graph
# ------------------------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def joins_and_graph(tables, df):
    meta = tables["port_metadata"].set_index("port_id")
    wx = tables["weather_hourly"]
    g = wx.groupby("port_id")[["grid_latitude", "grid_longitude"]].agg(["min", "max"])
    rows = []
    for p in meta.index:
        lat, lon = g.loc[p, ("grid_latitude", "min")], g.loc[p, ("grid_longitude", "min")]
        rows.append({"port_id": p, "timezone": meta.loc[p, "local_timezone"], "capacity_teu_per_hour": meta.loc[p, "capacity_teu_per_hour"],
                     "grid_points": int((g.loc[p, ("grid_latitude", "max")] != lat) + 1),
                     "grid_to_port_km": float(haversine(meta.loc[p, "latitude"], meta.loc[p, "longitude"], lat, lon)),
                     "gust_below_speed_hours": int(((wx["port_id"] == p) & (wx["wind_gusts_10m_kmh"] < wx["wind_speed_10m_kmh"])).sum()),
                     "visibility_missing_share": float(wx.loc[wx["port_id"] == p, "visibility_m"].isna().mean())})
    wxj = save(pd.DataFrame(rows), "weather_join_by_port")

    net = tables["port_network"]
    ports = list(meta.index)
    direct = net[net["route_type"] == "direct"]
    graph = {"nodes": len(ports), "directed_routes": len(net), "ordered_pairs_possible": len(ports) * (len(ports) - 1),
             "complete_directed_graph": bool(len(net.drop_duplicates(["origin_port_id", "destination_port_id"])) == len(ports) * (len(ports) - 1)),
             "direct_routes": len(direct), "transshipment_routes": int((net["route_type"] != "direct").sum()),
             "hubs": sorted(net["transshipment_port_id"].dropna().unique().tolist()),
             "direct_out_degree": direct.groupby("origin_port_id").size().to_dict(),
             "direct_symmetric_share": float(np.mean([((direct["origin_port_id"] == b) & (direct["destination_port_id"] == a)).any()
                                                     for a, b in zip(direct["origin_port_id"], direct["destination_port_id"])])),
             "distance_km_range": [float(net["route_distance_km"].min()), float(net["route_distance_km"].max())]}
    # cross-port dependence of dwell: is there anything a neighbour feature could carry?
    wide = df.pivot(index="ts", columns="port_id", values=TARGET)
    chg = wide.diff()
    lvl_c = wide.corr().where(~np.eye(len(ports), dtype=bool)).stack()
    chg_c = chg.corr().where(~np.eye(len(ports), dtype=bool)).stack()
    lagged = []
    for a in ports:
        for b in ports:
            if a != b:
                for lag in (1, 6, 24):
                    lagged.append(abs(chg[a].shift(lag).corr(chg[b])))
    resid = (wide - wide.rolling(24, min_periods=12).mean())
    graph.update({"dwell_level_corr_between_ports_median": float(lvl_c.median()), "dwell_level_corr_max_abs": float(lvl_c.abs().max()),
                  "dwell_change_corr_between_ports_max_abs": float(chg_c.abs().max()),
                  "dwell_change_lagged_corr_max_abs_lags_1_6_24": float(np.nanmax(lagged)),
                  "dwell_deviation_from_24h_mean_corr_median": float(resid.corr().where(~np.eye(len(ports), dtype=bool)).stack().median())})
    save(wide.corr(), "cross_port_dwell_level_corr", index=True)
    return wxj, graph


# ------------------------------------------------------------------------------------------
# Sequences
# ------------------------------------------------------------------------------------------
def windows(df, tel_status, wx_status, feature_cols, window=WINDOW, splits=(0.70, 0.85)):
    """Count samples (port, t): inputs t-window+1..t, target t+1, under different lineage policies."""
    hours = np.unique(df["ts"].values)
    cut1, cut2 = hours[int(len(hours) * splits[0])], hours[int(len(hours) * splits[1])]
    tel_cols = [c for c in feature_cols if c in tel_status.columns]
    wx_cols = [c for c in feature_cols if c in wx_status.columns]
    recon_in = ((tel_status[tel_cols] == "reconstructed").any(axis=1) | (wx_status[wx_cols] == "reconstructed").any(axis=1)).values
    recon_y = (tel_status[TARGET] == "reconstructed").values
    rows = []
    for p, g in df.groupby("port_id"):
        idx = g.index.values
        ts = g["ts"].values
        contiguous = np.r_[True, np.diff(ts) == np.timedelta64(1, "h")]
        n = len(idx)
        for k in range(window - 1, n - 1):
            span = slice(k - window + 1, k + 2)
            if not contiguous[k - window + 2:k + 2].all():
                continue
            t_target = ts[k + 1]
            t_first = ts[k - window + 1]
            split = "train" if t_target < cut1 else ("val" if t_target < cut2 else "test")
            # embargo: the window must not reach back across a split boundary
            crosses = (split == "val" and t_first < cut1) or (split == "test" and t_first < cut2)
            rows.append({"port_id": p, "t": ts[k], "split": split, "crosses_boundary": crosses,
                         "target_reconstructed": recon_y[idx[k + 1]], "inputs_reconstructed_hours": int(recon_in[idx[span][:-1]].sum())})
    w = pd.DataFrame(rows)
    return w, {"train_end": str(cut1), "val_end": str(cut2), "hours": len(hours)}


def sequence_counts(df, tel_status, wx_status, feature_cols, tag):
    out = []
    for L in (6, 12, 24, 48):
        w, cuts = windows(df, tel_status, wx_status, feature_cols, window=L)
        policies = {
            "A all processed hours (no lineage filter)": np.ones(len(w), bool),
            "B observed target (inputs may be reconstructed, masked)": ~w["target_reconstructed"].values,
            "C observed target and all inputs observed": (~w["target_reconstructed"] & (w["inputs_reconstructed_hours"] == 0)).values,
        }
        for pol, m in policies.items():
            ok = m & ~w["crosses_boundary"].values
            row = {"window_h": L, "policy": pol, "total": int(ok.sum())}
            for s in ("train", "val", "test"):
                row[s] = int((ok & (w["split"] == s).values).sum())
            row["dropped_by_embargo"] = int((m & w["crosses_boundary"].values).sum())
            out.append(row)
        if L == WINDOW:
            per_port = w.assign(B=~w["target_reconstructed"], C=~w["target_reconstructed"] & (w["inputs_reconstructed_hours"] == 0))
            per_port = per_port[~per_port["crosses_boundary"]].groupby(["port_id", "split"]).agg(
                policy_A=("t", "size"), policy_B=("B", "sum"), policy_C=("C", "sum")).unstack("split")
            per_port.columns = [f"{a}_{b}" for a, b in per_port.columns]
            save(per_port.reset_index(), f"sequence_counts_by_port_w12_{tag}")
            split_cuts = cuts
    return save(pd.DataFrame(out), f"sequence_counts_{tag}"), split_cuts


def baselines(df, tel_status, split_cuts):
    """Error of fixed-formula next-hour forecasts on the chronological test hours (no fitting on test).
    These are reference floors for any DL model, not models."""
    d = df.copy()
    d["target_observed"] = ~(tel_status[TARGET] == "reconstructed").shift(-1, fill_value=True).values
    g = d.groupby("port_id")[TARGET]
    preds = {"persistence (y_t)": d[TARGET]}
    for k in (3, 6, 12, 24):
        preds[f"trailing mean {k} h"] = g.transform(lambda s: s.rolling(k, min_periods=k).mean())
    train_end = pd.Timestamp(split_cuts["train_end"]).tz_localize("UTC")
    val_end = pd.Timestamp(split_cuts["val_end"]).tz_localize("UTC")
    tr = d["ts"] < train_end
    preds["port mean of training period"] = d["port_id"].map(d[tr].groupby("port_id")[TARGET].mean())
    # EWMA smoothing factor chosen on the training period only
    best = None
    for a in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        e = g.transform(lambda s: s.ewm(alpha=a, adjust=False).mean())
        err = ((e - d["y_next"])[tr]).pow(2).mean()
        best = (err, a, e) if best is None or err < best[0] else best
    preds[f"EWMA alpha={best[1]} (chosen on train)"] = best[2]
    test = (d["ts"] >= val_end) & d["y_next"].notna()
    rows = []
    for name, p in preds.items():
        for subset, m in [("all test targets", test), ("observed test targets only", test & d["target_observed"])]:
            e = (p - d["y_next"])[m].dropna()
            rows.append({"baseline": name, "subset": subset, "n": len(e), "mae": e.abs().mean(), "rmse": np.sqrt((e ** 2).mean())})
            if subset == "observed test targets only":
                for port, ep in e.groupby(d.loc[e.index, "port_id"]):
                    rows.append({"baseline": name, "subset": f"port {port}", "n": len(ep), "mae": ep.abs().mean(), "rmse": np.sqrt((ep ** 2).mean())})
    out = save(pd.DataFrame(rows), "baseline_floors")

    # partial correlation with next-hour dwell given the 12 h trailing mean (within port, training period)
    d["dwell_mean12"] = preds["trailing mean 12 h"]
    t = d[tr & d["y_next"].notna() & d["dwell_mean12"].notna()]
    w = lambda c: t[c] - t.groupby("port_id")[c].transform("mean")  # noqa: E731
    base = w("dwell_mean12")
    ry = w("y_next") - np.polyfit(base, w("y_next"), 1)[0] * base
    pc = []
    for c in [TARGET, "throughput_teu", "port_utilization_pct", "customs_hold_hours", "temperature_2m_c", "wind_speed_10m_kmh",
              "labour_reduced", "labour_strike", "equipment_availability_pct", "vessels_at_anchor", "vessel_time_in_port_hours",
              "wind_gusts_10m_kmh", "precipitation_mm", "disruption_active", "disruption_hours_elapsed", "holiday_local"]:
        x = w(c)
        rx = x - np.polyfit(base, x, 1)[0] * base
        pc.append({"feature": c, "partial_corr_given_12h_trailing_dwell_mean": float(np.corrcoef(rx, ry)[0, 1])})
    save(pd.DataFrame(pc), "partial_corr_given_trailing_mean")
    return out


def alignment_example(df, port="SGSIN", t_index=200):
    g = df[df["port_id"] == port].reset_index(drop=True)
    rows = g.iloc[t_index - WINDOW + 1:t_index + 2][["ts_hour", TARGET, "port_utilization_pct", "vessels_at_anchor"]].copy()
    rows["role"] = ["input (t-%d)" % (WINDOW - 1 - i) if i < WINDOW - 1 else "input (t)" for i in range(WINDOW)] + ["target (t+1)"]
    return save(rows, "alignment_example")


# ------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------
def figures(df, tel_status, tables):
    df = df.assign(recon=(tel_status[TARGET] == "reconstructed").values)
    ports = sorted(df["port_id"].unique())
    ev = tables["disruption_events"]
    fig, axes = plt.subplots(len(ports), 1, figsize=(14, 2.0 * len(ports)), sharex=True)
    for ax, p in zip(axes, ports):
        g = df[df["port_id"] == p]
        ax.plot(g["ts"], g[TARGET], lw=0.7, color="#2a78d6")
        r = g[g["recon"]]
        ax.scatter(r["ts"], r[TARGET], s=6, color="#e34948", zorder=3, label="reconstructed (not in raw)")
        for _, e in ev[ev["port_id"] == p].iterrows():
            ax.axvspan(pd.Timestamp(e["event_start"]), pd.Timestamp(e["event_end"]), color="#eda100", alpha=0.18, lw=0)
        ax.set_ylabel(p, rotation=0, ha="right", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("container_dwell_hours by port (orange bands: disruption events; red dots: values with no raw observation)")
    fig.tight_layout()
    fig.savefig(OUT["figures"] / "dwell_series_by_port.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    lags = range(1, 49)
    for p in ports:
        y = df.loc[df["port_id"] == p, TARGET].values
        axes[0].plot(lags, acf(y, lags), lw=1, label=p)
        d = np.diff(y)
        axes[1].plot(lags, acf(d, lags), lw=1)
    axes[0].set_title("Autocorrelation of dwell level")
    axes[1].set_title("Autocorrelation of hourly dwell change")
    for ax in axes:
        ax.set_xlabel("lag (h)")
        ax.axhline(0, color="#0b0b0b", lw=0.7)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT["figures"] / "dwell_autocorrelation.png", dpi=130)
    plt.close(fig)

    day = df.assign(day=df["ts"].dt.floor("D")).groupby(["port_id", "day"])["recon"].sum().unstack()
    fig, ax = plt.subplots(figsize=(14, 3.6))
    im = ax.imshow(day.values, aspect="auto", cmap="Reds")
    ax.set_yticks(range(len(day.index)), day.index)
    ax.set_xticks(range(0, day.shape[1], 3), [str(d.date()) for d in day.columns[::3]], rotation=60, fontsize=7)
    ax.set_title("Reconstructed dwell values per port and day (count of hours)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(OUT["figures"] / "dwell_reconstructed_hours_heatmap.png", dpi=130)
    plt.close(fig)


def dump_json(obj, name):
    (OUT["tables"] / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))


__all__ = ["PPT_SIGNALS"]


def event_coverage(tables, split_cuts):
    """Disruption-event hours and holiday days falling in each chronological split (temporal diversity)."""
    c1 = pd.Timestamp(split_cuts["train_end"]).tz_localize("UTC")
    c2 = pd.Timestamp(split_cuts["val_end"]).tz_localize("UTC")
    ev = tables["disruption_events"]
    rows = []
    for _, e in ev.iterrows():
        for h in pd.date_range(pd.Timestamp(e["event_start"]), pd.Timestamp(e["event_end"]), freq="h", inclusive="left"):
            rows.append({"port_id": e["port_id"], "event_type": e["event_type"],
                         "split": "train" if h < c1 else ("val" if h < c2 else "test")})
    d = pd.DataFrame(rows)
    by_type = pd.crosstab(d["event_type"], d["split"]).reindex(columns=["train", "val", "test"], fill_value=0)
    by_port = pd.crosstab(d["port_id"], d["split"]).reindex(columns=["train", "val", "test"], fill_value=0)
    save(by_type.reset_index(), "disruption_hours_by_type_and_split")
    save(by_port.reset_index(), "disruption_hours_by_port_and_split")
    return by_type, by_port
