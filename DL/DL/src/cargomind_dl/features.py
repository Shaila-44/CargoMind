"""Hourly feature panel: one row per (port, hour), unscaled, with cell-level provenance.

Every value in a row describes hour t and is known once hour t has closed:
* telemetry and weather are measurements of hour t;
* disruption state uses only whether an event has started by t and has not ended by t. An event's
  end time is used solely to say "still ongoing at t" - once an event has ended, that fact is in
  the past. Severity, affected capacity, planned/remaining duration and future events are never used.
"""
import numpy as np
import pandas as pd

from . import config


def _read(name):
    return pd.read_csv(config.PROCESSED / f"{name}.csv")


def load_provenance():
    """Audit lineage files, keyed by (port_id, ts_hour). Raises if the audit has not been run."""
    out = {}
    for table in ("port_hourly_telemetry", "weather_hourly"):
        path = config.AUDIT_LINEAGE / f"cell_lineage_{table}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run dl/data_audit/scripts/run_data_audit.py first")
        out[table] = pd.read_csv(path).set_index(["port_id", "ts_hour"])
    return out


def disruption_state(panel, events):
    """disruption_active / hours_elapsed / end_reconstructed for each (port, hour).

    Overlapping events at the same port: active if any is ongoing; elapsed = hours since the
    earliest start among the ongoing events (the disruption episode began then).
    """
    active = np.zeros(len(panel), bool)
    elapsed = np.zeros(len(panel))
    end_recon = np.zeros(len(panel), bool)
    ts = panel["ts"].values
    for _, e in events.iterrows():
        start, end = e["start"].to_datetime64(), e["end"].to_datetime64()
        m = (panel["port_id"].values == e["port_id"]) & (ts >= start) & (ts < end)
        h = (ts[m] - start) / np.timedelta64(1, "h")
        elapsed[m] = np.where(active[m], np.maximum(elapsed[m], h), h)
        active |= m
        end_recon |= m & e["end_reconstructed"]
    return active.astype(np.int8), elapsed, end_recon


def events_with_provenance():
    ev = _read("disruption_events")
    # naive UTC so they compare with numpy datetime64 hour stamps
    ev["start"] = pd.to_datetime(ev["event_start"], utc=True).dt.tz_convert(None)
    ev["end"] = pd.to_datetime(ev["event_end"], utc=True).dt.tz_convert(None)
    raw = pd.read_csv(config.RAW / "disruption_events.csv", dtype=str, keep_default_na=False)
    missing_end = set(raw.loc[raw["event_end"].str.strip().isin(["", "N/A", "NaN", "-"]), "event_id"].str.strip())
    ev["end_reconstructed"] = ev["event_id"].isin(missing_end)
    return ev


def build_panel():
    tel = _read("port_hourly_telemetry")
    wx = _read("weather_hourly")[["port_id", "ts_hour", "wind_gusts_10m_kmh"]]
    df = tel.merge(wx, on=["port_id", "ts_hour"], how="inner", validate="one_to_one")
    assert len(df) == len(tel) == len(wx), "telemetry and weather keys must match one-to-one"
    df["ts"] = pd.to_datetime(df["ts_hour"], utc=True)
    df = df.sort_values(["port_id", "ts"]).reset_index(drop=True)

    hours = np.unique(df["ts"].values)
    df["hour_index"] = np.searchsorted(hours, df["ts"].values)
    per_port = df.groupby("port_id")["hour_index"].agg(["min", "max", "size"])
    assert (per_port["min"] == 0).all() and (per_port["max"] == len(hours) - 1).all() and (per_port["size"] == len(hours)).all(), \
        "every port must cover the same contiguous hourly grid"
    assert (np.diff(hours) == np.timedelta64(1, "h")).all(), "hourly grid must be regular"

    ev = events_with_provenance()
    active, elapsed, end_recon = disruption_state(df, ev)
    panel = pd.DataFrame({
        "port_id": df["port_id"], "ts_hour": df["ts_hour"], "ts": df["ts"], "hour_index": df["hour_index"],
        "dwell": df[config.TARGET].astype(float),
        "utilization": df["port_utilization_pct"].astype(float),
        "labour_strike": (df["labour_status"] == "strike").astype(np.int8),
        # 2 hours have small negative gusts (data error); a gust cannot be negative
        "wind_gust": df["wind_gusts_10m_kmh"].clip(lower=0).astype(float),
        "disruption_active": active,
        "disruption_hours_elapsed": elapsed,
        "vessels_at_anchor": df["vessels_at_anchor"].astype(float),
        "disruption_end_reconstructed": end_recon.astype(np.int8),
        "wind_gust_clipped": (df["wind_gusts_10m_kmh"] < 0).astype(np.int8),
    })

    prov = load_provenance()
    key = pd.MultiIndex.from_arrays([panel["port_id"], panel["ts_hour"]])
    for feat, (table, col) in config.PROVENANCE_COLUMNS.items():
        status = prov[table][f"{col}__status"].reindex(key)
        assert status.notna().all(), f"provenance missing for {feat}"
        panel[f"prov_{feat}"] = status.map(config.STATUS_CODES).astype(np.int8).values
    return panel, ev, hours
