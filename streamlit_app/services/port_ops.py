"""Port operations service: read-only views of the synthetic port history for the overview dashboard.

Everything here is observed synthetic data (the DL hourly panel, port metadata and historical shipment outcomes),
replayed at a chosen hour. It runs no model and invents no live status: there is no live feed.

    from services import port_ops
    snap = port_ops.snapshot("2025-02-03 19:00+00:00")   # one row per port at that hour
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from cargomind_dl.config import DATASET
from cargomind_eda.config import PROCESSED

__all__ = ["panel", "ports", "port_names", "snapshot", "daily_trend", "shipment_outcomes", "active_events"]


@lru_cache(maxsize=1)
def port_names() -> dict[str, str]:
    """Port id -> display name, from the port metadata alone (no dependency on the DL hourly panel)."""
    meta = pd.read_csv(PROCESSED / "port_metadata.csv", usecols=["port_id", "port_name"])
    return dict(zip(meta["port_id"], meta["port_name"]))


@lru_cache(maxsize=1)
def panel() -> pd.DataFrame:
    """The DL hourly panel (one row per port-hour, raw units)."""
    p = pd.read_csv(DATASET / "hourly_panel.csv",
                    usecols=["port_id", "ts_hour", "dwell", "utilization", "labour_strike", "wind_gust",
                             "disruption_active", "vessels_at_anchor"])
    p["ts_hour"] = pd.to_datetime(p["ts_hour"], utc=True)
    return p


@lru_cache(maxsize=1)
def ports() -> pd.DataFrame:
    """Port names and coordinates (synthetic metadata), limited to ports in the panel."""
    meta = pd.read_csv(PROCESSED / "port_metadata.csv",
                       usecols=["port_id", "port_name", "region", "latitude", "longitude", "capacity_teu_per_hour"])
    return meta[meta["port_id"].isin(panel()["port_id"].unique())].reset_index(drop=True)


def snapshot(hour_utc) -> pd.DataFrame:
    """Each port's observed state at one hour, joined with its name and location."""
    ts = pd.Timestamp(hour_utc)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    rows = panel()[panel()["ts_hour"] == ts]
    return ports().merge(rows, on="port_id", how="inner")


def daily_trend(end_hour_utc, days: int = 7, port_id: str | None = None) -> pd.DataFrame:
    """Daily mean dwell and mean vessels at anchor for the `days` days ending at `end_hour_utc` (inclusive)."""
    end = pd.Timestamp(end_hour_utc)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    p = panel()
    p = p[(p["ts_hour"] > end - pd.Timedelta(days=days)) & (p["ts_hour"] <= end)]
    if port_id:
        p = p[p["port_id"] == port_id]
    daily = p.groupby(p["ts_hour"].dt.floor("D")).agg(dwell=("dwell", "mean"), anchored=("vessels_at_anchor", "mean"))
    return daily.reset_index().rename(columns={"ts_hour": "day"})


@lru_cache(maxsize=1)
def _shipments() -> pd.DataFrame:
    s = pd.read_csv(PROCESSED / "shipment_records.csv", usecols=["origin_port_id", "actual_arrival_ts", "delay_status"])
    s["actual_arrival_ts"] = pd.to_datetime(s["actual_arrival_ts"], utc=True, format="ISO8601")
    return s


def shipment_outcomes(end_hour_utc, days: int = 7, origin_port_id: str | None = None) -> pd.Series:
    """Counts of recorded delay outcomes for shipments that arrived in the `days` days before `end_hour_utc`.

    These are historical outcomes from the synthetic records, not model predictions.
    """
    end = pd.Timestamp(end_hour_utc)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    s = _shipments()
    s = s[(s["actual_arrival_ts"] > end - pd.Timedelta(days=days)) & (s["actual_arrival_ts"] <= end)]
    if origin_port_id:
        s = s[s["origin_port_id"] == origin_port_id]
    return s["delay_status"].value_counts().reindex(["On-Time", "At-Risk", "Critical"], fill_value=0)


@lru_cache(maxsize=1)
def _events() -> pd.DataFrame:
    e = pd.read_csv(PROCESSED / "disruption_events.csv",
                    usecols=["event_id", "port_id", "event_start", "event_end", "event_type", "severity",
                             "affected_capacity_pct", "event_description"])
    for col in ("event_start", "event_end"):
        e[col] = pd.to_datetime(e[col], utc=True, format="ISO8601")
    return e


def active_events(hour_utc) -> pd.DataFrame:
    """Disruption events (synthetic event records) in progress at `hour_utc`, with hours elapsed since they started.

    The event end time is used only to decide whether an event is still in progress; it is not returned, because it
    would not be known at the replay hour.
    """
    ts = pd.Timestamp(hour_utc)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    e = _events()
    live = e[(e["event_start"] <= ts) & (e["event_end"] > ts)].drop(columns="event_end").copy()
    live["hours_elapsed"] = ((ts - live["event_start"]).dt.total_seconds() / 3600).astype(int)
    return live.reset_index(drop=True)
