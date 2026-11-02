"""Build a raw 12-hour input window from the DL dataset's hourly panel (DL/Dataset/processed/hourly_panel.csv).

    from dl.inference.windows import window_from_dataset
    window = window_from_dataset("SGSIN", "2025-02-03 19:00:00+00:00")   # hours t-11 .. t, oldest first
    result = forecast(window, "SGSIN")

The panel is the hourly table the DL dataset was built from (raw units, one row per port-hour), so a window taken from
it is exactly what training and evaluation used. It covers 2025-01-01 00:00 .. 2025-02-11 15:00 UTC; there is no live
feed, so hours outside that range are rejected. `dwell_reconstructed` is derived from the panel's provenance code
(prov_dwell == 2: not observed in raw, filled during cleaning), as in the dataset build.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from cargomind_dl.config import DATASET

from .validation import REQUIRED_COLUMNS, WINDOW, InputValidationError


@lru_cache(maxsize=1)
def _panel() -> pd.DataFrame:
    p = pd.read_csv(DATASET / "hourly_panel.csv")
    p["ts_hour"] = pd.to_datetime(p["ts_hour"], utc=True)
    return p


def coverage() -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last hour available in the panel (UTC)."""
    p = _panel()
    return p["ts_hour"].min(), p["ts_hour"].max()


def window_from_dataset(port_id: str, window_end_utc) -> pd.DataFrame:
    """The 12 raw hourly rows ending at `window_end_utc` (inclusive) for `port_id`, in REQUIRED_COLUMNS order."""
    p = _panel()
    if port_id not in set(p["port_id"]):
        raise InputValidationError(f"unknown port_id {port_id!r}; expected one of {sorted(p['port_id'].unique())}")
    try:
        end = pd.Timestamp(window_end_utc)
    except (ValueError, TypeError) as e:
        raise InputValidationError(f"window_end_utc could not be parsed: {e}") from e
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    start = end - pd.Timedelta(hours=WINDOW - 1)
    first, last = coverage()
    if start < first or end > last:
        raise InputValidationError(f"window {start.isoformat()} .. {end.isoformat()} is outside the panel's coverage "
                                   f"{first.isoformat()} .. {last.isoformat()} (no live data feed)")
    rows = p[(p["port_id"] == port_id) & p["ts_hour"].between(start, end)].sort_values("ts_hour")
    if len(rows) != WINDOW:
        raise InputValidationError(f"the panel has {len(rows)} of the {WINDOW} hours {start.isoformat()} .. {end.isoformat()} for {port_id}")
    w = rows.assign(dwell_reconstructed=(rows["prov_dwell"] == 2).astype(int))
    return w[REQUIRED_COLUMNS].reset_index(drop=True)


def actual_dwell(port_id: str, hour_utc) -> float | None:
    """Observed dwell at an hour (for display next to a forecast), or None if the panel has no such hour."""
    p = _panel()
    ts = pd.Timestamp(hour_utc)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    v = p.loc[(p["port_id"] == port_id) & (p["ts_hour"] == ts), "dwell"]
    return float(v.iloc[0]) if len(v) else None
