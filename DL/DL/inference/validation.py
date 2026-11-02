"""Validation of raw (unscaled) 12-hour input windows. Invalid input raises; nothing is repaired or filled.

Expected input: a pandas DataFrame with one row per hour, oldest first, and these columns (raw units,
matching dl/dataset/feature_schema.json and the hourly panel used to build the dataset):

    ts_hour                   UTC timestamp of the hour (12 consecutive hours, strictly increasing)
    dwell                     container_dwell_hours, hours                         (> 0)
    utilization               port_utilization_pct, percent of capacity           [0, 100]
    labour_strike             1 if labour_status == "strike", else 0               {0, 1}
    wind_gust                 wind_gusts_10m_kmh, km/h                             [0, 300]
    disruption_active         1 if a disruption event is ongoing at the hour       {0, 1}
    disruption_hours_elapsed  hours since the ongoing disruption began (0 if none) >= 0
    vessels_at_anchor         vessels waiting at anchor                            integer >= 0
    dwell_reconstructed       1 if the dwell value was not observed but filled     {0, 1}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 12
NUMERIC_COLUMNS = ["dwell", "utilization", "labour_strike", "wind_gust", "disruption_active", "disruption_hours_elapsed",
                   "vessels_at_anchor", "dwell_reconstructed"]
REQUIRED_COLUMNS = ["ts_hour"] + NUMERIC_COLUMNS
BINARY = ["labour_strike", "disruption_active", "dwell_reconstructed"]
# Plausibility limits in raw units. They exist to catch wrong units or already-standardised input,
# not to bound legitimate operations: the dataset's dwell ranges 54.6-205.2 h, utilisation 9-100 %.
LIMITS = {"dwell": (1.0, 1000.0), "utilization": (0.0, 100.0), "wind_gust": (0.0, 300.0),
          "disruption_hours_elapsed": (0.0, 24.0 * 365), "vessels_at_anchor": (0.0, 1000.0)}


class InputValidationError(ValueError):
    """The input window cannot be used for a prediction. The message says why."""


def validate_window(window, port_id: str, known_ports: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Return a validated copy of the window (oldest first) and a list of non-fatal warnings."""
    if not isinstance(window, pd.DataFrame):
        raise InputValidationError(
            f"window must be a pandas DataFrame with named columns {REQUIRED_COLUMNS}; got {type(window).__name__}. "
            "Positional arrays are not accepted because feature order cannot be verified.")
    if not isinstance(port_id, str) or port_id not in known_ports:
        raise InputValidationError(f"unknown port_id {port_id!r}; expected one of {known_ports} (exact, case-sensitive)")
    missing = [c for c in REQUIRED_COLUMNS if c not in window.columns]
    if missing:
        raise InputValidationError(f"missing required columns: {missing}")
    if len(window) != WINDOW:
        raise InputValidationError(f"expected exactly {WINDOW} hourly rows, got {len(window)}")
    warnings = []
    extra = [c for c in window.columns if c not in REQUIRED_COLUMNS + ["port_id"]]
    if extra:
        warnings.append(f"ignored extra columns: {extra}")
    if "port_id" in window.columns and not (window["port_id"] == port_id).all():
        raise InputValidationError("the window's port_id column disagrees with the port_id argument")

    w = window[REQUIRED_COLUMNS].copy().reset_index(drop=True)
    try:
        ts = pd.to_datetime(w["ts_hour"], utc=True)
    except (ValueError, TypeError) as e:
        raise InputValidationError(f"ts_hour could not be parsed as timestamps: {e}") from e
    if ts.isna().any():
        raise InputValidationError("ts_hour contains missing timestamps")
    steps = ts.diff().iloc[1:]
    if (steps < pd.Timedelta(0)).all():
        raise InputValidationError("rows are in reverse time order; supply the 12 hours oldest first")
    if not (steps == pd.Timedelta(hours=1)).all():
        raise InputValidationError("ts_hour must be 12 consecutive hours, oldest first, exactly 1 h apart")
    if (ts.dt.minute != 0).any() or (ts.dt.second != 0).any():
        raise InputValidationError("ts_hour values must be on the hour")
    w["ts_hour"] = ts

    for c in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(w[c]) or pd.api.types.is_bool_dtype(w[c]):
            raise InputValidationError(f"column {c!r} must be numeric, got dtype {w[c].dtype}")
        v = w[c].to_numpy(dtype=float)
        if np.isnan(v).any():
            raise InputValidationError(f"column {c!r} contains missing values (NaN) at rows {np.flatnonzero(np.isnan(v)).tolist()}")
        if np.isinf(v).any():
            raise InputValidationError(f"column {c!r} contains infinite values")
        w[c] = v

    for c in BINARY:
        if not np.isin(w[c], [0.0, 1.0]).all():
            raise InputValidationError(f"column {c!r} must contain only 0 or 1")
    # already-standardised or wrong-unit input (checked before the range checks so the message is specific)
    if w["dwell"].max() < 10 or (w["dwell"] < 0).any():
        raise InputValidationError("dwell looks standardised or not in hours (values must be raw container dwell hours, > 1 h); "
                                   "this interface applies the scaling itself")
    if w["utilization"].max() <= 1.0:
        raise InputValidationError("utilization looks like a fraction or a scaled value; supply percent of capacity (0-100)")
    for c, (lo, hi) in LIMITS.items():
        bad = (w[c] < lo) | (w[c] > hi)
        if bad.any():
            raise InputValidationError(f"column {c!r} has values outside [{lo}, {hi}] at rows {np.flatnonzero(bad.to_numpy()).tolist()}")
    if not np.allclose(w["vessels_at_anchor"], np.round(w["vessels_at_anchor"])):
        raise InputValidationError("vessels_at_anchor must be whole numbers")
    inconsistent = ((w["disruption_active"] == 0) & (w["disruption_hours_elapsed"] > 0)).to_numpy()
    if inconsistent.any():
        raise InputValidationError(f"disruption_hours_elapsed > 0 while disruption_active == 0 at rows {np.flatnonzero(inconsistent).tolist()}")
    return w, warnings


def validate_dwell_history(dwell, window: int = WINDOW) -> np.ndarray:
    """Validation used by the EWMA fallback: exactly `window` raw dwell hours, oldest first, finite and positive."""
    v = np.asarray(dwell, dtype=float)
    if v.ndim != 1 or len(v) != window:
        raise InputValidationError(f"EWMA fallback needs exactly {window} dwell values (oldest first), got shape {v.shape}")
    if not np.isfinite(v).all():
        raise InputValidationError("EWMA fallback: dwell history contains NaN or infinite values")
    if v.max() < 10 or (v < LIMITS["dwell"][0]).any() or (v > LIMITS["dwell"][1]).any():
        raise InputValidationError("EWMA fallback: dwell values must be raw hours within [1, 1000] (not standardised)")
    return v
