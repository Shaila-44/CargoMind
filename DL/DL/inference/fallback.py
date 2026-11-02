"""EWMA baseline forecast (alpha 0.2): CargoMind's operational fallback. Not a neural model.

Uses the dataset stage's own EWMA definition (cargomind_dl.baselines.ewma_window: initialised at the
first of the 12 hours, alpha = 0.2), so a fallback forecast equals the EWMA baseline reported in all
DL evaluations.
"""
from __future__ import annotations

import numpy as np

from cargomind_dl.baselines import ewma_window  # noqa: E402

from .validation import validate_dwell_history  # noqa: E402

EWMA_ALPHA = 0.2


def ewma_forecast(dwell_hours, alpha: float = EWMA_ALPHA) -> float:
    """Next-hour dwell forecast (hours) from the last 12 raw dwell hours, oldest first."""
    v = validate_dwell_history(dwell_hours)
    return float(ewma_window(v[None, :], alpha)[0])


__all__ = ["ewma_forecast", "EWMA_ALPHA", "np"]
