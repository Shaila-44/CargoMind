"""DL service: next-hour port dwell-time forecasting.

A thin adapter over the DL workflow's inference interface (`dl.inference`, in DL/DL/inference). It adds no
model, preprocessing or validation logic of its own: input checks, scaling, model loading, inverse scaling and
the explicit EWMA fallback all live in the DL workflow.
"""
from __future__ import annotations

from dl.inference import (DEFAULT_MODEL, REQUIRED_COLUMNS, DwellPrediction, InputValidationError, ModelLoadError, ewma_forecast,
                          forecast, load_registry)

__all__ = ["DEFAULT_MODEL", "REQUIRED_COLUMNS", "DwellPrediction", "InputValidationError", "ModelLoadError", "ewma_forecast",
           "forecast", "available_models"]


def available_models() -> dict[str, dict]:
    """Registered neural models (name -> status, checkpoint, validation RMSE). EWMA is available as model='ewma'."""
    return {name: {k: entry[k] for k in ("architecture", "status", "checkpoint", "val_rmse_h_5_seed_mean")}
            for name, entry in load_registry()["models"].items()}
