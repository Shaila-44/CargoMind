"""DL service: next-hour port dwell-time forecasting.

A thin adapter over the DL workflow's inference interface (`dl.inference`, in DL/DL/inference). It adds no
model, preprocessing or validation logic of its own: input checks, scaling, model loading, inverse scaling and
the automatic fallback all live in the DL workflow.

    from services import dl_forecasting
    window = dl_forecasting.window_from_dataset("SGSIN", "2025-02-03 19:00+00:00")  # 12 raw hours from the DL dataset
    result = dl_forecasting.forecast(window, port_id="SGSIN")      # DwellPrediction; result.to_dict() for JSON

Production configuration (DL/DL/inference/model_registry.json): Transformer T_w (`transformer_t_w`) is the primary model;
if it cannot produce a valid prediction, the C_w LSTM (`c_w_lstm`) is used automatically and the result reports
`fallback_used=True`, `model_name="c_w_lstm"` and the reason. Invalid input windows raise InputValidationError.
EWMA (alpha 0.2) stays available as a benchmark (`model="ewma"`). The GNN is experimental and not available here.
"""
from __future__ import annotations

import pandas as pd

from cargomind_dl.config import DATASET
from dl.inference import (CONTRACT_VERSION, DEFAULT_MODEL, FALLBACK_MODEL, REQUIRED_COLUMNS, DwellPrediction, InputValidationError,
                          ModelLoadError, PredictionError, ewma_forecast, forecast, load_registry)
from dl.inference.windows import actual_dwell, coverage, window_from_dataset

__all__ = ["DEFAULT_MODEL", "FALLBACK_MODEL", "CONTRACT_VERSION", "REQUIRED_COLUMNS", "DwellPrediction", "InputValidationError",
           "ModelLoadError", "PredictionError", "ewma_forecast", "forecast", "available_models", "window_from_dataset", "actual_dwell",
           "coverage", "supported_ports"]


def supported_ports() -> list[str]:
    """Port ids present in the DL hourly panel (the only ports a dwell window can be built for)."""
    return sorted(pd.read_csv(DATASET / "hourly_panel.csv", usecols=["port_id"])["port_id"].unique().tolist())


def available_models() -> dict[str, dict]:
    """Registered neural models (name -> role, status, checkpoint, validation RMSE). EWMA is available as model='ewma'."""
    return {name: {k: entry[k] for k in ("architecture", "role", "status", "checkpoint", "val_rmse_h_5_seed_mean")}
            for name, entry in load_registry()["models"].items()}
