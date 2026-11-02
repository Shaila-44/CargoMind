"""Window-level context for evaluation: joins a split's windows to windows.csv (baseline predictions,
reconstruction counts) and hourly_panel.csv (disruption / strike state at hour t). Read-only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cargomind_dl.config import DATASET  # noqa: E402  (DL/Dataset/processed)
BASELINES = {"persistence": "baseline_persistence_hours", "12 h mean": "baseline_trailing_mean_12h_hours",
             "EWMA α=0.2": "baseline_ewma_alpha_0.2_hours"}


def context(window_id: np.ndarray, y_hours: np.ndarray) -> pd.DataFrame:
    w = pd.read_csv(DATASET / "windows.csv").set_index("window_id").loc[window_id].reset_index()
    if not np.allclose(w["target_hours"].values, y_hours, atol=1e-3):
        raise ValueError("windows.csv targets do not match the loaded split")
    panel = pd.read_csv(DATASET / "hourly_panel.csv").set_index(["port_id", "hour_index"])
    at_t = panel.loc[list(zip(w["port_id"], w["t_hour_index"])), ["disruption_active", "disruption_hours_elapsed", "labour_strike"]]
    w["disruption_active_t"] = at_t["disruption_active"].values
    w["disruption_hours_elapsed_t"] = at_t["disruption_hours_elapsed"].values
    w["labour_strike_t"] = at_t["labour_strike"].values
    w["y_hours"] = y_hours
    return w


def baseline_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: frame[col].to_numpy(float) for name, col in BASELINES.items()}
