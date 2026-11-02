"""Inference interface for CargoMind next-hour dwell forecasting (no training code is imported).

    from dl.inference import forecast
    result = forecast(window_df, port_id="SGSIN")                     # default research candidate (transformer_t_w)
    result = forecast(window_df, port_id="SGSIN", model="gru")        # explicit model choice
    result = forecast(window_df, port_id="SGSIN", fallback="ewma")    # EWMA only if the neural path fails
    result = forecast(window_df, port_id="SGSIN", model="ewma")       # the EWMA baseline itself

Pipeline: validate raw window -> apply the dataset's train-fitted transforms (cargomind_dl.sequences.transform,
saved scalers.json) -> select the checkpoint's channels by name, in the checkpoint's order -> model in eval mode
under torch.no_grad() -> invert with cargomind_dl.dataset.to_hours -> structured result.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cargomind_dl import config as data_config  # noqa: E402
from cargomind_dl import dataset as dl_dataset  # noqa: E402
from cargomind_dl.sequences import transform  # noqa: E402

from dl.models import load_checkpoint  # noqa: E402

from . import registry  # noqa: E402
from .fallback import EWMA_ALPHA, ewma_forecast  # noqa: E402
from .validation import InputValidationError, validate_window  # noqa: E402

RECONSTRUCTION_CHANNEL = "dwell_reconstructed"
TYPICAL_RANGE_Z = 4.0  # |z| of the latest dwell beyond this -> warning (outside the port's training-period spread)


class ModelLoadError(RuntimeError):
    """A requested model or its preprocessing artifacts are missing, altered or incompatible."""


@dataclass
class DwellPrediction:
    predicted_dwell_hours: float
    method: str                 # "neural" | "ewma_fallback" | "ewma_baseline"
    model_name: str             # registry name, or "ewma_alpha_0.2"
    model_version: str          # checkpoint path + short SHA-256, or the EWMA definition
    port_id: str
    window_end_utc: str | None  # last input hour (t)
    target_hour_utc: str | None  # the forecast hour (t + 1)
    warnings: list[str] = field(default_factory=list)
    note: str = "point forecast only; no calibrated uncertainty is provided"

    def to_dict(self) -> dict:
        return asdict(self)


class DwellForecaster:
    """One loaded neural model plus the exact preprocessing artifacts its checkpoint was trained with."""

    def __init__(self, checkpoint: Path, name: str, expected_sha256: str | None = None, dataset_dir: Path | None = None):
        checkpoint, dataset_dir = Path(checkpoint), Path(dataset_dir or data_config.DATASET)
        if not checkpoint.exists():
            raise ModelLoadError(f"checkpoint for model {name!r} not found: {checkpoint}")
        sha = registry.sha256(checkpoint)
        if expected_sha256 and sha != expected_sha256:
            raise ModelLoadError(f"checkpoint {checkpoint} does not match the registry (sha256 {sha[:12]} != {expected_sha256[:12]})")
        files = {}
        for fname in ("feature_schema.json", "scalers.json"):
            p = dataset_dir / fname
            if not p.exists():
                raise ModelLoadError(f"required preprocessing artifact missing: {p}")
            files[fname] = p
        try:
            model, meta = load_checkpoint(checkpoint)
        except (RuntimeError, KeyError, TypeError, ValueError) as e:
            raise ModelLoadError(f"could not load checkpoint {checkpoint}: {e}") from e
        for fname, p in files.items():
            ref = meta["dataset_reference"].get(fname, {}).get("sha256")
            if ref != registry.sha256(p):
                raise ModelLoadError(f"{p} is not the {fname} this checkpoint was trained with (incompatible schema or scalers)")
        self.schema = json.loads(files["feature_schema.json"].read_text())
        self.scalers = json.loads(files["scalers.json"].read_text())
        self.ports = self.schema["arrays"]["port_index"]["ports"]
        available = list(self.schema["arrays"]["X"]["axis_2"]) + [RECONSTRUCTION_CHANNEL]
        if list(self.schema["arrays"]["X"]["axis_2"]) != list(data_config.FEATURES):
            raise ModelLoadError("feature_schema.json channel order differs from the preprocessing code's feature order")
        channels = list(meta["input_channels"])
        if any(c not in available for c in channels) or channels != [c for c in available if c in channels]:
            raise ModelLoadError(f"checkpoint channels {channels} are not an ordered subset of the schema channels {available}")
        self.model, self.meta, self.channels, self.name = model, meta, channels, name
        try:
            shown = checkpoint.resolve().relative_to(registry.PROJECT_ROOT).as_posix()
        except ValueError:  # checkpoint outside the project: show its file name only
            shown = checkpoint.name
        self.version = f"{shown}@{sha[:12]}"

    def preprocess(self, window: pd.DataFrame, port_id: str) -> tuple[np.ndarray, list[str], pd.DataFrame]:
        """Validated raw window -> model input [1, 12, C] in this checkpoint's channel order."""
        w, warnings = validate_window(window, port_id, self.ports)
        panel = w.assign(port_id=port_id)
        scaled = transform(panel, self.scalers)  # the dataset build's own train-fitted transforms
        cols = {c: scaled[c].to_numpy(np.float32) for c in data_config.FEATURES}
        cols[RECONSTRUCTION_CHANNEL] = w[RECONSTRUCTION_CHANNEL].to_numpy(np.float32)
        x = np.stack([cols[c] for c in self.channels], axis=-1)[None]
        z_last = float(scaled["dwell"].iloc[-1])
        if abs(z_last) > TYPICAL_RANGE_Z:
            warnings.append(f"latest dwell is {z_last:+.1f} training SDs from this port's training mean; neural forecasts are least "
                            "reliable outside the training range (see DL/Evaluation/reports/DL_final_report.md, limitations)")
        if w[RECONSTRUCTION_CHANNEL].any():
            warnings.append("input contains hours flagged as reconstructed dwell")
        return x, warnings, w

    def predict(self, window: pd.DataFrame, port_id: str) -> DwellPrediction:
        x, warnings, w = self.preprocess(window, port_id)
        port_index = np.array([self.ports.index(port_id)])
        self.model.eval()
        with torch.no_grad():
            z = self.model(torch.from_numpy(x), torch.from_numpy(port_index)).numpy()
        hours = float(dl_dataset.to_hours(z.astype(np.float64), port_index, self.scalers)[0])
        if not np.isfinite(hours):
            raise RuntimeError(f"model {self.name} produced a non-finite prediction")
        end = w["ts_hour"].iloc[-1]
        return DwellPrediction(predicted_dwell_hours=hours, method="neural", model_name=self.name, model_version=self.version,
                               port_id=port_id, window_end_utc=end.isoformat(), target_hour_utc=(end + pd.Timedelta(hours=1)).isoformat(),
                               warnings=warnings)


_CACHE: dict = {}


def load_model(name: str = registry.DEFAULT_MODEL, registry_path: Path | None = None, dataset_dir: Path | None = None) -> DwellForecaster:
    """Load a registered neural model by name. An unknown or unavailable model raises; no other model is substituted."""
    try:
        reg = registry.load_registry(registry_path or registry.REGISTRY_PATH)
    except FileNotFoundError as e:
        raise ModelLoadError(str(e)) from e
    if name not in reg["models"]:
        raise ModelLoadError(f"unknown model {name!r}; registered neural models: {sorted(reg['models'])} (EWMA: model='ewma')")
    entry = reg["models"][name]
    key = (name, str(registry_path), str(dataset_dir))
    if key not in _CACHE:
        _CACHE[key] = DwellForecaster(registry.PROJECT_ROOT / entry["checkpoint"], name, entry["checkpoint_sha256"], dataset_dir)
    return _CACHE[key]


def _ewma_result(window, port_id, method: str, warnings: list[str]) -> DwellPrediction:
    if not isinstance(window, pd.DataFrame) or "dwell" not in window.columns or "ts_hour" not in window.columns:
        raise InputValidationError("EWMA needs a DataFrame with 'ts_hour' and raw 'dwell' columns (12 hours, oldest first)")
    ts = pd.to_datetime(window["ts_hour"], utc=True, errors="coerce")
    if len(ts) != 12 or ts.isna().any() or not (ts.diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise InputValidationError("EWMA needs 12 consecutive hourly rows, oldest first")
    hours = ewma_forecast(window["dwell"].to_numpy(dtype=float, na_value=np.nan))
    end = ts.iloc[-1]
    return DwellPrediction(predicted_dwell_hours=hours, method=method, model_name="ewma_alpha_0.2",
                           model_version=f"EWMA over the 12 input hours, alpha {EWMA_ALPHA}, initialised at the first hour (baseline, not a neural model)",
                           port_id=port_id, window_end_utc=end.isoformat(), target_hour_utc=(end + pd.Timedelta(hours=1)).isoformat(),
                           warnings=warnings)


def forecast(window: pd.DataFrame, port_id: str, model: str = registry.DEFAULT_MODEL, fallback: str = "none",
             registry_path: Path | None = None, dataset_dir: Path | None = None) -> DwellPrediction:
    """Next-hour dwell forecast in hours.

    model:    a registered neural model name, or "ewma" for the EWMA baseline itself.
    fallback: "none" (default) - any model or input error is raised;
              "ewma" - if the neural model cannot be loaded or the window fails validation, return the EWMA forecast
                       instead (method="ewma_fallback", with the reason in warnings). EWMA itself still requires
                       12 valid, ordered raw dwell hours; otherwise the original error is raised.
    """
    if fallback not in ("none", "ewma"):
        raise ValueError("fallback must be 'none' or 'ewma'")
    if model == "ewma":
        return _ewma_result(window, port_id, "ewma_baseline", [])
    try:
        return load_model(model, registry_path, dataset_dir).predict(window, port_id)
    except (ModelLoadError, InputValidationError) as err:
        if fallback != "ewma":
            raise
        try:
            return _ewma_result(window, port_id, "ewma_fallback", [f"neural model {model!r} not used: {err}"])
        except InputValidationError as fb_err:
            raise InputValidationError(f"{err}; EWMA fallback also unavailable: {fb_err}") from err
