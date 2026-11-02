"""Inference interface for CargoMind next-hour dwell forecasting (no training code is imported).

    from dl.inference import forecast
    result = forecast(window_df, port_id="SGSIN")                     # production: transformer_t_w, automatic c_w_lstm fallback
    result = forecast(window_df, port_id="SGSIN", fallback="none")    # transformer_t_w only; any failure raises
    result = forecast(window_df, port_id="SGSIN", model="gru")        # a research model by name (no automatic fallback)
    result = forecast(window_df, port_id="SGSIN", fallback="ewma")    # legacy: EWMA if the neural path fails
    result = forecast(window_df, port_id="SGSIN", model="ewma")       # the EWMA baseline itself (benchmark/reference)

Pipeline: validate raw window -> apply the dataset's train-fitted transforms (cargomind_dl.sequences.transform,
saved scalers.json) -> select the checkpoint's channels by name, in the checkpoint's order -> model in eval mode
under torch.no_grad() -> invert with cargomind_dl.dataset.to_hours -> output check -> structured result.

Automatic fallback (fallback="auto", the default): when the default model (transformer_t_w) cannot be loaded (missing,
altered or incompatible checkpoint or preprocessing artifacts) or returns an invalid result (non-finite, or outside the
plausible dwell range), the registered fallback model (c_w_lstm) is used and the result says so (fallback_used,
fallback_reason, warnings). Invalid *input* never triggers the fallback: both models use the same validated window, so
the input error is raised instead.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
from .validation import LIMITS, WINDOW, InputValidationError, validate_window  # noqa: E402

RECONSTRUCTION_CHANNEL = "dwell_reconstructed"
TYPICAL_RANGE_Z = 4.0  # |z| of the latest dwell beyond this -> warning (outside the port's training-period spread)
CONTRACT_VERSION = "cargomind-dl-inference/1.0"
FORECAST_HORIZON_HOURS = 1
FALLBACK_MODES = ("auto", "none", "lstm", "ewma")


class ModelLoadError(RuntimeError):
    """A requested model or its preprocessing artifacts are missing, altered or incompatible."""


class PredictionError(RuntimeError):
    """A loaded model produced an invalid result (non-finite or implausible dwell)."""


@dataclass
class DwellPrediction:
    """Output contract (CONTRACT_VERSION). Optional fields are None when they do not apply."""
    predicted_dwell_hours: float
    method: str                 # "neural" | "neural_fallback" | "ewma_fallback" | "ewma_baseline"
    model_name: str             # model that produced the number: registry name, or "ewma_alpha_0.2"
    model_version: str          # checkpoint path + short SHA-256, or the EWMA definition
    port_id: str
    window_end_utc: str | None  # last input hour (t)
    target_hour_utc: str | None  # the forecast hour (t + 1)
    warnings: list[str] = field(default_factory=list)
    note: str = "point forecast only; no calibrated uncertainty is provided"
    model_role: str = "primary"            # "primary" | "fallback" | "research" | "baseline" (registry role)
    requested_model: str | None = None     # the model the caller asked for (default: transformer_t_w)
    fallback_used: bool = False
    fallback_reason: str | None = None     # optional: why the requested model was not used
    forecast_horizon_hours: int = FORECAST_HORIZON_HOURS
    window_start_utc: str | None = None    # first input hour (t - 11)
    window_hours: int = WINDOW
    generated_at_utc: str | None = None    # when the forecast was produced
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


class DwellForecaster:
    """One loaded neural model plus the exact preprocessing artifacts its checkpoint was trained with."""

    def __init__(self, checkpoint: Path, name: str, expected_sha256: str | None = None, dataset_dir: Path | None = None,
                 role: str = "research"):
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
        self.role = role
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
            raise PredictionError(f"model {self.name} produced a non-finite prediction")
        lo, hi = LIMITS["dwell"]
        if not lo <= hours <= hi:
            raise PredictionError(f"model {self.name} produced an implausible dwell of {hours:.1f} h (outside [{lo}, {hi}])")
        start, end = w["ts_hour"].iloc[0], w["ts_hour"].iloc[-1]
        return DwellPrediction(predicted_dwell_hours=hours, method="neural", model_name=self.name, model_version=self.version,
                               port_id=port_id, window_end_utc=end.isoformat(), target_hour_utc=(end + pd.Timedelta(hours=1)).isoformat(),
                               warnings=warnings, model_role=self.role, requested_model=self.name, window_start_utc=start.isoformat(),
                               generated_at_utc=_now())


_CACHE: dict = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        _CACHE[key] = DwellForecaster(registry.PROJECT_ROOT / entry["checkpoint"], name, entry["checkpoint_sha256"], dataset_dir,
                                      role=entry.get("role", "research"))
    return _CACHE[key]


def _ewma_result(window, port_id, method: str, warnings: list[str]) -> DwellPrediction:
    if not isinstance(window, pd.DataFrame) or "dwell" not in window.columns or "ts_hour" not in window.columns:
        raise InputValidationError("EWMA needs a DataFrame with 'ts_hour' and raw 'dwell' columns (12 hours, oldest first)")
    ts = pd.to_datetime(window["ts_hour"], utc=True, errors="coerce")
    if len(ts) != 12 or ts.isna().any() or not (ts.diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise InputValidationError("EWMA needs 12 consecutive hourly rows, oldest first")
    hours = ewma_forecast(window["dwell"].to_numpy(dtype=float, na_value=np.nan))
    start, end = ts.iloc[0], ts.iloc[-1]
    return DwellPrediction(predicted_dwell_hours=hours, method=method, model_name="ewma_alpha_0.2",
                           model_version=f"EWMA over the 12 input hours, alpha {EWMA_ALPHA}, initialised at the first hour (baseline, not a neural model)",
                           port_id=port_id, window_end_utc=end.isoformat(), target_hour_utc=(end + pd.Timedelta(hours=1)).isoformat(),
                           warnings=warnings, model_role="baseline", window_start_utc=start.isoformat(), generated_at_utc=_now())


def forecast(window: pd.DataFrame, port_id: str, model: str = registry.DEFAULT_MODEL, fallback: str = "auto",
             registry_path: Path | None = None, dataset_dir: Path | None = None) -> DwellPrediction:
    """Next-hour dwell forecast in hours.

    model:    a registered neural model name (default: the primary, transformer_t_w), or "ewma" for the EWMA baseline.
    fallback: "auto" (default) - if model is the registry default and it cannot be loaded or returns an invalid result,
                       the registry fallback model (c_w_lstm) is used. Other models get no automatic fallback.
              "lstm" - the registry fallback model is used whenever `model` fails to load or predict.
              "none" - any model or input error is raised.
              "ewma" - legacy: if the neural model cannot be loaded or the window fails validation, return the EWMA
                       baseline (method="ewma_fallback"). EWMA itself still requires 12 valid, ordered raw dwell hours.
    Input validation errors are raised under "auto", "lstm" and "none": the fallback model needs the same valid window.
    If the fallback model also fails, ModelLoadError / PredictionError is raised naming both failures.
    """
    if fallback not in FALLBACK_MODES:
        raise ValueError(f"fallback must be one of {FALLBACK_MODES}")
    if model == "ewma":
        return _ewma_result(window, port_id, "ewma_baseline", [])
    try:
        result = load_model(model, registry_path, dataset_dir).predict(window, port_id)
        result.requested_model = model
        return result
    except InputValidationError as err:
        if fallback != "ewma":
            raise
        primary_error: Exception = err
    except (ModelLoadError, PredictionError, RuntimeError) as err:
        primary_error = err
        if fallback == "none" or (fallback == "auto" and model != registry.DEFAULT_MODEL):
            raise
    if fallback == "ewma":
        try:
            r = _ewma_result(window, port_id, "ewma_fallback", [f"neural model {model!r} not used: {primary_error}"])
        except InputValidationError as fb_err:
            raise InputValidationError(f"{primary_error}; EWMA fallback also unavailable: {fb_err}") from primary_error
        r.requested_model, r.fallback_used, r.fallback_reason = model, True, f"{type(primary_error).__name__}: {primary_error}"
        return r
    fb_name = registry.FALLBACK_MODEL
    if fb_name == model:
        raise primary_error
    try:
        r = load_model(fb_name, registry_path, dataset_dir).predict(window, port_id)
    except (ModelLoadError, PredictionError, RuntimeError) as fb_err:
        raise type(fb_err)(f"primary model {model!r} failed ({primary_error}); fallback model {fb_name!r} also failed ({fb_err})") from primary_error
    reason = f"{type(primary_error).__name__}: {primary_error}"
    r.method, r.model_role, r.requested_model = "neural_fallback", "fallback", model
    r.fallback_used, r.fallback_reason = True, reason
    r.warnings.insert(0, f"fallback model {fb_name!r} used because {model!r} could not produce a valid prediction ({reason})")
    return r
