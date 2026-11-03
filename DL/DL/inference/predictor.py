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
# =============================================================================
# THE DEEP LEARNING INFERENCE & PRODUCTION PREDICTION ENGINE
# =============================================================================
# "Inference" means putting a trained deep learning model to work on live, incoming data!
#
# This file is the primary gateway for predicting next-hour container dwell times:
#   1. You give it a 12-hour history of port telemetry (past 12 hours of hourly dwell,
#      ship queues, weather, and congestion).
#   2. It validates the data to make sure no hours are missing or corrupt.
#   3. It scales the numbers using the exact statistics learned during training.
#   4. It feeds the data through our champion neural network (the Transformer).
#   5. It converts the model's abstract mathematical score back into REAL HOURS (e.g. "28.5 hours").
#   6. AUTOMATIC SAFETY NET (FALLBACK):
#      In real-world logistics, software must never crash! If the primary Transformer
#      model file is missing or corrupted, the system automatically falls back to a
#      reliable LSTM model or moving-average baseline, warning the user without failing.
# =============================================================================

from __future__ import annotations

# --- Standard library imports ---
import json  # For reading feature schema and scaler JSON files
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone  # For stamping predictions with exact UTC timestamps
from pathlib import Path

# --- Third-party library imports ---
import numpy as np
import pandas as pd
import torch

# --- Internal package imports ---
from cargomind_dl import config as data_config
from cargomind_dl import dataset as dl_dataset
from cargomind_dl.sequences import transform

from dl.models import load_checkpoint

from . import registry
from .fallback import EWMA_ALPHA, ewma_forecast
from .validation import LIMITS, WINDOW, InputValidationError, validate_window

# Name of the column tracking whether dwell was reconstructed during cleaning
RECONSTRUCTION_CHANNEL = "dwell_reconstructed"

# If a port's current dwell is more than 4 standard deviations from its historical mean,
# we log a warning because neural networks are less confident on extreme unprecedented outliers.
TYPICAL_RANGE_Z = 4.0

# Version contract for the structured prediction output
CONTRACT_VERSION = "cargomind-dl-inference/1.0"
FORECAST_HORIZON_HOURS = 1
FALLBACK_MODES = ("auto", "none", "lstm", "ewma")


class ModelLoadError(RuntimeError):
    """Raised when a requested model checkpoint or its required preprocessing files cannot be loaded."""


class PredictionError(RuntimeError):
    """Raised when a loaded model produces an impossible or invalid result (like negative or infinite hours)."""


@dataclass
class DwellPrediction:
    """The official structured result contract returned for every dwell forecast.

    Contains:
      - predicted_dwell_hours: The forecasted container dwell time in real hours.
      - method: How the number was generated ('neural', 'neural_fallback', or 'ewma_baseline').
      - model_name: Which specific architecture produced the forecast (e.g. 'transformer_t_w').
      - model_version: File path and cryptographic hash of the model checkpoint used.
      - port_id: The port code (e.g. 'SGSIN' for Singapore).
      - window_start_utc / window_end_utc: The 12-hour input window timestamps.
      - target_hour_utc: The exact hour being predicted (t + 1).
      - warnings: Any operational warnings (e.g. extreme values or fallback triggers).
      - fallback_used: True if the primary model failed and a backup model stepped in.
    """
    predicted_dwell_hours: float
    method: str                 # "neural" | "neural_fallback" | "ewma_fallback" | "ewma_baseline"
    model_name: str             # Model that produced the number
    model_version: str          # Checkpoint path + short SHA-256 hash
    port_id: str
    window_end_utc: str | None  # Last input hour (t)
    target_hour_utc: str | None  # The forecast hour (t + 1)
    warnings: list[str] = field(default_factory=list)
    note: str = "point forecast only; no calibrated uncertainty is provided"
    model_role: str = "primary"            # "primary" | "fallback" | "research" | "baseline"
    requested_model: str | None = None     # The model originally requested by the caller
    fallback_used: bool = False
    fallback_reason: str | None = None     # Explanation if fallback was triggered
    forecast_horizon_hours: int = FORECAST_HORIZON_HOURS
    window_start_utc: str | None = None    # First input hour (t - 11)
    window_hours: int = WINDOW
    generated_at_utc: str | None = None    # Timestamp when forecast was computed
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict:
        """Convert the prediction object to a clean Python dictionary."""
        return asdict(self)


class DwellForecaster:
    """Wrapper that manages a loaded neural model and its matching preprocessing scalers."""

    def __init__(
        self,
        checkpoint: Path,
        name: str,
        expected_sha256: str | None = None,
        dataset_dir: Path | None = None,
        role: str = "research",
    ):
        checkpoint, dataset_dir = Path(checkpoint), Path(dataset_dir or data_config.DATASET)

        # Step 1: Ensure checkpoint file exists
        if not checkpoint.exists():
            raise ModelLoadError(f"checkpoint for model {name!r} not found: {checkpoint}")

        # Step 2: Verify checksum against the model registry
        sha = registry.sha256(checkpoint)
        if expected_sha256 and sha != expected_sha256:
            raise ModelLoadError(f"checkpoint {checkpoint} does not match the registry (sha256 {sha[:12]} != {expected_sha256[:12]})")

        # Step 3: Verify required scaler and schema files exist
        files = {}
        for fname in ("feature_schema.json", "scalers.json"):
            p = dataset_dir / fname
            if not p.exists():
                raise ModelLoadError(f"required preprocessing artifact missing: {p}")
            files[fname] = p

        # Step 4: Load model weights and verify dataset binding
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
        except ValueError:
            shown = checkpoint.name
        self.version = f"{shown}@{sha[:12]}"

    def preprocess(self, window: pd.DataFrame, port_id: str) -> tuple[np.ndarray, list[str], pd.DataFrame]:
        """Convert a raw 12-hour DataFrame into scaled neural network input [1, 12, Channels]."""
        # 1. Validate that the input window has 12 consecutive hourly rows without gaps
        w, warnings = validate_window(window, port_id, self.ports)

        # 2. Apply port-specific z-score normalization (subtract mean, divide by standard deviation)
        panel = w.assign(port_id=port_id)
        scaled = transform(panel, self.scalers)

        # 3. Stack columns into an array in the exact order expected by the neural network
        cols = {c: scaled[c].to_numpy(np.float32) for c in data_config.FEATURES}
        cols[RECONSTRUCTION_CHANNEL] = w[RECONSTRUCTION_CHANNEL].to_numpy(np.float32)
        x = np.stack([cols[c] for c in self.channels], axis=-1)[None]

        # 4. Check for statistical anomalies in the input
        z_last = float(scaled["dwell"].iloc[-1])
        if abs(z_last) > TYPICAL_RANGE_Z:
            warnings.append(
                f"latest dwell is {z_last:+.1f} training SDs from this port's training mean; "
                "neural forecasts are least reliable outside the training range"
            )
        if w[RECONSTRUCTION_CHANNEL].any():
            warnings.append("input contains hours flagged as reconstructed dwell")

        return x, warnings, w

    def predict(self, window: pd.DataFrame, port_id: str) -> DwellPrediction:
        """Run the neural network forward pass and convert output back into real hours."""
        # Preprocess input window
        x, warnings, w = self.preprocess(window, port_id)
        port_index = np.array([self.ports.index(port_id)])

        # Run inference in eval mode without calculating gradients (saves memory & fast)
        self.model.eval()
        with torch.no_grad():
            z = self.model(torch.from_numpy(x), torch.from_numpy(port_index)).numpy()

        # Invert scaling: Convert the model's z-score output back into real hours!
        hours = float(dl_dataset.to_hours(z.astype(np.float64), port_index, self.scalers)[0])

        # Sanity check: Ensure output is finite and physically plausible
        if not np.isfinite(hours):
            raise PredictionError(f"model {self.name} produced a non-finite prediction")
        lo, hi = LIMITS["dwell"]
        if not lo <= hours <= hi:
            raise PredictionError(f"model {self.name} produced an implausible dwell of {hours:.1f} h (outside [{lo}, {hi}])")

        start, end = w["ts_hour"].iloc[0], w["ts_hour"].iloc[-1]
        return DwellPrediction(
            predicted_dwell_hours=hours,
            method="neural",
            model_name=self.name,
            model_version=self.version,
            port_id=port_id,
            window_end_utc=end.isoformat(),
            target_hour_utc=(end + pd.Timedelta(hours=1)).isoformat(),
            warnings=warnings,
            model_role=self.role,
            requested_model=self.name,
            window_start_utc=start.isoformat(),
            generated_at_utc=_now(),
        )


# Global in-memory cache so models aren't reloaded from disk on every single prediction call
_CACHE: dict = {}


def _now() -> str:
    """Return current UTC time formatted as an ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_model(
    name: str = registry.DEFAULT_MODEL,
    registry_path: Path | None = None,
    dataset_dir: Path | None = None,
) -> DwellForecaster:
    """Load a registered neural forecaster by name (e.g. 'transformer_t_w' or 'c_w_lstm')."""
    try:
        reg = registry.load_registry(registry_path or registry.REGISTRY_PATH)
    except FileNotFoundError as e:
        raise ModelLoadError(str(e)) from e

    if name not in reg["models"]:
        raise ModelLoadError(f"unknown model {name!r}; registered neural models: {sorted(reg['models'])} (EWMA: model='ewma')")

    entry = reg["models"][name]
    key = (name, str(registry_path), str(dataset_dir))
    if key not in _CACHE:
        _CACHE[key] = DwellForecaster(
            registry.PROJECT_ROOT / entry["checkpoint"],
            name,
            entry["checkpoint_sha256"],
            dataset_dir,
            role=entry.get("role", "research"),
        )
    return _CACHE[key]


def _ewma_result(window, port_id, method: str, warnings: list[str]) -> DwellPrediction:
    """Calculate the simple Exponentially Weighted Moving Average baseline forecast."""
    if not isinstance(window, pd.DataFrame) or "dwell" not in window.columns or "ts_hour" not in window.columns:
        raise InputValidationError("EWMA needs a DataFrame with 'ts_hour' and raw 'dwell' columns (12 hours, oldest first)")

    ts = pd.to_datetime(window["ts_hour"], utc=True, errors="coerce")
    if len(ts) != 12 or ts.isna().any() or not (ts.diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise InputValidationError("EWMA needs 12 consecutive hourly rows, oldest first")

    # Compute moving average
    hours = ewma_forecast(window["dwell"].to_numpy(dtype=float, na_value=np.nan))
    start, end = ts.iloc[0], ts.iloc[-1]

    return DwellPrediction(
        predicted_dwell_hours=hours,
        method=method,
        model_name="ewma_alpha_0.2",
        model_version=f"EWMA over the 12 input hours, alpha {EWMA_ALPHA}, initialised at the first hour (baseline, not a neural model)",
        port_id=port_id,
        window_end_utc=end.isoformat(),
        target_hour_utc=(end + pd.Timedelta(hours=1)).isoformat(),
        warnings=warnings,
        model_role="baseline",
        window_start_utc=start.isoformat(),
        generated_at_utc=_now(),
    )


def forecast(
    window: pd.DataFrame,
    port_id: str,
    model: str = registry.DEFAULT_MODEL,
    fallback: str = "auto",
    registry_path: Path | None = None,
    dataset_dir: Path | None = None,
) -> DwellPrediction:
    """Predict next-hour container dwell time for a port with automatic fallback protection.

    Parameters:
      window: A DataFrame of exactly 12 hourly rows for this port.
      port_id: Port code (e.g. 'SGSIN', 'NLRTM', 'USLAX').
      model: The requested model name (defaults to 'transformer_t_w').
      fallback: Fallback strategy ('auto', 'lstm', 'ewma', or 'none').

    How Fallback Works:
      If 'auto' is selected and the primary Transformer fails for any reason (corrupt checkpoint,
      unsupported architecture, non-finite output), the engine automatically falls back to the
      LSTM model ('c_w_lstm') and documents the incident in the output warnings.
    """
    if fallback not in FALLBACK_MODES:
        raise ValueError(f"fallback must be one of {FALLBACK_MODES}")

    # If caller specifically asked for the simple EWMA baseline
    if model == "ewma":
        return _ewma_result(window, port_id, "ewma_baseline", [])

    # Try running the primary requested neural model
    try:
        result = load_model(model, registry_path, dataset_dir).predict(window, port_id)
        result.requested_model = model
        return result
    except InputValidationError as err:
        # Invalid input data: Fallback models would fail too, so raise immediately unless legacy ewma
        if fallback != "ewma":
            raise
        primary_error: Exception = err
    except (ModelLoadError, PredictionError, RuntimeError) as err:
        primary_error = err
        if fallback == "none" or (fallback == "auto" and model != registry.DEFAULT_MODEL):
            raise

    # Fallback Option 1: Legacy EWMA baseline
    if fallback == "ewma":
        try:
            r = _ewma_result(window, port_id, "ewma_fallback", [f"neural model {model!r} not used: {primary_error}"])
        except InputValidationError as fb_err:
            raise InputValidationError(f"{primary_error}; EWMA fallback also unavailable: {fb_err}") from primary_error
        r.requested_model, r.fallback_used, r.fallback_reason = model, True, f"{type(primary_error).__name__}: {primary_error}"
        return r

    # Fallback Option 2: The official registered backup neural model (LSTM)
    fb_name = registry.FALLBACK_MODEL
    if fb_name == model:
        raise primary_error

    try:
        r = load_model(fb_name, registry_path, dataset_dir).predict(window, port_id)
    except (ModelLoadError, PredictionError, RuntimeError) as fb_err:
        raise type(fb_err)(
            f"primary model {model!r} failed ({primary_error}); fallback model {fb_name!r} also failed ({fb_err})"
        ) from primary_error

    # Successfully fell back to backup model: Record full transparency in the result
    reason = f"{type(primary_error).__name__}: {primary_error}"
    r.method, r.model_role, r.requested_model = "neural_fallback", "fallback", model
    r.fallback_used, r.fallback_reason = True, reason
    r.warnings.insert(
        0,
        f"fallback model {fb_name!r} used because {model!r} could not produce a valid prediction ({reason})"
    )
    return r
