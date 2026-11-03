"""Save / load trained forecasters. Depends only on torch and dl.models - not on the training code -
so a later inference module can load a model on its own."""
# =============================================================================
# WHAT IS A CHECKPOINT & WHY DO WE NEED INTEGRITY CHECKS?
# =============================================================================
# Training a deep learning model takes significant time, computational power,
# and electricity. Once a model is trained, you don't want to re-train it every
# time you need a prediction!
#
# A "Checkpoint" is a complete snapshot of the trained neural network saved to a file:
#   - Model architecture settings (number of layers, neurons, channels).
#   - Trainable parameters (millions of learned connection weights).
#   - Exact list of input feature names in the exact required order.
#
# WHY DO WE RECORD SHA-256 CHECKSUMS OF DATASET ARTIFACTS?
# Imagine you teach a student to calculate temperatures in Fahrenheit.
# Later in production, someone replaces the thermometer with one that outputs Celsius!
# The student would produce completely wrong answers without realizing why.
#
# To prevent this silent failure:
# Whenever a model is saved, we take cryptographic fingerprints (SHA-256 hashes)
# of the data scalers (`scalers.json`) and schema (`feature_schema.json`).
# When loading the model, it verifies that those files haven't been modified since
# the model was trained. If they changed, it refuses to run, protecting your system.
# =============================================================================

from __future__ import annotations

# --- Standard library imports ---
import hashlib  # Used to compute SHA-256 cryptographic fingerprints of files
import json     # For saving configuration files in readable JSON format
from pathlib import Path  # Object-oriented path handling across Mac/Linux/Windows

# --- PyTorch imports ---
import torch

# --- Internal model definitions ---
from .recurrent import ModelConfig, RecurrentForecaster
from .gnn import GCNLSTMForecaster, GNNConfig
from .transformer import TransformerConfig, TransformerForecaster

# Registry of supported deep learning model families
# Maps model type string -> (Configuration Class, Neural Network Class)
MODEL_TYPES = {
    "recurrent": (ModelConfig, RecurrentForecaster),
    "transformer": (TransformerConfig, TransformerForecaster),
    "gnn": (GNNConfig, GCNLSTMForecaster),
}


def model_type_of(model) -> str:
    """Identify which model family an instantiated model belongs to."""
    for name, (_, cls) in MODEL_TYPES.items():
        if isinstance(model, cls):
            return name
    raise TypeError(f"unsupported model class {type(model).__name__}")


from cargomind_dl.config import DATASET as DATASET_DIR  # noqa: E402 (Points to DL/Dataset/processed)


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 checksum (digital fingerprint) of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_reference() -> dict:
    """Create a tamper-evident record of the exact preprocessing files used during training.

    Records the file path and SHA-256 hash for:
      - feature_schema.json (defines input feature channels)
      - scalers.json (mean and standard deviation scaling factors per port)
      - split_metadata.json (chronological cutoff dates)
    """
    out = {}
    for name in ("feature_schema.json", "scalers.json", "split_metadata.json"):
        p = DATASET_DIR / name
        out[name] = {"path": f"DL/Dataset/processed/{name}", "sha256": file_sha256(p)}
    return out


def save_checkpoint(
    path: Path,
    model: RecurrentForecaster | TransformerForecaster | GCNLSTMForecaster,
    channels: list[str],
    extra: dict | None = None,
) -> None:
    """Save a complete model bundle to disk.

    Saves:
      1. Model type ('recurrent', 'transformer', or 'gnn')
      2. Model configuration dictionary (hyperparameters)
      3. State dict (learned weights and biases)
      4. Channel names in exact column order
      5. Target definition and invertibility instructions
      6. Dataset reference hashes for integrity verification
      7. Any optional metadata (epochs, training loss, validation RMSE)
    """
    payload = {
        "model_type": model_type_of(model),
        "model_config": model.cfg.to_dict(),
        "state_dict": model.state_dict(),
        "input_channels": list(channels),
        "target": "next-hour container_dwell_hours, per-port z-score; invert with cargomind_dl.dataset.to_hours",
        "dataset_reference": dataset_reference(),
        "extra": extra or {},
    }
    # Ensure directory exists before saving
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Save using PyTorch binary format
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    map_location: str = "cpu",
) -> tuple[RecurrentForecaster | TransformerForecaster | GCNLSTMForecaster, dict]:
    """Rebuild and load a trained model ready for evaluation or inference.

    Steps performed:
      1. Load checkpoint file from disk.
      2. Verify which model family it belongs to.
      3. Instantiate the model class with its original configuration.
      4. Restore learned weights and put the model into .eval() mode.
      5. Verify that preprocessing files match their original SHA-256 checksums!
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)

    # Look up the configuration and model classes (defaults to 'recurrent' for legacy checkpoints)
    cfg_cls, model_cls = MODEL_TYPES[payload.get("model_type", "recurrent")]

    # Reconstruct model and load weights
    model = model_cls(cfg_cls(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()  # Put into inference evaluation mode (disables dropout)

    # Verify cryptographic integrity of preprocessing artifacts
    for name, ref in payload["dataset_reference"].items():
        current = file_sha256(DATASET_DIR / name)
        if current != ref["sha256"]:
            raise RuntimeError(
                f"Data integrity error: {name} was modified since this model was trained "
                f"({ref['sha256'][:12]} -> {current[:12]}). Predictions would be invalid!"
            )

    return model, payload


def write_config_json(path: Path, payload_like: dict) -> None:
    """Helper function to dump configuration dictionaries to human-readable JSON files."""
    Path(path).write_text(json.dumps(payload_like, indent=2, default=str))
