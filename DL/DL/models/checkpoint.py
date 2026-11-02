"""Save / load trained forecasters. Depends only on torch and dl.models - not on the training code -
so a later inference module can load a model on its own."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .recurrent import ModelConfig, RecurrentForecaster
from .gnn import GCNLSTMForecaster, GNNConfig
from .transformer import TransformerConfig, TransformerForecaster

# checkpoints written before the Transformer existed have no "model_type" and are recurrent
MODEL_TYPES = {"recurrent": (ModelConfig, RecurrentForecaster), "transformer": (TransformerConfig, TransformerForecaster),
               "gnn": (GNNConfig, GCNLSTMForecaster)}


def model_type_of(model) -> str:
    for name, (_, cls) in MODEL_TYPES.items():
        if isinstance(model, cls):
            return name
    raise TypeError(f"unsupported model class {type(model).__name__}")

from cargomind_dl.config import DATASET as DATASET_DIR  # noqa: E402  (DL/Dataset/processed)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_reference() -> dict:
    """Pointers to the files a user needs to build inputs and invert outputs (no copies, no refits)."""
    out = {}
    for name in ("feature_schema.json", "scalers.json", "split_metadata.json"):
        p = DATASET_DIR / name
        out[name] = {"path": f"DL/Dataset/processed/{name}", "sha256": file_sha256(p)}
    return out


def save_checkpoint(path: Path, model: RecurrentForecaster | TransformerForecaster | GCNLSTMForecaster, channels: list[str],
                    extra: dict | None = None) -> None:
    """Weights + everything needed to rebuild and use the model."""
    payload = {
        "model_type": model_type_of(model),
        "model_config": model.cfg.to_dict(),
        "state_dict": model.state_dict(),
        "input_channels": list(channels),
        "target": "next-hour container_dwell_hours, per-port z-score; invert with cargomind_dl.dataset.to_hours",
        "dataset_reference": dataset_reference(),
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> tuple[RecurrentForecaster | TransformerForecaster | GCNLSTMForecaster, dict]:
    """Rebuild the model in eval mode. Raises if the saved dataset files have changed since training."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg_cls, model_cls = MODEL_TYPES[payload.get("model_type", "recurrent")]
    model = model_cls(cfg_cls(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for name, ref in payload["dataset_reference"].items():
        current = file_sha256(DATASET_DIR / name)
        if current != ref["sha256"]:
            raise RuntimeError(f"{name} changed since this model was trained ({ref['sha256'][:12]} -> {current[:12]})")
    return model, payload


def write_config_json(path: Path, payload_like: dict) -> None:
    Path(path).write_text(json.dumps(payload_like, indent=2, default=str))
