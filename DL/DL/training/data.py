"""Thin adapter over the existing loader (DL/DL/src/cargomind_dl/dataset.py). No scaling, no imputation:
arrays are used exactly as saved; features can only be selected by name."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from cargomind_dl import dataset as dl_dataset


@dataclass
class SplitData:
    split: str
    X: np.ndarray            # [N, 12, C] float32, scaled
    y: np.ndarray            # [N] float32, per-port z-score
    y_hours: np.ndarray      # [N] float32
    port_index: np.ndarray   # [N] int64
    window_id: np.ndarray    # [N]
    channels: list[str]
    provenance: np.ndarray   # [N, 12, 5] int8, kept for subgroup evaluation

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X), torch.from_numpy(self.y), torch.from_numpy(self.port_index)


def load(split: str, mask_channels: tuple[str, ...] = ("dwell",), drop_features: tuple[str, ...] = ()) -> SplitData:
    d = dl_dataset.load_split(split, mask_channels=mask_channels)
    channels = list(d["channels"])
    unknown = [f for f in drop_features if f not in channels]
    if unknown:
        raise ValueError(f"cannot drop unknown channels {unknown}; available {channels}")
    keep = [i for i, c in enumerate(channels) if c not in drop_features]
    return SplitData(split=split, X=np.ascontiguousarray(d["X"][..., keep], dtype=np.float32), y=d["y"].astype(np.float32),
                     y_hours=d["y_hours"].astype(np.float32), port_index=d["port_index"].astype(np.int64),
                     window_id=d["window_id"], channels=[channels[i] for i in keep], provenance=d["provenance"])


def to_hours(y_scaled: np.ndarray, port_index: np.ndarray) -> np.ndarray:
    """The dataset's own inverse transform (never re-implemented here)."""
    return dl_dataset.to_hours(np.asarray(y_scaled, dtype=np.float64), np.asarray(port_index))


def dwell_std_by_port_index() -> np.ndarray:
    """Per-port dwell SD (hours) from the saved scaler, ordered by port_index. Read, never refitted."""
    sc = dl_dataset.load_scalers()["dwell"]["per_port"]
    ports = json.loads((dl_dataset.config.DATASET / "feature_schema.json").read_text())["arrays"]["port_index"]["ports"]
    return np.array([sc[p]["std"] for p in ports], dtype=np.float64)
