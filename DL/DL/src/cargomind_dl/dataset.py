"""Loader for the saved DL dataset (dl/dataset/), shared by later LSTM / GRU / 1D-CNN experiments."""
import json

import numpy as np

from . import config

DWELL_RECON_CHANNEL = "dwell_reconstructed"


def load_split(split, mask_channels=("dwell",)):
    """Arrays for one split. X gets one extra 0/1 channel per name in `mask_channels`
    (1 = that feature's value at that hour has no raw observation). Pass () for the 7 features only."""
    d = dict(np.load(config.DATASET / f"{split}.npz"))
    names = list(config.PROVENANCE_COLUMNS)
    extra = [(d["provenance"][..., names.index(f)] == config.STATUS_CODES["reconstructed"]).astype(np.float32)[..., None]
             for f in mask_channels]
    d["X"] = np.concatenate([d["X"], *extra], axis=-1) if extra else d["X"]
    d["channels"] = config.FEATURES + [f"{f}_reconstructed" for f in mask_channels]
    return d


def load_scalers():
    return json.loads((config.DATASET / "scalers.json").read_text())


def to_hours(y_scaled, port_index, scalers=None):
    """Invert the per-port dwell scaling (for evaluation in hours)."""
    sc = (scalers or load_scalers())["dwell"]["per_port"]
    ports = json.loads((config.DATASET / "feature_schema.json").read_text())["arrays"]["port_index"]["ports"]
    mu = np.array([sc[p]["mean"] for p in ports])[port_index]
    sd = np.array([sc[p]["std"] for p in ports])[port_index]
    return np.asarray(y_scaled) * sd + mu
