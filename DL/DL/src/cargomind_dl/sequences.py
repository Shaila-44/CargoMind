"""Chronological split, training-only scalers and 12-hour sliding windows.

Split: by the global index of the TARGET hour (same cut-offs for every port).
Window (port p, hour t): inputs = hours t-11 .. t, target = dwell at hour t+1 of the same port.
A window is kept only if
  1. all 13 hours exist and are consecutive for that port (guaranteed by the panel checks),
  2. every input hour lies in the same split as the target hour (embargo after each boundary),
  3. the target value was observed or unit-converted in raw - reconstructed targets are excluded
     from training loss and from evaluation (they are not observations).
Reconstructed INPUT values are kept as published in Dataset/Processed (no re-imputation, so no new
values are invented and no future values are used) and are identified by the provenance masks.
"""
import numpy as np
import pandas as pd

from . import config

SPLITS = ("train", "val", "test")


def split_bounds(n_hours):
    a = int(round(n_hours * config.TRAIN_FRACTION))
    b = int(round(n_hours * (config.TRAIN_FRACTION + config.VAL_FRACTION)))
    return {"train": (0, a), "val": (a, b), "test": (b, n_hours)}  # [start, end) in hour_index


def split_of(hour_index, bounds):
    for s in SPLITS:
        lo, hi = bounds[s]
        if lo <= hour_index < hi:
            return s
    raise ValueError(hour_index)


# ------------------------------------------------------------------------------------------
# Scalers (fitted on training-period hours only; reconstructed cells excluded from the fit)
# ------------------------------------------------------------------------------------------
SCALING = {
    "dwell": "per-port z-score",
    "utilization": "fixed: value / 100 (bounded 0-1, no fitting)",
    "labour_strike": "none (0/1)",
    "wind_gust": "global z-score",
    "disruption_active": "none (0/1)",
    "disruption_hours_elapsed": "log1p, then global z-score",
    "vessels_at_anchor": "log1p, then per-port z-score",
}


def _fit_rows(panel, bounds, feat):
    train = panel["hour_index"] < bounds["train"][1]
    if f"prov_{feat}" in panel:
        train &= panel[f"prov_{feat}"] != config.STATUS_CODES["reconstructed"]
    return panel[train]


def fit_scalers(panel, bounds):
    sc = {}
    for feat in ("dwell", "vessels_at_anchor"):
        rows = _fit_rows(panel, bounds, feat)
        v = rows[feat] if feat == "dwell" else np.log1p(rows[feat])
        stats = v.groupby(rows["port_id"]).agg(["mean", "std", "size"])
        sc[feat] = {"method": SCALING[feat], "per_port": {p: {"mean": float(r["mean"]), "std": float(r["std"]), "n_fit": int(r["size"])}
                                                          for p, r in stats.iterrows()}}
    for feat in ("wind_gust", "disruption_hours_elapsed"):
        rows = _fit_rows(panel, bounds, feat)
        v = rows[feat] if feat == "wind_gust" else np.log1p(rows[feat])
        sc[feat] = {"method": SCALING[feat], "mean": float(v.mean()), "std": float(v.std()), "n_fit": int(len(v))}
    for feat in ("utilization", "labour_strike", "disruption_active"):
        sc[feat] = {"method": SCALING[feat]}
    sc["_fit_period"] = {"hour_index": list(bounds["train"]), "note": "training-period hours only; reconstructed cells excluded"}
    return sc


def transform(panel, sc):
    out = pd.DataFrame(index=panel.index)
    port = panel["port_id"]
    for feat in ("dwell", "vessels_at_anchor"):
        mu = port.map({p: s["mean"] for p, s in sc[feat]["per_port"].items()})
        sd = port.map({p: s["std"] for p, s in sc[feat]["per_port"].items()})
        v = panel[feat] if feat == "dwell" else np.log1p(panel[feat])
        out[feat] = (v - mu) / sd
    out["wind_gust"] = (panel["wind_gust"] - sc["wind_gust"]["mean"]) / sc["wind_gust"]["std"]
    s = sc["disruption_hours_elapsed"]
    out["disruption_hours_elapsed"] = (np.log1p(panel["disruption_hours_elapsed"]) - s["mean"]) / s["std"]
    out["utilization"] = panel["utilization"] / 100.0
    out["labour_strike"] = panel["labour_strike"].astype(float)
    out["disruption_active"] = panel["disruption_active"].astype(float)
    assert out.notna().all().all(), "scaled features must be finite"
    return out[config.FEATURES]


def inverse_dwell(z, ports, sc):
    s = sc["dwell"]["per_port"]
    return np.array([zz * s[p]["std"] + s[p]["mean"] for zz, p in zip(z, ports)])


# ------------------------------------------------------------------------------------------
# Windows
# ------------------------------------------------------------------------------------------
def build_windows(panel, scaled, bounds):
    W = config.WINDOW
    prov_cols = [f"prov_{f}" for f in config.PROVENANCE_COLUMNS]
    flag_cols = ["disruption_end_reconstructed", "wind_gust_clipped"]
    ports = sorted(panel["port_id"].unique())
    rows, X, P, F, y_z = [], [], [], [], []
    Xall = scaled.to_numpy(np.float32)
    Pall = panel[prov_cols].to_numpy(np.int8)
    Fall = panel[flag_cols].to_numpy(np.int8)
    for p in ports:
        idx = np.flatnonzero(panel["port_id"].values == p)  # sorted by hour, contiguous grid
        hi = panel["hour_index"].values[idx]
        for k in range(W - 1, len(idx) - config.HORIZON):
            inp = idx[k - W + 1:k + 1]
            tgt = idx[k + config.HORIZON]
            assert (np.diff(hi[k - W + 1:k + 2]) == 1).all()  # 13 consecutive hours
            t_split = split_of(hi[k + config.HORIZON], bounds)
            first_split = split_of(hi[k - W + 1], bounds)
            tgt_status = panel["prov_dwell"].values[tgt]
            rows.append({
                "port_id": p, "port_index": ports.index(p),
                "t_hour_index": int(hi[k]), "t_ts": panel["ts_hour"].values[idx[k]],
                "target_hour_index": int(hi[k + 1]), "target_ts": panel["ts_hour"].values[tgt],
                "split": t_split, "crosses_split_boundary": first_split != t_split,
                "target_status": {v: k_ for k_, v in config.STATUS_CODES.items()}[int(tgt_status)],
                "target_hours": float(panel["dwell"].values[tgt]),
                "last_input_dwell_hours": float(panel["dwell"].values[idx[k]]),
                "n_reconstructed_input_cells": int((Pall[inp] == config.STATUS_CODES["reconstructed"]).sum()),
                "n_reconstructed_dwell_inputs": int((Pall[inp, 0] == config.STATUS_CODES["reconstructed"]).sum()),
                "any_disruption_end_reconstructed": bool(Fall[inp, 0].any()),
            })
            X.append(Xall[inp])
            P.append(Pall[inp])
            F.append(Fall[inp])
            y_z.append(Xall[tgt, 0])
    meta = pd.DataFrame(rows)
    meta["target_observed"] = meta["target_status"] != "reconstructed"
    meta["keep"] = ~meta["crosses_split_boundary"] & meta["target_observed"]
    meta["exclusion_reason"] = np.where(meta["crosses_split_boundary"], "inputs cross a split boundary (embargo)",
                                        np.where(~meta["target_observed"], "target reconstructed (not an observation)", ""))
    return meta, np.stack(X), np.stack(P), np.stack(F), np.array(y_z, np.float32)
