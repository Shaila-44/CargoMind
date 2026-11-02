"""Metrics in original hours, subgroup breakdowns and paired block-bootstrap comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

HIGH_SIGNAL = ("NLRTM", "USLAX", "CNSHA", "INNSA", "SGSIN")  # from the DL EDA (signal share 83-99 %)
NOISE_DOMINATED = ("MYPKG", "DEHAM", "AEJEA", "INMAA")  # signal share 11-36 %; LKCMB (58 %) sits between
MIN_N = 30  # subgroups smaller than this are reported but flagged as too small to interpret


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(p - y)))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((p - y) ** 2) / ss) if ss > 0 else float("nan")


def port_balanced_rmse(y: np.ndarray, p: np.ndarray, port: np.ndarray) -> float:
    """Mean of per-port RMSEs (each port weighted equally; NLRTM otherwise dominates pooled error)."""
    return float(np.mean([rmse(y[port == q], p[port == q]) for q in np.unique(port)]))


def subgroups(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    g = {"all": np.ones(len(frame), bool)}
    g["disruption active at t"] = frame["disruption_active_t"].values == 1
    g["no disruption at t (normal)"] = frame["disruption_active_t"].values == 0
    for lo, hi, lab in [(1, 6, "disruption 1–6 h old"), (7, 24, "disruption 7–24 h old"), (25, 999, "disruption > 24 h old")]:
        g[lab] = (frame["disruption_hours_elapsed_t"].values >= lo) & (frame["disruption_hours_elapsed_t"].values <= hi)
    g["labour strike at t"] = frame["labour_strike_t"].values == 1
    g["no strike at t"] = frame["labour_strike_t"].values == 0
    g["≥ 1 reconstructed dwell input"] = frame["n_reconstructed_dwell_inputs"].values > 0
    g["all dwell inputs observed"] = frame["n_reconstructed_dwell_inputs"].values == 0
    g["high-signal ports"] = frame["port_id"].isin(HIGH_SIGNAL).values
    g["noise-dominated ports"] = frame["port_id"].isin(NOISE_DOMINATED).values
    for p in sorted(frame["port_id"].unique()):
        g[f"port {p}"] = frame["port_id"].values == p
    return g


def breakdown(frame: pd.DataFrame, preds: dict[str, np.ndarray], reference: str = "EWMA α=0.2") -> pd.DataFrame:
    y = frame["y_hours"].values
    rows = []
    for name, m in subgroups(frame).items():
        n = int(m.sum())
        if n == 0:
            continue
        for model, p in preds.items():
            rows.append({"subgroup": name, "n": n, "small_sample": n < MIN_N, "model": model, "mae_h": mae(y[m], p[m]),
                         "rmse_h": rmse(y[m], p[m]), "r2": r2(y[m], p[m]) if n > 2 else np.nan,
                         "rmse_minus_reference_h": rmse(y[m], p[m]) - rmse(y[m], preds[reference][m])})
    return pd.DataFrame(rows)


def paired_bootstrap(frame: pd.DataFrame, p_model: np.ndarray, p_ref: np.ndarray, blocks: str, reps: int = 2000, seed: int = 42) -> dict:
    """95 % interval for RMSE(model) − RMSE(reference), resampling whole blocks (days or port-days)."""
    y = frame["y_hours"].values
    day = pd.to_datetime(frame["target_ts"], utc=True).dt.floor("D").astype(str).values
    key = day if blocks == "day" else np.char.add(np.char.add(day.astype(str), "|"), frame["port_id"].values.astype(str))
    uniq = np.unique(key)
    idx_by = [np.flatnonzero(key == k) for k in uniq]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(reps):
        idx = np.concatenate([idx_by[i] for i in rng.integers(0, len(uniq), len(uniq))])
        d.append(rmse(y[idx], p_model[idx]) - rmse(y[idx], p_ref[idx]))
    d = np.array(d)
    return {"blocks": blocks, "n_blocks": int(len(uniq)), "point": rmse(y, p_model) - rmse(y, p_ref),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "share_model_better": float((d < 0).mean())}
