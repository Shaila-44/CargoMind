"""Reference forecasts that any DL model must beat. They see exactly what a model sees: the 12 input
hours of the window (no history before the window, nothing after hour t).

* persistence:   y_hat(t+1) = dwell(t)
* trailing mean: y_hat(t+1) = mean(dwell(t-11 .. t))
* EWMA:          exponentially weighted mean over the window (initialised at dwell(t-11));
                 smoothing factor chosen on TRAINING windows only.
All in hours; errors on windows with observed targets only.
"""
import numpy as np
import pandas as pd

from . import config


def ewma_window(dw, alpha):
    s = dw[:, 0].copy()
    for j in range(1, dw.shape[1]):
        s = alpha * dw[:, j] + (1 - alpha) * s
    return s


def predictions(dwell_hours, alpha):
    return {"persistence": dwell_hours[:, -1], "trailing_mean_12h": dwell_hours.mean(1),
            f"ewma_alpha_{alpha}": ewma_window(dwell_hours, alpha)}


def choose_alpha(dwell_train, y_train):
    scores = {a: float(np.sqrt(((ewma_window(dwell_train, a) - y_train) ** 2).mean())) for a in config.EWMA_ALPHAS}
    return min(scores, key=scores.get), scores


def evaluate(meta, preds):
    rows = []
    for name, p in preds.items():
        err = p - meta["target_hours"].values
        for split in ("train", "val", "test"):
            m = (meta["split"] == split).values
            for port in ["ALL"] + sorted(meta["port_id"].unique()):
                mm = m if port == "ALL" else m & (meta["port_id"] == port).values
                e = err[mm]
                rows.append({"baseline": name, "split": split, "port_id": port, "n": int(mm.sum()),
                             "mae_h": float(np.abs(e).mean()), "rmse_h": float(np.sqrt((e ** 2).mean())),
                             "bias_h": float(e.mean())})
    return pd.DataFrame(rows)
