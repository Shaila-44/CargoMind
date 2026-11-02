"""Post-hoc probability calibration for the 3-class classifiers (one-vs-rest, then renormalised).

Same scheme as sklearn's CalibratedClassifierCV for multiclass problems: one calibrator per class, fitted on that
class's predicted probability against the one-vs-rest outcome, then the three calibrated values are normalised to sum
to 1 (uniform if all three are 0).

* "sigmoid"  - Platt scaling: logistic regression on the logit of the class probability;
* "isotonic" - isotonic regression on the class probability (monotone, non-parametric).

A calibrator only re-maps probabilities. It is fitted on held-out (validation) predictions, never on the rows the
model was trained on, and it does not change the saved model.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6
METHODS = ("sigmoid", "isotonic")


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class OvRCalibrator:
    def __init__(self, method: str):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "OvRCalibrator":
        proba, y = np.asarray(proba, float), np.asarray(y)
        if proba.ndim != 2 or proba.shape[1] != 3 or len(proba) != len(y):
            raise ValueError("proba must be [n, 3] and aligned with y")
        self.calibrators_ = []
        for k in range(3):
            target = (y == k).astype(int)
            if self.method == "sigmoid":
                c = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(proba[:, k])[:, None], target)
            else:
                c = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(proba[:, k], target)
            self.calibrators_.append(c)
        self.n_fit_ = int(len(y))
        self.class_counts_ = [int((y == k).sum()) for k in range(3)]
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        proba = np.asarray(proba, float)
        cols = []
        for k, c in enumerate(self.calibrators_):
            if self.method == "sigmoid":
                cols.append(c.predict_proba(_logit(proba[:, k])[:, None])[:, 1])
            else:
                cols.append(c.predict(proba[:, k]))
        q = np.column_stack(cols)
        s = q.sum(1, keepdims=True)
        return np.where(s > 0, q / np.where(s > 0, s, 1), 1.0 / 3)
