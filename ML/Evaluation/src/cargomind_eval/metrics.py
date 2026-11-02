"""Metric definitions, implemented from first principles (confusion matrix, ranks) so that the
numbers stored by the ML stage can be checked against an independent computation.
"""
import numpy as np
from scipy.stats import rankdata

from .common import CLASSES, LABELS

EPS = 1e-15


def confusion(y, pred):
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (np.asarray(y), np.asarray(pred)), 1)
    return cm


def auc_rank(pos, score):
    """ROC-AUC via the Mann-Whitney statistic (ties get mid-ranks)."""
    pos = np.asarray(pos, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def average_precision(pos, score):
    """Step-wise area under the precision-recall curve (same definition as sklearn's AP)."""
    pos = np.asarray(pos, bool)
    if pos.sum() == 0:
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    s, p = score[order], pos[order]
    # group tied scores: evaluate precision/recall only at the last index of each distinct score
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(p)[last]
    fp = (last + 1) - tp
    prec, rec = tp / (tp + fp), tp / pos.sum()
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def brier_multiclass(y, proba):
    """Mean over rows of sum_k (p_k - 1[y=k])^2; range 0 (perfect) to 2."""
    onehot = np.eye(3)[np.asarray(y)]
    return float(((proba - onehot) ** 2).sum(1).mean())


def log_loss(y, proba):
    p = np.clip(proba[np.arange(len(y)), np.asarray(y)], EPS, 1)
    return float(-np.log(p).mean())


def ece_top_label(y, proba, bins=10):
    """Expected calibration error of the top-label confidence, equal-width bins."""
    conf, pred = proba.max(1), proba.argmax(1)
    correct = pred == np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(correct[b == i].mean() - conf[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any()))


def ece_classwise(y, proba, bins=10):
    """Mean over classes of the one-vs-rest ECE with equal-frequency bins."""
    y = np.asarray(y)
    out = []
    for k in LABELS:
        p, o = proba[:, k], (y == k).astype(float)
        q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
        b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
        out.append(sum(abs(o[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in np.unique(b)))
    return float(np.mean(out)), [float(v) for v in out]


def classification(y, proba, pred=None):
    """All supervised metrics for one model on one set. `pred` defaults to the argmax decision."""
    y = np.asarray(y)
    pred = proba.argmax(1) if pred is None else np.asarray(pred)
    cm = confusion(y, pred)
    n = cm.sum()
    support = cm.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.nan_to_num(np.diag(cm) / cm.sum(0))
        rec = np.nan_to_num(np.diag(cm) / support)
        f1 = np.nan_to_num(2 * prec * rec / (prec + rec))
    w = support / n
    out = {"n": int(n), "accuracy": np.trace(cm) / n,
           "macro_precision": prec.mean(), "macro_recall": rec.mean(), "macro_f1": f1.mean(),
           "weighted_precision": (w * prec).sum(), "weighted_recall": (w * rec).sum(), "weighted_f1": (w * f1).sum(),
           "log_loss": log_loss(y, proba), "brier_multiclass": brier_multiclass(y, proba),
           "ece_top_label": ece_top_label(y, proba), "ece_classwise": ece_classwise(y, proba)[0]}
    aucs = [auc_rank(y == k, proba[:, k]) for k in LABELS]
    out["roc_auc_ovr_macro"] = float(np.nanmean(aucs))
    out["roc_auc_ovr_weighted"] = float(np.nansum(np.array(aucs) * w))
    for k, c in enumerate(CLASSES):
        key = c.lower().replace("-", "_")
        out.update({f"{key}_precision": prec[k], f"{key}_recall": rec[k], f"{key}_f1": f1[k], f"{key}_support": int(support[k]),
                    f"{key}_roc_auc_ovr": aucs[k], f"{key}_pr_auc_ovr": average_precision(y == k, proba[:, k]),
                    f"{key}_prevalence": support[k] / n, f"{key}_predicted_share": (pred == k).mean()})
    out.update({
        "critical_false_negatives": int(support[2] - cm[2, 2]),
        "critical_pred_on_time": int(cm[2, 0]), "critical_pred_at_risk": int(cm[2, 1]),
        "critical_pred_on_time_rate": cm[2, 0] / max(1, support[2]),
        "on_time_pred_critical": int(cm[0, 2]), "at_risk_pred_critical": int(cm[1, 2]),
        "critical_alerts": int(cm[:, 2].sum()), "false_critical_alerts": int(cm[0, 2] + cm[1, 2]),
        "at_risk_pred_on_time": int(cm[1, 0]), "on_time_pred_at_risk": int(cm[0, 1]),
        "ordinal_mae": float(np.abs(pred - y).mean()),
        "two_class_errors": int(cm[2, 0] + cm[0, 2]),  # On-Time <-> Critical: skips the middle class
    })
    return out
