"""Statistical uncertainty with cluster (block) bootstraps.

Shipments that leave the same port on the same day share congestion, weather and disruption
episodes, so they are not independent. The ML stage resampled whole departure days; that scheme is
kept as the primary one. The test period has only 11 departure days, so day-level intervals are
coarse; an origin-port x day scheme (more, smaller clusters - assumes ports are independent within
a day) is reported as a second view. Model comparisons are *paired*: both models are scored on the
same resample, so the interval is for the difference itself.
"""
import numpy as np
import pandas as pd

from . import metrics as M
from .common import ANO_LABEL, OUT, SUP_LABEL, block_resamples, mlc



def _f1(y, p):
    cm = M.confusion(y, p.argmax(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.nan_to_num(np.diag(cm) / cm.sum(0))
        rec = np.nan_to_num(np.diag(cm) / cm.sum(1))
        f1 = np.nan_to_num(2 * prec * rec / (prec + rec))
    return f1, cm.sum(1) / cm.sum()


SUP_METRICS = {
    "macro_f1": lambda y, p: _f1(y, p)[0].mean(),
    "weighted_f1": lambda y, p: float((_f1(y, p)[0] * _f1(y, p)[1]).sum()),
    "accuracy": lambda y, p: (p.argmax(1) == y).mean(),
    "critical_recall": lambda y, p: ((p.argmax(1) == 2) & (y == 2)).sum() / max(1, (y == 2).sum()),
    "critical_precision": lambda y, p: ((p.argmax(1) == 2) & (y == 2)).sum() / max(1, (p.argmax(1) == 2).sum()),
    "critical_f1": lambda y, p: _f1(y, p)[0][2],
    "at_risk_recall": lambda y, p: ((p.argmax(1) == 1) & (y == 1)).sum() / max(1, (y == 1).sum()),
    "critical_roc_auc": lambda y, p: M.auc_rank(y == 2, p[:, 2]),
    "roc_auc_ovr_macro": lambda y, p: np.mean([M.auc_rank(y == k, p[:, k]) for k in range(3)]),
    "log_loss": lambda y, p: M.log_loss(y, p),
    "brier_multiclass": lambda y, p: M.brier_multiclass(y, p),
}
PAIRS = [("logistic_regression", "random_forest"), ("logistic_regression", "xgboost"),
         ("logistic_regression", "lightgbm"), ("xgboost", "lightgbm")]


def blocks_for(df, scheme):
    day = df["departure_ts"].dt.floor("D").astype(str).values
    if scheme == "day":
        return day
    return np.char.add(np.char.add(day.astype(str), "|"), df["origin_port_id"].astype(str).values)


def single_model_ci(df, probas, scheme="day"):
    y = df["y"].values
    blocks = blocks_for(df, scheme)
    draws = {n: {k: [] for k in SUP_METRICS} for n in probas}
    for idx in block_resamples(blocks):
        for n, p in probas.items():
            for k, f in SUP_METRICS.items():
                draws[n][k].append(f(y[idx], p[idx]))
    rows = []
    for n, d in draws.items():
        for k, v in d.items():
            rows.append({"model": SUP_LABEL[n], "metric": k, "scheme": scheme, "point": SUP_METRICS[k](y, probas[n]),
                         "ci_low": np.percentile(v, 2.5), "ci_high": np.percentile(v, 97.5), "boot_sd": np.std(v)})
    return pd.DataFrame(rows)


def paired_supervised(df, probas, scheme):
    y = df["y"].values
    blocks = blocks_for(df, scheme)
    diffs = {(a, b, k): [] for a, b in PAIRS for k in SUP_METRICS}
    for idx in block_resamples(blocks):
        vals = {n: {k: f(y[idx], probas[n][idx]) for k, f in SUP_METRICS.items()} for n in probas}
        for a, b in PAIRS:
            for k in SUP_METRICS:
                diffs[(a, b, k)].append(vals[a][k] - vals[b][k])
    rows = []
    for (a, b, k), v in diffs.items():
        v = np.asarray(v)
        point = SUP_METRICS[k](y, probas[a]) - SUP_METRICS[k](y, probas[b])
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append({"model_a": SUP_LABEL[a], "model_b": SUP_LABEL[b], "metric": k, "scheme": scheme,
                     "difference_a_minus_b": point, "ci_low": lo, "ci_high": hi, "share_of_resamples_a_better":
                     float(((v < 0) if k in ("log_loss", "brier_multiclass") else (v > 0)).mean()),
                     "interval_excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def paired_anomaly(df, scores, thresholds, scheme):
    y = df["y"].values
    blocks = blocks_for(df, scheme)
    a, b = "isolation_forest", "one_class_svm"

    def stats(idx, name):
        s, yy = scores[name][idx], y[idx]
        flag = s > thresholds[name]
        pos_d, pos_c = yy > 0, yy == 2
        return {"delayed_pr_auc": M.average_precision(pos_d, s), "delayed_roc_auc": M.auc_rank(pos_d, s),
                "critical_pr_auc": M.average_precision(pos_c, s), "critical_roc_auc": M.auc_rank(pos_c, s),
                "delayed_precision": (flag & pos_d).sum() / max(1, flag.sum()), "delayed_recall": (flag & pos_d).sum() / max(1, pos_d.sum()),
                "critical_recall": (flag & pos_c).sum() / max(1, pos_c.sum()), "on_time_flag_rate": flag[yy == 0].mean()}

    full = np.arange(len(y))
    base = {n: stats(full, n) for n in (a, b)}
    draws = {k: [] for k in base[a]}
    single = {n: {k: [] for k in base[a]} for n in (a, b)}
    for idx in block_resamples(blocks):
        sa, sb = stats(idx, a), stats(idx, b)
        for k in draws:
            draws[k].append(sa[k] - sb[k])
            single[a][k].append(sa[k])
            single[b][k].append(sb[k])
    rows = []
    for k, v in draws.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append({"model_a": ANO_LABEL[a], "model_b": ANO_LABEL[b], "metric": k, "scheme": scheme,
                     "difference_a_minus_b": base[a][k] - base[b][k], "ci_low": lo, "ci_high": hi,
                     "share_of_resamples_a_better": float(((np.asarray(v) < 0) if k == "on_time_flag_rate" else (np.asarray(v) > 0)).mean()),
                     "interval_excludes_zero": bool(lo > 0 or hi < 0)})
    ci = [{"model": ANO_LABEL[n], "metric": k, "scheme": scheme, "point": base[n][k],
           "ci_low": np.percentile(single[n][k], 2.5), "ci_high": np.percentile(single[n][k], 97.5)} for n in (a, b) for k in base[n]]
    return pd.DataFrame(rows), pd.DataFrame(ci)


def run(test, test_proba, ano_scores, thresholds):
    ci = pd.concat([single_model_ci(test, test_proba, s) for s in ("day", "port_day")])
    ci.to_csv(OUT["metrics"] / "supervised_metric_ci.csv", index=False)
    pairs = pd.concat([paired_supervised(test, test_proba, s) for s in ("day", "port_day")])
    ano = [paired_anomaly(test, ano_scores, thresholds, s) for s in ("day", "port_day")]
    pairs = pd.concat([pairs, *[a[0] for a in ano]])
    pairs.to_csv(OUT["metrics"] / "pairwise_comparisons.csv", index=False)
    ano_ci = pd.concat([a[1] for a in ano])
    ano_ci.to_csv(OUT["metrics"] / "anomaly_metric_ci.csv", index=False)
    n_blocks = {s: len(np.unique(blocks_for(test, s))) for s in ("day", "port_day")}
    return ci, pairs, ano_ci, n_blocks


__all__ = ["run", "mlc"]
