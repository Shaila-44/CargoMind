"""Anomaly detector evaluation (Isolation Forest, One-Class SVM).

The detectors score how unusual the origin port's operating state is; they are not delay-class
predictors. Labels are used only to measure how the scores relate to realised delay. Two positive
mappings: "delayed" (At-Risk or Critical) - the ML stage's primary mapping - and "Critical only".
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import metrics as M
from .common import ANO_LABEL, ANOMALY, ANOMALY_INPUT_COLUMNS, CLASS_COLORS, CLASSES, FIG, MODEL_COLORS, OUT, SEED, mlc, plt, savefig


def detector_metrics(y, s, thr):
    y = np.asarray(y)
    flag = s > thr
    out = {"n": int(len(y)), "threshold": thr, "flag_rate": flag.mean()}
    for pos_name, pos in [("delayed", y > 0), ("critical", y == 2)]:
        tp, fp = (flag & pos).sum(), (flag & ~pos).sum()
        fn, tn = (~flag & pos).sum(), (~flag & ~pos).sum()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        out.update({f"{pos_name}_prevalence": pos.mean(), f"{pos_name}_precision": prec, f"{pos_name}_recall": rec,
                    f"{pos_name}_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                    f"{pos_name}_fpr": fp / max(1, fp + tn), f"{pos_name}_lift": prec / pos.mean(),
                    f"{pos_name}_roc_auc": M.auc_rank(pos, s), f"{pos_name}_pr_auc": M.average_precision(pos, s)})
    for k, c in enumerate(CLASSES):
        out[f"flag_rate_{c}"] = flag[y == k].mean()
        out[f"share_of_flags_{c}"] = (flag & (y == k)).sum() / max(1, flag.sum())
    return out


def verify_scores(test, scores):
    stored = pd.read_csv(mlc.RESULT_DIRS["anomaly"] / "test_anomaly_scores.csv")
    rows = []
    for name, s in scores.items():
        st = stored.set_index("shipment_id").loc[test["shipment_id"], f"{name}_score"].values
        rows.append({"model": ANO_LABEL[name], "max_abs_score_diff": float(np.abs(st - s).max()),
                     "flags_identical": bool((stored.set_index("shipment_id").loc[test["shipment_id"], f"{name}_flag"].values ==
                                              (s > joblib.load(mlc.MODELS / f"{name}.joblib")["threshold"])).all())})
    return pd.DataFrame(rows)


def cost_and_stability(ctx, test):
    """Evaluation-only re-fits: wall-clock cost and sensitivity to the random seed / subsample."""
    train = ctx["train"]
    normal = train[train["y"] == 0][ANOMALY_INPUT_COLUMNS]
    y = test["y"].values
    mods = {m.NAME: m for m in ANOMALY}
    rows, stab = [], []
    for name, m in mods.items():
        saved = joblib.load(mlc.MODELS / f"{name}.joblib")
        params = saved["pipeline"].named_steps["model"].get_params()
        base = m.anomaly_score(saved["pipeline"], test[ANOMALY_INPUT_COLUMNS])
        t0 = time.perf_counter()
        for _ in range(5):
            m.anomaly_score(saved["pipeline"], test[ANOMALY_INPUT_COLUMNS])
        score_ms_per_1000 = (time.perf_counter() - t0) / 5 / len(test) * 1000 * 1000
        for seed in [SEED, 1, 2, 3, 4, 5]:
            if name == "isolation_forest":
                fit_X = normal
                pipe = m.build_pipeline(**{k: params[k] for k in ("n_estimators", "max_samples", "max_features")})
                pipe.set_params(model__random_state=seed)
            else:
                fit_X = normal.sample(min(len(normal), mlc.OCSVM_MAX_TRAIN), random_state=seed)
                pipe = m.build_pipeline(**{k: params[k] for k in ("kernel", "nu", "gamma")})
            t0 = time.perf_counter()
            pipe.fit(fit_X)
            fit_s = time.perf_counter() - t0
            thr = float(np.quantile(m.anomaly_score(pipe, normal), mlc.ANOMALY_FLAG_QUANTILE))
            s = m.anomaly_score(pipe, test[ANOMALY_INPUT_COLUMNS])
            flag, base_flag = s > thr, base > saved["threshold"]
            stab.append({"model": ANO_LABEL[name], "seed": seed, "fit_seconds": fit_s,
                         "spearman_vs_saved": spearmanr(s, base)[0],
                         "flag_jaccard_vs_saved": (flag & base_flag).sum() / max(1, (flag | base_flag).sum()),
                         **{k: v for k, v in detector_metrics(y, s, thr).items()
                            if k in ("flag_rate", "delayed_precision", "delayed_recall", "delayed_roc_auc", "delayed_pr_auc",
                                     "critical_recall", "critical_pr_auc")}})
        rows.append({"model": ANO_LABEL[name], "fit_rows": len(normal) if name == "isolation_forest" else mlc.OCSVM_MAX_TRAIN,
                     "fit_seconds_median": float(np.median([r["fit_seconds"] for r in stab if r["model"] == ANO_LABEL[name]])),
                     "score_ms_per_1000_rows": score_ms_per_1000,
                     "model_file_mb": os.path.getsize(mlc.MODELS / f"{name}.joblib") / 1e6,
                     "tuning_grid_seconds_stored": __import__("json").loads((mlc.MODELS / f"{name}_config.json").read_text())["seconds"]})
    stab = pd.DataFrame(stab)
    stab.to_csv(OUT["validation"] / "anomaly_seed_stability.csv", index=False)
    return pd.DataFrame(rows), stab


def incremental_value(test, scores, lr_proba):
    """Does the anomaly flag carry information about Critical outcomes beyond Logistic Regression?"""
    y = test["y"].values
    pred = lr_proba.argmax(1)
    rows = []
    for name, s in scores.items():
        thr = joblib.load(mlc.MODELS / f"{name}.joblib")["threshold"]
        flag = s > thr
        for k, c in enumerate(CLASSES):
            for f in (True, False):
                m = (pred == k) & (flag == f)
                rows.append({"model": ANO_LABEL[name], "analysis": "Critical rate by LR prediction x flag", "lr_predicted": c,
                             "flagged": f, "n": int(m.sum()), "critical_rate": (y[m] == 2).mean() if m.any() else np.nan,
                             "delayed_rate": (y[m] > 0).mean() if m.any() else np.nan})
        missed = (y == 2) & (pred != 2)
        rows.append({"model": ANO_LABEL[name], "analysis": "Critical shipments missed by LR that the detector flags",
                     "n": int(missed.sum()), "share": flag[missed].mean()})
        rows.append({"model": ANO_LABEL[name], "analysis": "Critical shipments caught by LR that the detector flags",
                     "n": int(((y == 2) & (pred == 2)).sum()), "share": flag[(y == 2) & (pred == 2)].mean()})
        # AUC of the anomaly score for Critical inside deciles of LR's P(Critical): information not already in LR
        dec = pd.qcut(lr_proba[:, 2], 10, labels=False, duplicates="drop")
        aucs, ws = [], []
        for d in np.unique(dec):
            m = dec == d
            if 0 < (y[m] == 2).sum() < m.sum():
                aucs.append(M.auc_rank(y[m] == 2, s[m]))
                ws.append((y[m] == 2).sum() * (y[m] != 2).sum())
        rows.append({"model": ANO_LABEL[name], "analysis": "Critical ROC-AUC of anomaly score within LR P(Critical) deciles (pair-weighted)",
                     "n": len(aucs), "share": float(np.average(aucs, weights=ws))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT["metrics"] / "anomaly_incremental_value.csv", index=False)
    return out


def plot_distributions(y, scores, thresholds, tag):
    fig, axes = plt.subplots(2, len(scores), figsize=(6.2 * len(scores), 7.2))
    for j, (name, s) in enumerate(scores.items()):
        thr = thresholds[name]
        ax = axes[0, j]
        lo, hi = np.percentile(s, [0.5, 99.5])
        bins = np.linspace(lo, hi, 45)
        for k, c in enumerate(CLASSES):
            ax.hist(s[y == k], bins=bins, density=True, histtype="step", lw=1.6, color=CLASS_COLORS[c], label=f"{c} (n={(y == k).sum()})")
        ax.axvline(thr, color="#52514e", ls="--", lw=1)
        ax.set_title(f"{ANO_LABEL[name]}: score density by true class ({tag})")
        ax.set_xlabel("anomaly score (higher = more unusual)")
        ax.legend(fontsize=8)
        ax = axes[1, j]
        for k, c in enumerate(CLASSES):
            v = np.sort(s[y == k])
            ax.plot(v, 1 - np.arange(1, len(v) + 1) / len(v), color=CLASS_COLORS[c], lw=1.6, label=c)
        ax.axvline(thr, color="#52514e", ls="--", lw=1)
        ax.text(thr, 0.95, " flag threshold", fontsize=8, color="#52514e")
        ax.set_ylabel("share of class with score above x")
        ax.set_xlabel("anomaly score")
        ax.set_title("Survival curves: share of each class above a score")
    savefig(fig, FIG["anomaly_distributions"] / f"anomaly_scores_{tag}.png")


def threshold_sweep(y, scores, tag):
    rows = []
    for name, s in scores.items():
        for q in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]:
            thr = float(np.quantile(s, q))
            r = detector_metrics(y, s, thr)
            rows.append({"model": ANO_LABEL[name], "set": tag, "flag_quantile_of_set": q,
                         **{k: r[k] for k in ("flag_rate", "delayed_precision", "delayed_recall", "critical_precision",
                                              "critical_recall", "delayed_fpr")}})
    return pd.DataFrame(rows)


def run(ctx, test, scores_test, scores_val, lr_test_proba):
    val = ctx["train"].iloc[ctx["folds"][ctx["val_fold"]][1]]
    thresholds = {n: joblib.load(mlc.MODELS / f"{n}.joblib")["threshold"] for n in scores_test}
    rows = []
    for tag, df, sc in [("validation", val, scores_val), ("test", test, scores_test)]:
        for name, s in sc.items():
            rows.append({"model": ANO_LABEL[name], "set": tag, **detector_metrics(df["y"].values, s, thresholds[name])})
        plot_distributions(df["y"].values, sc, thresholds, tag)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT["metrics"] / "anomaly_metrics.csv", index=False)
    sweep = pd.concat([threshold_sweep(val["y"].values, scores_val, "validation"),
                       threshold_sweep(test["y"].values, scores_test, "test")])
    sweep.to_csv(OUT["metrics"] / "anomaly_threshold_sweep.csv", index=False)
    cost, stab = cost_and_stability(ctx, test)
    cost.to_csv(OUT["metrics"] / "anomaly_cost.csv", index=False)
    inc = incremental_value(test, scores_test, lr_test_proba)
    return metrics, cost, stab, inc, sweep
