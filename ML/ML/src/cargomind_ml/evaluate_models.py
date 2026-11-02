"""Evaluation of the saved models.

Supervised: accuracy, per-class / macro / weighted precision-recall-F1, one-vs-rest ROC-AUC,
log loss, ordinal error, Critical->On-Time miss rate, confusion matrices and classification
reports, on the validation block and on the test period (all shipments and sea only). Test-period
uncertainty is estimated with a bootstrap that resamples whole departure days, because shipments
leaving the same port on the same day share conditions and are not independent.

Anomaly: scores are compared with the known classes; they are never turned into class predictions.
Positive mapping: anomaly-positive = delayed (At-Risk or Critical); Critical-only is reported too.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as ss  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config  # noqa: E402
from .features import FEATURES  # noqa: E402
from .preprocessing import ANOMALY_INPUT_COLUMNS  # noqa: E402

LABELS = [0, 1, 2]
CLASS_COLORS = {"On-Time": "#2a78d6", "At-Risk": "#eda100", "Critical": "#e34948"}  # same as EDA config


def setup_plots():
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.titlesize": 11, "axes.titleweight": "semibold",
                         "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e6e5e1", "legend.frameon": False})


# ---------------------------------------------------------------------------------------------
# Supervised metrics
# ---------------------------------------------------------------------------------------------
def classification_metrics(y, proba):
    y = np.asarray(y)
    pred = proba.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    out = {
        "n": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y, pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y, pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "log_loss": log_loss(y, proba, labels=LABELS),
        # mean |predicted class - true class| on the ordinal scale 0/1/2
        "ordinal_mae": float(np.abs(pred - y).mean()),
        # costliest error: a Critical shipment predicted On-Time
        "critical_predicted_on_time_rate": float(((pred == 0) & (y == 2)).sum() / max(1, (y == 2).sum())),
    }
    for i, c in enumerate(config.CLASSES):
        key = c.lower().replace("-", "_")
        out.update({f"{key}_precision": p[i], f"{key}_recall": r[i], f"{key}_f1": f[i], f"{key}_support": int(s[i])})
    if len(np.unique(y)) == 3:
        out["roc_auc_ovr_macro"] = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=LABELS)
        out["roc_auc_ovr_weighted"] = roc_auc_score(y, proba, multi_class="ovr", average="weighted", labels=LABELS)
        for i, c in enumerate(config.CLASSES):
            out[f"{c.lower().replace('-', '_')}_roc_auc_ovr"] = roc_auc_score(y == i, proba[:, i])
    return out


def day_block_bootstrap(y, proba, days, reps=config.BOOTSTRAP_REPS, seed=config.SEED):
    """95 % percentile intervals, resampling departure days with replacement."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), proba.argmax(1)
    uniq = np.unique(days)
    idx_by_day = {d: np.flatnonzero(days == d) for d in uniq}
    vals = {"macro_f1": [], "critical_recall": [], "accuracy": [], "weighted_f1": []}
    for _ in range(reps):
        idx = np.concatenate([idx_by_day[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        yy, pp = y[idx], pred[idx]
        vals["macro_f1"].append(f1_score(yy, pp, average="macro", labels=LABELS, zero_division=0))
        vals["weighted_f1"].append(f1_score(yy, pp, average="weighted", labels=LABELS, zero_division=0))
        vals["accuracy"].append((yy == pp).mean())
        vals["critical_recall"].append(recall_score(yy, pp, labels=[2], average="macro", zero_division=0))
    return {f"{k}_ci95": (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}


def save_classification_outputs(name, label, y, proba, tag):
    pred = proba.argmax(1)
    rep = classification_report(y, pred, labels=LABELS, target_names=config.CLASSES, digits=3, zero_division=0, output_dict=True)
    pd.DataFrame(rep).T.to_csv(config.RESULT_DIRS["classification_reports"] / f"{name}_{tag}.csv")
    (config.RESULT_DIRS["classification_reports"] / f"{name}_{tag}.txt").write_text(
        f"{label} - {tag}\n\n" + classification_report(y, pred, labels=LABELS, target_names=config.CLASSES, digits=3, zero_division=0))
    cm = confusion_matrix(y, pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"true {c}" for c in config.CLASSES], columns=[f"pred {c}" for c in config.CLASSES])
    cm_df.to_csv(config.RESULT_DIRS["confusion_matrices"] / f"{name}_{tag}.csv")
    return cm


def plot_confusion_grid(cms, tag, title):
    """One panel per model: cell = share of the true class (row-normalised recall), with counts."""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6))
    axes = np.atleast_1d(axes)
    for ax, (label, cm) in zip(axes, cms.items()):
        share = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{share[i, j]:.0%}\n({cm[i, j]:,})", ha="center", va="center", fontsize=8.5,
                        color="white" if share[i, j] > 0.55 else "#0b0b0b")
        ax.set_xticks(range(3), config.CLASSES, fontsize=8)
        ax.set_yticks(range(3), config.CLASSES, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true" if ax is axes[0] else "")
        ax.set_title(label)
        ax.grid(False)
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(config.RESULT_DIRS["figures"] / f"confusion_matrices_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def evaluate_supervised(ctx, modules):
    train, test, folds, k = ctx["train"], ctx["test"], ctx["folds"], ctx["val_fold"]
    tr_idx, val_idx = folds[k]
    val = train.iloc[val_idx]
    test_sea = test[test["shipping_mode"] == "sea"]
    days = test["departure_ts"].dt.floor("D").values
    rows, probas = [], {}
    cms = {"validation": {}, "test": {}, "test_sea": {}}
    entries = [(m.NAME, m.LABEL) for m in modules] + [("dummy_prior", "Majority-class reference")]
    for name, label in entries:
        model = joblib.load(config.MODELS / f"{name}.joblib")
        # validation: same hyperparameters, refit on the fold's training rows only
        fold_model = clone(model).fit(train.iloc[tr_idx][FEATURES], train.iloc[tr_idx]["y"])
        p_val = fold_model.predict_proba(val[FEATURES])
        p_test = model.predict_proba(test[FEATURES])
        p_sea = model.predict_proba(test_sea[FEATURES])
        probas[name] = p_test
        for tag, df, p in [("validation", val, p_val), ("test", test, p_test), ("test_sea", test_sea, p_sea)]:
            m = classification_metrics(df["y"], p)
            if tag == "test":
                m.update(day_block_bootstrap(df["y"].values, p, days))
            rows.append({"model": label, "key": name, "set": tag, **m})
            cm = save_classification_outputs(name, label, df["y"].values, p, tag)
            if name != "dummy_prior":
                cms[tag][label] = cm
    for tag, title in [("validation", "Validation block (last CV fold)"), ("test", "Test period - all shipments"),
                       ("test_sea", "Test period - sea shipments")]:
        plot_confusion_grid(cms[tag], tag, f"Confusion matrices, row-normalised (recall per true class) - {title}")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.RESULT_DIRS["metrics"] / "supervised_metrics.csv", index=False)
    pred_df = test[["shipment_id", "departure_ts", "origin_port_id", "shipping_mode", "delay_status", config.OUTCOME]].copy()
    for name, p in probas.items():
        for i, c in enumerate(config.CLASSES):
            pred_df[f"{name}_p_{c}"] = p[:, i]
        pred_df[f"{name}_pred"] = np.array(config.CLASSES)[p.argmax(1)]
    pred_df.to_csv(config.RESULT_DIRS["metrics"] / "test_predictions.csv", index=False)
    return metrics


# ---------------------------------------------------------------------------------------------
# Anomaly evaluation
# ---------------------------------------------------------------------------------------------
def _cliffs_delta(a, b):
    """P(a > b) - P(a < b) via the Mann-Whitney U statistic."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    u = ss.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def anomaly_metrics(y, score, flag):
    y = np.asarray(y)
    out = {"n": int(len(y)), "anomaly_detection_rate": float(flag.mean())}
    for pos_name, pos in [("delayed", y > 0), ("critical", y == 2)]:
        out.update({
            f"{pos_name}_prevalence": float(pos.mean()),
            f"{pos_name}_precision": precision_score(pos, flag, zero_division=0),
            f"{pos_name}_recall": recall_score(pos, flag, zero_division=0),
            f"{pos_name}_f1": f1_score(pos, flag, zero_division=0),
            f"{pos_name}_roc_auc": roc_auc_score(pos, score),
            f"{pos_name}_pr_auc": average_precision_score(pos, score),
        })
    for i, c in enumerate(config.CLASSES):
        out[f"flag_rate_{c}"] = float(flag[y == i].mean())
        out[f"median_score_{c}"] = float(np.median(score[y == i]))
    out["cliffs_delta_AtRisk_vs_OnTime"] = _cliffs_delta(score[y == 1], score[y == 0])
    out["cliffs_delta_Critical_vs_OnTime"] = _cliffs_delta(score[y == 2], score[y == 0])
    out["kruskal_p"] = float(ss.kruskal(*[score[y == i] for i in LABELS]).pvalue)
    return out


def evaluate_anomaly(ctx, modules):
    train, test, folds, k = ctx["train"], ctx["test"], ctx["folds"], ctx["val_fold"]
    val = train.iloc[folds[k][1]]
    rows, scores = [], {}
    for m in modules:
        bundle = joblib.load(config.MODELS / f"{m.NAME}.joblib")
        pipe, thr = bundle["pipeline"], bundle["threshold"]
        for tag, df in [("validation", val), ("test", test), ("test_sea", test[test["shipping_mode"] == "sea"])]:
            s = m.anomaly_score(pipe, df[ANOMALY_INPUT_COLUMNS])
            rows.append({"model": m.LABEL, "key": m.NAME, "set": tag, "threshold": thr, **anomaly_metrics(df["y"].values, s, s > thr)})
            if tag == "test":
                scores[m.LABEL] = (s, thr)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.RESULT_DIRS["anomaly"] / "anomaly_metrics.csv", index=False)

    y = test["y"].values
    out = test[["shipment_id", "departure_ts", "origin_port_id", "delay_status"]].copy()
    dist = []
    for label, (s, thr) in scores.items():
        key = label.lower().replace(" ", "_").replace("-", "_")
        out[f"{key}_score"], out[f"{key}_flag"] = s, s > thr
        for i, c in enumerate(config.CLASSES):
            q = np.percentile(s[y == i], [5, 25, 50, 75, 95])
            dist.append({"model": label, "class": c, "n": int((y == i).sum()), "mean": s[y == i].mean(),
                         **{f"p{p}": v for p, v in zip([5, 25, 50, 75, 95], q)}, "flag_rate": float((s[y == i] > thr).mean())})
    out.to_csv(config.RESULT_DIRS["anomaly"] / "test_anomaly_scores.csv", index=False)
    pd.DataFrame(dist).to_csv(config.RESULT_DIRS["anomaly"] / "score_distribution_by_class.csv", index=False)
    _plot_anomaly_distributions(scores, y)
    return metrics


def _plot_anomaly_distributions(scores, y):
    fig, axes = plt.subplots(1, len(scores), figsize=(6 * len(scores), 3.8))
    for ax, (label, (s, thr)) in zip(np.atleast_1d(axes), scores.items()):
        data = [s[y == i] for i in LABELS]
        parts = ax.violinplot(data, showextrema=False, showmedians=True)
        for body, c in zip(parts["bodies"], config.CLASSES):
            body.set_facecolor(CLASS_COLORS[c])
            body.set_alpha(0.75)
        parts["cmedians"].set_color("#0b0b0b")
        ax.axhline(thr, color="#52514e", ls="--", lw=1)
        ax.text(3.45, thr, " flag\n threshold", va="center", fontsize=8, color="#52514e")
        rates = [(d > thr).mean() for d in data]
        ax.set_xticks([1, 2, 3], [f"{c}\n{r:.0%} flagged" for c, r in zip(config.CLASSES, rates)])
        ax.set_ylabel("anomaly score (higher = more unusual)")
        ax.set_title(f"{label}: score by true class (test period)")
    fig.tight_layout()
    fig.savefig(config.RESULT_DIRS["figures"] / "anomaly_score_distributions.png", bbox_inches="tight")
    plt.close(fig)


def main(ctx, supervised_modules, anomaly_modules):
    setup_plots()
    sup = evaluate_supervised(ctx, supervised_modules)
    ano = evaluate_anomaly(ctx, anomaly_modules)
    return sup, ano
