"""Supervised classifier evaluation: metric verification, per-class analysis, confusion matrices,
class-boundary error analysis, ROC and precision-recall curves."""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from . import metrics as M
from .common import CLASS_COLORS, CLASSES, FEATURES, FIG, MODEL_COLORS, OUT, SUP_LABEL, mlc, plt, savefig
from .refits import proba_from

ERROR_MEANING = {
    (2, 0): ("Critical predicted On-Time", "missed warning: a >48 h delay gets no alert at all - the most dangerous error"),
    (2, 1): ("Critical predicted At-Risk", "severity underestimated: an alert is raised but at the lower tier, so the response may be too weak"),
    (1, 0): ("At-Risk predicted On-Time", "missed intervention window for a 12-48 h delay"),
    (1, 2): ("At-Risk predicted Critical", "over-escalation: a moderate delay is handled as severe (extra cost, but the shipment is delayed)"),
    (0, 1): ("On-Time predicted At-Risk", "unnecessary low-tier alert; cheap individually, erodes trust if frequent"),
    (0, 2): ("On-Time predicted Critical", "false high-severity alarm: expensive mitigation (re-routing, customer notices) for nothing"),
}


def evaluation_sets(ctx, oof, test_proba):
    """{set name: (frame, {model: proba})}. Validation = fold 2 block; folds 0-1 kept for temporal analysis."""
    test = ctx["test"]
    sea = (test["shipping_mode"] == "sea").values
    sets = {}
    for k, g in oof.groupby("fold"):
        sets[f"cv_fold_{k}"] = (g, {n: proba_from(g, n) for n in SUP_LABEL})
    sets["validation"] = sets[f"cv_fold_{ctx['val_fold']}"]
    sets["test"] = (test, test_proba)
    sets["test_sea"] = (test[sea], {n: p[sea] for n, p in test_proba.items()})
    return sets


def metric_table(sets, dummy_test_proba):
    rows = []
    for tag, (df, probas) in sets.items():
        for name, p in probas.items():
            rows.append({"model": SUP_LABEL[name], "key": name, "set": tag, **M.classification(df["y"].values, p)})
        if tag == "test":
            rows.append({"model": "Majority-class reference", "key": "dummy_prior", "set": tag,
                         **M.classification(df["y"].values, dummy_test_proba)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT["metrics"] / "supervised_metrics.csv", index=False)
    return out


def verify_against_stored(table):
    """Independent metric implementation vs. ml/results/metrics/supervised_metrics.csv."""
    stored = pd.read_csv(mlc.RESULT_DIRS["metrics"] / "supervised_metrics.csv")
    cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1",
            "log_loss", "roc_auc_ovr_macro", "roc_auc_ovr_weighted", "on_time_recall", "at_risk_recall", "critical_recall",
            "critical_precision", "critical_f1", "critical_roc_auc_ovr", "ordinal_mae"]
    rename = {"critical_predicted_on_time_rate": "critical_pred_on_time_rate"}
    stored = stored.rename(columns=rename)
    rows = []
    for tag in ["validation", "test", "test_sea"]:
        for _, s in stored[stored["set"] == tag].iterrows():
            mine = table[(table["set"] == tag) & (table["key"] == s["key"])]
            if mine.empty:
                continue
            mine = mine.iloc[0]
            for c in cols + ["critical_pred_on_time_rate"]:
                if c in s and pd.notna(s[c]):
                    rows.append({"set": tag, "model": s["model"], "metric": c, "stored": s[c], "recomputed": mine[c],
                                 "abs_diff": abs(s[c] - mine[c])})
    out = pd.DataFrame(rows)
    out.to_csv(OUT["validation"] / "metric_verification.csv", index=False)
    return out


def per_class_table(table):
    rows = []
    for _, r in table.iterrows():
        for c in CLASSES:
            key = c.lower().replace("-", "_")
            rows.append({"model": r["model"], "set": r["set"], "class": c, "support": r[f"{key}_support"],
                         "prevalence": r[f"{key}_prevalence"], "predicted_share": r[f"{key}_predicted_share"],
                         "precision": r[f"{key}_precision"], "recall": r[f"{key}_recall"], "f1": r[f"{key}_f1"],
                         "roc_auc_ovr": r[f"{key}_roc_auc_ovr"], "pr_auc_ovr": r[f"{key}_pr_auc_ovr"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT["metrics"] / "per_class_metrics.csv", index=False)
    return out


def confusion_outputs(sets):
    rows = []
    for tag in ["validation", "test", "test_sea"]:
        df, probas = sets[tag]
        y = df["y"].values
        fig, axes = plt.subplots(1, len(probas), figsize=(3.7 * len(probas), 3.8))
        for ax, (name, p) in zip(axes, probas.items()):
            cm = M.confusion(y, p.argmax(1))
            share = cm / cm.sum(1, keepdims=True)
            ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{share[i, j]:.0%}\n({cm[i, j]:,})", ha="center", va="center", fontsize=8.5,
                            color="white" if share[i, j] > 0.55 else "#0b0b0b")
            ax.set_xticks(range(3), CLASSES, fontsize=8)
            ax.set_yticks(range(3), CLASSES, fontsize=8)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true" if ax is axes[0] else "")
            ax.set_title(SUP_LABEL[name])
            ax.grid(False)
            for (i, j), (label, meaning) in ERROR_MEANING.items():
                rows.append({"set": tag, "model": SUP_LABEL[name], "error": label, "count": int(cm[i, j]),
                             "share_of_true_class": share[i, j], "share_of_all_shipments": cm[i, j] / cm.sum(),
                             "operational_meaning": meaning})
        fig.suptitle(f"Confusion matrices ({tag}), row-normalised: share of each true class", y=1.02)
        savefig(fig, FIG["confusion_matrices"] / f"confusion_{tag}.png")
    out = pd.DataFrame(rows)
    out.to_csv(OUT["metrics"] / "confusion_analysis.csv", index=False)
    return out


DELAY_BINS = [-np.inf, 0, 6, 10, 12, 14, 18, 30, 42, 46, 48, 50, 54, 72, 96, np.inf]


def boundary_analysis(test, test_proba):
    """Prediction mix as a function of the realised delay, to see whether errors sit at the 12 h / 48 h cut-offs."""
    d = test[mlc.OUTCOME].values
    bins = pd.cut(d, DELAY_BINS, right=True)
    rows = []
    for name, p in test_proba.items():
        pred = p.argmax(1)
        y = test["y"].values
        dist = np.minimum(np.abs(d - 12), np.abs(d - 48))
        for b in bins.categories:
            m = bins == b
            if m.sum() == 0:
                continue
            rows.append({"model": SUP_LABEL[name], "delay_bin_h": str(b), "n": int(m.sum()),
                         "error_rate": float((pred[m] != y[m]).mean()),
                         **{f"pred_{c}": float((pred[m] == k).mean()) for k, c in enumerate(CLASSES)},
                         "mean_p_critical": float(p[m, 2].mean())})
        for c_idx, c in enumerate(CLASSES):
            m = y == c_idx
            rows.append({"model": SUP_LABEL[name], "delay_bin_h": f"summary {c}: median h to nearest threshold, errors vs correct",
                         "n": int(m.sum()), "error_rate": float((pred[m] != y[m]).mean()),
                         "median_dist_errors_h": float(np.median(dist[m & (pred != y)])) if (m & (pred != y)).any() else np.nan,
                         "median_dist_correct_h": float(np.median(dist[m & (pred == y)])) if (m & (pred == y)).any() else np.nan,
                         "share_errors_within_6h_of_threshold": float((dist[m & (pred != y)] <= 6).mean()) if (m & (pred != y)).any() else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(OUT["metrics"] / "boundary_analysis.csv", index=False)

    # figure: prediction mix by realised delay for each model
    grid = out[~out["delay_bin_h"].str.startswith("summary")]
    fig, axes = plt.subplots(1, len(test_proba), figsize=(4.4 * len(test_proba), 3.9), sharey=True)
    for ax, name in zip(axes, test_proba):
        g = grid[grid["model"] == SUP_LABEL[name]]
        x = np.arange(len(g))
        bottom = np.zeros(len(g))
        for c in CLASSES:
            ax.bar(x, g[f"pred_{c}"], bottom=bottom, color=CLASS_COLORS[c], width=0.85, label=f"predicted {c}")
            bottom += g[f"pred_{c}"].values
        labels = [s.replace("(", "").replace("]", "").replace(", ", "–").replace("-inf–0.0", "≤0").replace("96.0–inf", ">96").replace(".0", "")
                  for s in g["delay_bin_h"]]
        ax.set_xticks(x, labels, rotation=70, fontsize=7.5)
        cats = list(g["delay_bin_h"])
        for thr in (12, 48):
            pos = [i for i, s in enumerate(cats) if s.endswith(f", {thr}.0]")][0] + 0.5
            ax.axvline(pos, color="#0b0b0b", lw=1, ls="--")
        ax.set_title(SUP_LABEL[name])
        ax.set_xlabel("realised delay (h)")
    axes[0].set_ylabel("share of shipments in the delay bin")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Predicted class by realised delay (test). Dashed lines: 12 h and 48 h class boundaries", y=1.02)
    savefig(fig, FIG["confusion_matrices"] / "prediction_mix_by_realised_delay_test.png")
    return out


def curves(sets):
    for tag in ["validation", "test"]:
        df, probas = sets[tag]
        y = df["y"].values
        for kind, folder in [("roc", "roc_curves"), ("pr", "precision_recall")]:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
            for k, (ax, c) in enumerate(zip(axes, CLASSES)):
                pos = y == k
                for name, p in probas.items():
                    if kind == "roc":
                        fpr, tpr, _ = roc_curve(pos, p[:, k])
                        ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=1.4, label=f"{SUP_LABEL[name]} ({M.auc_rank(pos, p[:, k]):.3f})")
                    else:
                        pr, rc, _ = precision_recall_curve(pos, p[:, k])
                        ax.plot(rc, pr, color=MODEL_COLORS[name], lw=1.4, label=f"{SUP_LABEL[name]} ({M.average_precision(pos, p[:, k]):.3f})")
                if kind == "roc":
                    ax.plot([0, 1], [0, 1], color="#bbb", lw=0.8, ls=":")
                    ax.set_xlabel("false-positive rate")
                    ax.set_ylabel("true-positive rate")
                else:
                    ax.axhline(pos.mean(), color="#bbb", lw=0.8, ls=":", label=f"prevalence {pos.mean():.2f}")
                    ax.set_xlabel("recall")
                    ax.set_ylabel("precision")
                ax.set_title(f"{c} vs rest")
                ax.legend(fontsize=7.5, loc="lower right" if kind == "roc" else "upper right")
            fig.suptitle(f"{'ROC' if kind == 'roc' else 'Precision-recall'} curves, one-vs-rest ({tag}); legend: "
                         f"{'AUC' if kind == 'roc' else 'average precision'}", y=1.02)
            savefig(fig, FIG[folder] / f"{kind}_ovr_{tag}.png")


def dummy_test_proba(test):
    return joblib.load(mlc.MODELS / "dummy_prior.joblib").predict_proba(test[FEATURES])
