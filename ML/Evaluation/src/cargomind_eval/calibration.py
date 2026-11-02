"""Probability quality: reliability diagrams, multiclass Brier score, log loss, ECE, confidence
distributions and class separation.

All four classifiers were trained with balanced class weights, which pushes predicted
probabilities toward a uniform class prior. Two post-hoc corrections are evaluated on the test
period, both derived without any test data:
* prior correction - multiply by training class shares / (1/3) and renormalise (no fitting);
* vector scaling - multinomial logistic regression on log-probabilities fitted on the pooled
  out-of-fold validation predictions.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import metrics as M
from .common import CLASS_COLORS, CLASSES, FIG, MODEL_COLORS, OUT, SUP_LABEL, plt, savefig
from .refits import proba_from


def prior_correct(p, prior):
    q = p * (np.asarray(prior) * 3)
    return q / q.sum(1, keepdims=True)


def vector_scaler(p_val, y_val):
    lr = LogisticRegression(C=1e4, max_iter=5000)
    lr.fit(np.log(np.clip(p_val, 1e-9, 1)), y_val)
    return lambda p: lr.predict_proba(np.log(np.clip(p, 1e-9, 1)))


def reliability_points(pos, p, bins=10):
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    return pd.DataFrame({"mean_pred": [p[b == i].mean() for i in np.unique(b)],
                         "observed": [pos[b == i].mean() for i in np.unique(b)],
                         "n": [(b == i).sum() for i in np.unique(b)]})


def _plot_reliability(y, probas, title, path):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    for k, c in enumerate(CLASSES):
        ax = axes[k]
        for name, p in probas.items():
            r = reliability_points(y == k, p[:, k])
            ax.plot(r["mean_pred"], r["observed"], "o-", ms=3.5, lw=1.3, color=MODEL_COLORS[name], label=SUP_LABEL[name])
        ax.plot([0, 1], [0, 1], color="#bbb", ls=":", lw=0.9)
        ax.set_title(f"P({c}) - one-vs-rest")
        ax.set_xlabel("mean predicted probability (decile bin)")
        ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax = axes[3]
    for name, p in probas.items():
        conf, correct = p.max(1), p.argmax(1) == y
        r = reliability_points(correct, conf)
        ax.plot(r["mean_pred"], r["observed"], "o-", ms=3.5, lw=1.3, color=MODEL_COLORS[name], label=SUP_LABEL[name])
    ax.plot([0, 1], [0, 1], color="#bbb", ls=":", lw=0.9)
    ax.set_title("Top-label confidence vs accuracy")
    ax.set_xlabel("confidence of predicted class")
    ax.set_ylabel("share correct")
    ax.set_xlim(0.3, 1)
    ax.set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    fig.suptitle(title, y=1.02)
    savefig(fig, path)


def _plot_confidence_and_separation(y, probas, path):
    fig, axes = plt.subplots(2, len(probas), figsize=(4.2 * len(probas), 7))
    for j, (name, p) in enumerate(probas.items()):
        ax = axes[0, j]
        ax.hist(p.max(1), bins=np.linspace(1 / 3, 1, 25), color=MODEL_COLORS[name], alpha=0.85)
        ax.set_title(f"{SUP_LABEL[name]}: confidence of predicted class")
        ax.set_xlabel("max class probability")
        ax = axes[1, j]
        data = [p[y == k, 2] for k in range(3)]
        parts = ax.violinplot(data, showextrema=False, showmedians=True)
        for body, c in zip(parts["bodies"], CLASSES):
            body.set_facecolor(CLASS_COLORS[c])
            body.set_alpha(0.75)
        parts["cmedians"].set_color("#0b0b0b")
        ax.set_xticks([1, 2, 3], [f"true {c}" for c in CLASSES], fontsize=8)
        ax.set_ylabel("predicted P(Critical)")
        ax.set_title("P(Critical) by true class")
    fig.suptitle("Confidence distribution (top) and separation of P(Critical) between true classes (bottom), test period", y=1.01)
    savefig(fig, path)


def run(sets, oof, prior):
    rows, bins_rows = [], []
    y_val_all = oof["y"].values
    for tag in ["validation", "test"]:
        df, probas = sets[tag]
        y = df["y"].values
        _plot_reliability(y, probas, f"Reliability diagrams ({tag}); dotted line = perfect calibration",
                          FIG["calibration"] / f"reliability_{tag}.png")
        for name, p in probas.items():
            variants = {"raw": p}
            if tag == "test":
                variants["prior-corrected (training shares)"] = prior_correct(p, prior)
                variants["vector scaling (fit on OOF validation)"] = vector_scaler(proba_from(oof, name), y_val_all)(p)
            for v, q in variants.items():
                ece_cls = M.ece_classwise(y, q)[1]
                conf = q.max(1)
                band = (conf >= 0.75) & (conf < 0.85)
                rows.append({"set": tag, "model": SUP_LABEL[name], "probabilities": v,
                             "log_loss": M.log_loss(y, q), "brier_multiclass": M.brier_multiclass(y, q),
                             "ece_top_label": M.ece_top_label(y, q), "ece_classwise_mean": float(np.mean(ece_cls)),
                             **{f"ece_{c}": e for c, e in zip(CLASSES, ece_cls)},
                             **{f"mean_p_{c}": q[:, k].mean() for k, c in enumerate(CLASSES)},
                             **{f"observed_{c}": (y == k).mean() for k, c in enumerate(CLASSES)},
                             "mean_confidence": conf.mean(), "accuracy": (q.argmax(1) == y).mean(),
                             "n_conf_75_85": int(band.sum()), "accuracy_conf_75_85": (q.argmax(1) == y)[band].mean() if band.any() else np.nan,
                             "critical_recall": ((q.argmax(1) == 2) & (y == 2)).sum() / (y == 2).sum(),
                             "macro_f1_argmax": M.classification(y, q)["macro_f1"],
                             "p_critical_median_true_critical": float(np.median(q[y == 2, 2])),
                             "p_critical_median_true_on_time": float(np.median(q[y == 0, 2]))})
            if tag == "test":
                for k, c in enumerate(CLASSES):
                    r = reliability_points(y == k, p[:, k])
                    r.insert(0, "class", c)
                    r.insert(0, "model", SUP_LABEL[name])
                    bins_rows.append(r)
        if tag == "test":
            _plot_confidence_and_separation(y, probas, FIG["calibration"] / "confidence_and_separation_test.png")
            corrected = {n: prior_correct(p, prior) for n, p in probas.items()}
            _plot_reliability(y, corrected, "Reliability after prior correction to training class shares (test)",
                              FIG["calibration"] / "reliability_test_prior_corrected.png")
    out = pd.DataFrame(rows)
    out.to_csv(OUT["metrics"] / "calibration_metrics.csv", index=False)
    pd.concat(bins_rows).to_csv(OUT["metrics"] / "reliability_bins_test.csv", index=False)
    return out
