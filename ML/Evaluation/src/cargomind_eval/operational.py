"""Operational view: alert volumes, missed Critical shipments, and decision policies chosen on
validation data only (pooled out-of-fold predictions of the three CV blocks), then applied once to
the test period.

Policy family: predict Critical when P(Critical) >= t; otherwise take the more probable of On-Time
and At-Risk. Three rules for choosing t on validation:
* "argmax" - the models' default decision (no tuning);
* "max macro-F1" - t that maximises validation macro-F1;
* "Critical recall >= 0.80" - the largest t whose validation Critical recall is at least 0.80;
* "min expected cost" - t that minimises an illustrative cost matrix (an assumption, stated below).
"""
import numpy as np
import pandas as pd

from . import metrics as M
from .common import CLASSES, FIG, MODEL_COLORS, OUT, SUP_LABEL, plt, savefig
from .refits import proba_from

# Illustrative misclassification costs (rows = true, cols = predicted). Assumed, not estimated:
# missing a Critical delay entirely is ~10x a false high-severity alarm.
COST = np.array([[0.0, 0.5, 1.0],
                 [3.0, 0.0, 0.5],
                 [10.0, 4.0, 0.0]])
GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def decide(p, t):
    pred = np.where(p[:, 1] > p[:, 0], 1, 0)
    return np.where(p[:, 2] >= t, 2, pred)


def _eval(y, pred):
    cm = M.confusion(y, pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.nan_to_num(np.diag(cm) / cm.sum(0))
        rec = np.nan_to_num(np.diag(cm) / cm.sum(1))
        f1 = np.nan_to_num(2 * prec * rec / (prec + rec))
    n = cm.sum()
    return {"macro_f1": f1.mean(), "accuracy": np.trace(cm) / n, "critical_recall": rec[2], "critical_precision": prec[2],
            "at_risk_recall": rec[1], "at_risk_precision": prec[1], "on_time_recall": rec[0],
            "critical_alerts_per_1000": cm[:, 2].sum() / n * 1000, "any_alert_per_1000": cm[:, 1:].sum() / n * 1000,
            "missed_critical_per_1000": (cm[2, 0] + cm[2, 1]) / n * 1000, "critical_as_on_time_per_1000": cm[2, 0] / n * 1000,
            "false_critical_alerts_per_1000": (cm[0, 2] + cm[1, 2]) / n * 1000, "on_time_as_critical_per_1000": cm[0, 2] / n * 1000,
            "expected_cost_per_shipment": float((cm * COST).sum() / n),
            "critical_caught": int(cm[2, 2]), "critical_missed": int(cm[2, 0] + cm[2, 1]), "critical_as_on_time": int(cm[2, 0]),
            "false_critical_alerts": int(cm[0, 2] + cm[1, 2])}


def choose_thresholds(y_val, p_val):
    curve = pd.DataFrame([{"t": t, **_eval(y_val, decide(p_val, t))} for t in GRID])
    picks = {"max macro-F1": float(curve.loc[curve["macro_f1"].idxmax(), "t"]),
             "min expected cost": float(curve.loc[curve["expected_cost_per_shipment"].idxmin(), "t"])}
    ok = curve[curve["critical_recall"] >= 0.80]
    picks["Critical recall >= 0.80"] = float(ok["t"].max()) if len(ok) else float(GRID.min())
    return picks, curve


def run(oof, test, test_proba):
    y_val, y_test = oof["y"].values, test["y"].values
    rows, curves = [], []
    for name, p_test in test_proba.items():
        p_val = proba_from(oof, name)
        picks, curve = choose_thresholds(y_val, p_val)
        curve.insert(0, "model", SUP_LABEL[name])
        curves.append(curve)
        rows.append({"model": SUP_LABEL[name], "policy": "argmax (default)", "threshold_t": np.nan,
                     **{f"val_{k}": v for k, v in _eval(y_val, p_val.argmax(1)).items() if k in ("macro_f1", "critical_recall", "critical_precision")},
                     **_eval(y_test, p_test.argmax(1))})
        for pol, t in picks.items():
            rows.append({"model": SUP_LABEL[name], "policy": pol, "threshold_t": t,
                         **{f"val_{k}": v for k, v in _eval(y_val, decide(p_val, t)).items() if k in ("macro_f1", "critical_recall", "critical_precision")},
                         **_eval(y_test, decide(p_test, t))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT["metrics"] / "operational_policies.csv", index=False)
    curves = pd.concat(curves)
    curves.to_csv(OUT["metrics"] / "threshold_curves_validation.csv", index=False)

    # test-period trade-off curve (descriptive only: thresholds were picked on validation above)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for name, p_test in test_proba.items():
        tc = pd.DataFrame([{"t": t, **_eval(y_test, decide(p_test, t))} for t in GRID])
        axes[0].plot(tc["false_critical_alerts_per_1000"], tc["critical_recall"], color=MODEL_COLORS[name], lw=1.5, label=SUP_LABEL[name])
        sel = out[(out["model"] == SUP_LABEL[name])]
        axes[0].scatter(sel["false_critical_alerts_per_1000"], sel["critical_recall"], color=MODEL_COLORS[name], s=22, zorder=3)
        vc = curves[curves["model"] == SUP_LABEL[name]]
        axes[1].plot(vc["t"], vc["critical_recall"], color=MODEL_COLORS[name], lw=1.4, label=f"{SUP_LABEL[name]} recall")
        axes[1].plot(vc["t"], vc["critical_precision"], color=MODEL_COLORS[name], lw=1.4, ls="--")
    axes[0].set_xlabel("false Critical alerts per 1,000 shipments (test)")
    axes[0].set_ylabel("Critical recall (test)")
    axes[0].set_title("Test trade-off; dots = policies chosen on validation")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("threshold t on P(Critical)")
    axes[1].set_ylabel("validation Critical recall (solid) / precision (dashed)")
    axes[1].set_title("Validation (pooled CV blocks) threshold curves")
    axes[1].legend(fontsize=8)
    savefig(fig, FIG["operational"] / "critical_alert_tradeoff.png")
    return out, curves


__all__ = ["run", "CLASSES"]
