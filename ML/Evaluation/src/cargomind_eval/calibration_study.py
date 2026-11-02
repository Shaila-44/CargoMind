"""Sigmoid vs isotonic calibration of the classifier probabilities, chosen on validation data only.

Inputs (no model is retrained):
* ML/Evaluation/data/oof_validation_predictions.csv - out-of-fold probabilities on the three time-series CV validation
  blocks (fold 0: 01-17..01-21, fold 1: 01-21..01-25, fold 2: 01-25..01-29), from evaluation-only refits that reproduce
  the stored CV scores exactly (ML/Evaluation/validation/reproducibility_checks.md);
* ML/ML/results/metrics/test_predictions.csv - the saved models' test probabilities (diagnostic only, see below).

Protocol, fixed before any result was computed:
1. Forward-chained validation: fit each calibrator on blocks < k and evaluate it on block k (k = 1, 2), so a
   calibrator is never evaluated on its own fitting rows or on earlier time than it was fitted on.
2. A method is *justified* for a model if, compared with the raw probabilities, it lowers BOTH log loss and top-label
   ECE on BOTH evaluation blocks, and lowers Critical one-vs-rest ROC-AUC by at most 0.005 on the pooled blocks.
   Among justified methods, the lower pooled log loss wins. If none is justified, raw probabilities stay the only output.
3. The chosen method is refitted on all three validation blocks (2,905 shipments) for use at inference.
4. The test period is scored once with the step-3 calibrators as a diagnostic. It is reported, never used for the choice.

Outputs: ML/Evaluation/calibration_study/ (metrics CSVs, decision JSON, reliability figure); the production calibrator
(if justified) is saved by scripts/run_calibration_study.py to ML/ML/models/<model>_calibrator.joblib.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cargomind_ml.calibration import METHODS, OvRCalibrator

from . import metrics as M
from .common import CLASSES, EVAL_ROOT, SUP_LABEL, mlc, plt, savefig

OUT = EVAL_ROOT / "calibration_study"
MODELS = ("xgboost", "logistic_regression")  # primary classifier + research challenger
MAX_AUC_DROP = 0.005
EVAL_FOLDS = (1, 2)


def _proba(df, name):
    return df[[f"{name}_p_{c}" for c in CLASSES]].to_numpy(float)


def _metrics(y, p):
    m = M.classification(y, p)
    return {"n": m["n"], "log_loss": m["log_loss"], "brier_multiclass": m["brier_multiclass"], "ece_top_label": m["ece_top_label"],
            "ece_classwise": m["ece_classwise"], "mean_p_critical": float(p[:, 2].mean()), "observed_critical": float((np.asarray(y) == 2).mean()),
            "mean_p_on_time": float(p[:, 0].mean()), "observed_on_time": float((np.asarray(y) == 0).mean()),
            "macro_f1_argmax": m["macro_f1"], "accuracy_argmax": m["accuracy"], "critical_recall_argmax": m["critical_recall"],
            "critical_precision_argmax": m["critical_precision"], "at_risk_recall_argmax": m["at_risk_recall"],
            "critical_roc_auc": m["critical_roc_auc_ovr"], "roc_auc_ovr_macro": m["roc_auc_ovr_macro"]}


def forward_validation(oof: pd.DataFrame) -> pd.DataFrame:
    rows, pooled = [], {}
    for name in MODELS:
        for k in EVAL_FOLDS:
            fit, ev = oof[oof["fold"] < k], oof[oof["fold"] == k]
            p_ev, y_ev = _proba(ev, name), ev["y"].to_numpy()
            variants = {"raw": p_ev}
            for method in METHODS:
                variants[method] = OvRCalibrator(method).fit(_proba(fit, name), fit["y"].to_numpy()).transform(p_ev)
            for v, p in variants.items():
                rows.append({"model": name, "eval_block": f"fold {k}", "fit_blocks": f"folds < {k}", "n_fit": len(fit), "variant": v,
                             **_metrics(y_ev, p)})
                pooled.setdefault((name, v), []).append((y_ev, p))
    for (name, v), parts in pooled.items():
        y = np.concatenate([a for a, _ in parts])
        p = np.vstack([b for _, b in parts])
        rows.append({"model": name, "eval_block": "pooled folds 1-2", "fit_blocks": "forward-chained", "n_fit": None, "variant": v, **_metrics(y, p)})
    return pd.DataFrame(rows)


def decide(val: pd.DataFrame) -> dict:
    out = {}
    for name in MODELS:
        d = val[val["model"] == name]
        checks = {}
        for method in METHODS:
            per_block = []
            for k in EVAL_FOLDS:
                r = d[(d["eval_block"] == f"fold {k}") & (d["variant"] == "raw")].iloc[0]
                c = d[(d["eval_block"] == f"fold {k}") & (d["variant"] == method)].iloc[0]
                per_block.append({"block": k, "log_loss_better": bool(c["log_loss"] < r["log_loss"]),
                                  "ece_better": bool(c["ece_top_label"] < r["ece_top_label"])})
            pr = d[(d["eval_block"] == "pooled folds 1-2") & (d["variant"] == "raw")].iloc[0]
            pc = d[(d["eval_block"] == "pooled folds 1-2") & (d["variant"] == method)].iloc[0]
            auc_drop = float(pr["critical_roc_auc"] - pc["critical_roc_auc"])
            ok = all(b["log_loss_better"] and b["ece_better"] for b in per_block) and auc_drop <= MAX_AUC_DROP
            checks[method] = {"justified": ok, "per_block": per_block, "pooled_log_loss": float(pc["log_loss"]),
                              "pooled_raw_log_loss": float(pr["log_loss"]), "critical_auc_drop": auc_drop}
        justified = [m for m in METHODS if checks[m]["justified"]]
        choice = min(justified, key=lambda m: checks[m]["pooled_log_loss"]) if justified else None
        out[name] = {"chosen_method": choice, "checks": checks}
    return out


def test_diagnostic(oof: pd.DataFrame, test_pred: pd.DataFrame, decision: dict) -> tuple[pd.DataFrame, dict]:
    rows, fitted = [], {}
    y = test_pred["delay_status"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    for name in MODELS:
        p_raw = _proba(test_pred, name)
        rows.append({"model": name, "set": "test (diagnostic)", "variant": "raw", **_metrics(y, p_raw)})
        for method in METHODS:
            cal = OvRCalibrator(method).fit(_proba(oof, name), oof["y"].to_numpy())
            fitted[(name, method)] = cal
            rows.append({"model": name, "set": "test (diagnostic)", "variant": method, **_metrics(y, cal.transform(p_raw))})
    return pd.DataFrame(rows), fitted


def reliability_figure(oof: pd.DataFrame, test_pred: pd.DataFrame, fitted: dict) -> None:
    from .calibration import reliability_points

    y_test = test_pred["delay_status"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"raw": "#8a8985", "sigmoid": "#2a78d6", "isotonic": "#e34948"}
    # left: forward-chained validation (fit folds 0-1 -> fold 2); right: test diagnostic (fit on all validation blocks)
    fit, ev = oof[oof["fold"] < 2], oof[oof["fold"] == 2]
    for ax, (title, y, base, cal_for) in zip(axes, [
            ("Validation block 2 (calibrators fitted on blocks 0-1)", ev["y"].to_numpy(), _proba(ev, "xgboost"),
             lambda m: OvRCalibrator(m).fit(_proba(fit, "xgboost"), fit["y"].to_numpy())),
            ("Test period, diagnostic (calibrators fitted on all validation blocks)", y_test, _proba(test_pred, "xgboost"),
             lambda m: fitted[("xgboost", m)])]):
        ax.plot([0, 1], [0, 1], color="#c3c2be", lw=1)
        for v in ("raw", *METHODS):
            p = base if v == "raw" else cal_for(v).transform(base)
            pts = reliability_points((y == 2).astype(float), p[:, 2])
            ax.plot(pts["mean_pred"], pts["observed"], marker="o", ms=4, color=colors[v], label=v)
        ax.set(title=title, xlabel="mean predicted P(Critical)", xlim=(0, 1), ylim=(0, 1))
        ax.title.set_fontsize(10)
    axes[0].set_ylabel("observed Critical share")
    axes[0].legend()
    fig.suptitle("XGBoost: reliability of P(Critical), raw vs calibrated (10 equal-count bins)")
    fig.tight_layout()
    savefig(fig, OUT / "reliability_xgboost_critical.png")


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(EVAL_ROOT / "data" / "oof_validation_predictions.csv")
    test_pred = pd.read_csv(mlc.RESULT_DIRS["metrics"] / "test_predictions.csv")
    val = forward_validation(oof)
    val.to_csv(OUT / "validation_forward_chained.csv", index=False)
    decision = decide(val)
    test, fitted = test_diagnostic(oof, test_pred, decision)
    test.to_csv(OUT / "test_diagnostic.csv", index=False)
    reliability_figure(oof, test_pred, fitted)
    info = {"protocol": __doc__.split("Protocol, fixed before any result was computed:")[1].split("Outputs:")[0].strip(),
            "models": {n: SUP_LABEL[n] for n in MODELS}, "decision": decision,
            "validation_blocks": {int(k): int(n) for k, n in oof.groupby("fold").size().items()},
            "validation_critical_share_by_block": {int(k): float((g["y"] == 2).mean()) for k, g in oof.groupby("fold")},
            "test_critical_share": float((test_pred["delay_status"] == "Critical").mean())}
    (OUT / "decision.json").write_text(json.dumps(info, indent=2))
    return {"validation": val, "test": test, "decision": decision, "fitted": fitted, "oof": oof}
