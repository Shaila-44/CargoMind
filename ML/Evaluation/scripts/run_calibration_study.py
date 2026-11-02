"""Sigmoid vs isotonic calibration study (validation only; no model is retrained). See cargomind_eval.calibration_study.

Run from the CargoMind project root:

    .venv/bin/python ML/Evaluation/scripts/run_calibration_study.py

Writes ML/Evaluation/calibration_study/. If calibration is justified for the primary classifier under the pre-registered
rule, the chosen calibrator (fitted on all three validation blocks) is saved to ML/ML/models/xgboost_calibrator.joblib
with a JSON description; otherwise any previously saved calibrator is left untouched and nothing is written there.
"""
import json
import warnings

import joblib

from cargomind_eval import calibration_study as cs
from cargomind_ml import config as mlc

warnings.filterwarnings("ignore")
PRIMARY = "xgboost"


def main():
    res = cs.run()
    cols = ["model", "eval_block", "variant", "log_loss", "brier_multiclass", "ece_top_label", "ece_classwise", "mean_p_critical",
            "observed_critical", "macro_f1_argmax", "critical_recall_argmax", "critical_precision_argmax", "critical_roc_auc"]
    print(res["validation"][cols].round(4).to_string(index=False))
    print(res["test"][[c for c in cols if c != "eval_block"]].round(4).to_string(index=False))
    print(json.dumps({k: v["chosen_method"] for k, v in res["decision"].items()}))
    method = res["decision"][PRIMARY]["chosen_method"]
    if method is None:
        print(f"calibration not justified for {PRIMARY}; no calibrator saved")
        return
    cal = res["fitted"][(PRIMARY, method)]
    joblib.dump(cal, mlc.MODELS / f"{PRIMARY}_calibrator.joblib")
    (mlc.MODELS / f"{PRIMARY}_calibrator.json").write_text(json.dumps({
        "model": PRIMARY, "method": method, "scheme": "one-vs-rest per class, renormalised (cargomind_ml.calibration.OvRCalibrator)",
        "fitted_on": "out-of-fold probabilities of the three time-series CV validation blocks (ML/Evaluation/data/oof_validation_predictions.csv)",
        "n_fit": cal.n_fit_, "class_counts_fit": dict(zip(mlc.CLASSES, cal.class_counts_)),
        "selection": "ML/Evaluation/calibration_study/decision.json", "raw_probabilities_unchanged": True}, indent=2))
    print(f"saved {PRIMARY} calibrator ({method})")


if __name__ == "__main__":
    main()
