"""Consolidated comparison tables.

Supervised classifiers and anomaly detectors are tabulated separately: they answer different
questions (which of three risk classes vs. how unusual the operating state is).

Selection rules (fixed before looking at test results):
* best overall supervised classifier: highest mean macro-F1 across the time-series CV folds;
* best Critical-class detector: highest mean Critical recall across the CV folds, reported with
  Critical precision so that recall bought by over-flagging is visible;
* best anomaly detector: highest validation PR-AUC for "delayed" (At-Risk or Critical).
Test-period numbers are reported next to each choice as the out-of-time confirmation.
"""
# =============================================================================
# WHAT DOES THIS MODULE DO?
# =============================================================================
# In machine learning, training several competing models is like holding a sports tournament.
# Every model (Logistic Regression, Random Forest, XGBoost, LightGBM, Isolation Forest,
# One-Class SVM) competes to see which one performs best.
#
# This file acts as the official "TOURNAMENT JUDGE":
# 1. It gathers performance report cards from all models across cross-validation and test periods.
# 2. It compiles clear, structured comparison tables.
# 3. It applies predefined, rigorous selection criteria to pick the winning models:
#    - Best Overall Classifier (highest balanced accuracy / Macro-F1)
#    - Best Critical Delay Catcher (highest recall on severe delays)
#    - Most Interpretable Model (easiest for humans to audit)
#    - Best Anomaly Detector (best at flagging abnormal port conditions)
# 4. It exports a comprehensive markdown report (`model_comparison.md`) and JSON summary.
# =============================================================================

# --- Standard library imports ---
import json  # For saving the tournament winners to selection.json

# --- Third-party library imports ---
import pandas as pd  # For creating and pivoting tabular comparison tables

# --- Internal project imports ---
from . import config

# Output folder where all comparison tables and reports will be saved
OUT = config.RESULT_DIRS["comparison"]


def _fmt(df, digits=3):
    """Helper function to format a pandas DataFrame as a clean Markdown table with rounded decimals."""
    return df.to_markdown(index=False, floatfmt=f".{digits}f")


def supervised_tables(metrics, modules):
    """Compile side-by-side comparison tables for all supervised classifiers.

    Computes:
      1. Cross-validation (CV) performance table (from hyperparameter tuning folds)
      2. Validation block performance
      3. Out-of-time Test set performance (the latest 25% of shipments never seen during tuning)
      4. Sea-only test performance (testing performance specifically on maritime cargo)
    """
    # --- Step 1: Gather Cross-Validation tuning scores ---
    cv = {}
    for m in modules:
        # Load the configuration JSON saved during training
        cfg = json.loads((config.MODELS / f"{m.NAME}_config.json").read_text())
        s = cfg["cv_scores_best"]
        cv[m.LABEL] = {
            "CV Macro F1 (mean of 3 folds)": s["f1_macro"]["cv_mean"],
            "CV Macro F1 sd": s["f1_macro"]["cv_std"],
            "CV Critical Recall": s["recall_critical"]["cv_mean"],
            "CV At-Risk Recall": s["recall_at_risk"]["cv_mean"],
        }
    cv = pd.DataFrame(cv).T

    # --- Step 2: Extract test, validation, and sub-population tables ---
    tables = {}
    for tag in ["validation", "test", "test_sea"]:
        d = metrics[metrics["set"] == tag].set_index("model")
        t = pd.DataFrame({
            "Accuracy": d["accuracy"],
            "Macro Precision": d["macro_precision"],
            "Macro Recall": d["macro_recall"],
            "Macro F1": d["macro_f1"],
            "Weighted F1": d["weighted_f1"],
            "Critical Recall": d["critical_recall"],
            "Critical Precision": d["critical_precision"],
            "At-Risk Recall": d["at_risk_recall"],
            "ROC-AUC (OvR macro)": d.get("roc_auc_ovr_macro"),
            "Critical→On-Time rate": d["critical_predicted_on_time_rate"],
            "Log loss": d["log_loss"],
        })
        # For the final test set, also include 95% bootstrap confidence intervals
        if tag == "test":
            t["Macro F1 95% CI (day bootstrap)"] = d["macro_f1_ci95"].map(lambda v: f"{v[0]:.3f}–{v[1]:.3f}")
            t["Critical Recall 95% CI"] = d["critical_recall_ci95"].map(lambda v: f"{v[0]:.3f}–{v[1]:.3f}")

        tables[tag] = t.reset_index().rename(columns={"model": "Model"})
        # Save as CSV for auditing and dashboard ingestion
        tables[tag].to_csv(OUT / f"supervised_comparison_{tag}.csv", index=False)

    cv.reset_index().rename(columns={"index": "Model"}).to_csv(OUT / "supervised_comparison_cv.csv", index=False)
    return tables, cv


def anomaly_tables(metrics):
    """Compile comparison tables for anomaly detectors (Isolation Forest vs One-Class SVM).

    Evaluates anomaly detectors under two lenses:
      1. 'delayed'  : How well do high anomaly scores correlate with ANY delay (At-Risk or Critical)?
      2. 'critical' : How well do high anomaly scores correlate specifically with SEVERE delays?
    """
    tables = {}
    for pos in ["delayed", "critical"]:
        for tag in ["validation", "test"]:
            d = metrics[metrics["set"] == tag].set_index("model")
            t = pd.DataFrame({
                "Precision": d[f"{pos}_precision"],
                "Recall": d[f"{pos}_recall"],
                "F1": d[f"{pos}_f1"],
                "ROC-AUC": d[f"{pos}_roc_auc"],
                "PR-AUC": d[f"{pos}_pr_auc"],
                "PR-AUC baseline (prevalence)": d[f"{pos}_prevalence"],
                "Detection rate": d["anomaly_detection_rate"],
                "Flag rate On-Time": d["flag_rate_On-Time"],
                "Flag rate At-Risk": d["flag_rate_At-Risk"],
                "Flag rate Critical": d["flag_rate_Critical"],
                "Cliff's δ Critical vs On-Time": d["cliffs_delta_Critical_vs_OnTime"],
            }).reset_index().rename(columns={"model": "Model"})
            tables[(pos, tag)] = t
            t.to_csv(OUT / f"anomaly_comparison_{pos}_{tag}.csv", index=False)
    return tables


def main(sup_metrics, ano_metrics, modules):
    """Main execution function for model comparison.

    Generates all comparison tables, evaluates selection criteria, saves
    'selection.json' with official model choices, and writes 'model_comparison.md'.
    """
    sup, cv = supervised_tables(sup_metrics, modules)
    ano = anomaly_tables(ano_metrics)
    labels = [m.LABEL for m in modules]

    # --- Apply predefined selection rules ---
    # 1. Best overall classifier: Highest CV Macro F1 across the 3 time-series folds
    best_overall = cv["CV Macro F1 (mean of 3 folds)"].idxmax()

    # 2. Best critical detector: Highest CV Critical Recall (catches the most disasters)
    best_critical = cv["CV Critical Recall"].idxmax()

    # 3. Best anomaly detector: Highest Precision-Recall AUC on the validation set
    val_ano = ano[("delayed", "validation")].set_index("Model")
    best_anomaly = val_ano["PR-AUC"].idxmax()

    test = sup["test"].set_index("Model")
    test_ano = ano[("delayed", "test")].set_index("Model")

    # Record the official selections into a dictionary
    picks = {
        "best_overall_supervised": best_overall,
        "best_critical_detector": best_critical,
        "most_interpretable": "Logistic Regression",
        "best_anomaly_detector": best_anomaly,
        "test_best_macro_f1": test.loc[labels, "Macro F1"].idxmax(),
        "test_best_critical_recall": test.loc[labels, "Critical Recall"].idxmax(),
    }
    # Save the decision to JSON so the inference engine knows which models to load by default
    (OUT / "selection.json").write_text(json.dumps(picks, indent=2))

    # --- Process Ablation and Sensitivity results for the final report ---
    abl = pd.read_csv(OUT / "ablation_and_sensitivity.csv")
    abl_wide = abl.pivot_table(index="variant", columns="model", values="macro_f1", sort=False).round(3)
    abl_crit = abl.pivot_table(index="variant", columns="model", values="critical_recall", sort=False).round(3)

    # --- Construct Markdown report ---
    md = [
        "# Model comparison (generated by `compare_models.py`)", "",
        "All numbers are computed by the pipeline; nothing is hand-entered. Supervised and anomaly models are "
        "**not** compared with each other: they solve different tasks.", "",
        "## Supervised classifiers - test period (latest 25 % of shipments, never used for tuning)", "",
        _fmt(sup["test"][["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "Critical Recall"]]), "",
        "Additional test metrics:", "",
        _fmt(sup["test"][["Model", "Critical Precision", "At-Risk Recall", "ROC-AUC (OvR macro)", "Critical→On-Time rate",
                          "Log loss", "Macro F1 95% CI (day bootstrap)", "Critical Recall 95% CI"]]), "",
        "## Supervised classifiers - test period, sea shipments only", "",
        _fmt(sup["test_sea"][["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "Critical Recall"]]), "",
        "## Supervised classifiers - tuning (time-series CV) and validation block", "",
        _fmt(cv.reset_index().rename(columns={"index": "Model"})), "",
        _fmt(sup["validation"][["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "Critical Recall"]]), "",
        "## Anomaly detectors - test period (positive = At-Risk or Critical)", "",
        _fmt(ano[("delayed", "test")][["Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]]), "",
        _fmt(ano[("delayed", "test")][["Model", "PR-AUC baseline (prevalence)", "Detection rate", "Flag rate On-Time",
                                        "Flag rate At-Risk", "Flag rate Critical", "Cliff's δ Critical vs On-Time"]]), "",
        "## Anomaly detectors - test period (positive = Critical only)", "",
        _fmt(ano[("critical", "test")][["Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "PR-AUC baseline (prevalence)"]]), "",
        "## Ablations and sensitivity checks - test macro-F1 (tuned hyperparameters, refit only)", "",
        abl_wide.reset_index().to_markdown(index=False), "",
        "Test Critical recall for the same variants:", "",
        abl_crit.reset_index().to_markdown(index=False), "",
        "## Selections (rules fixed before test evaluation)", "",
        f"* Best overall supervised classifier (CV macro-F1): **{best_overall}** "
        f"(test macro-F1 {test.loc[best_overall, 'Macro F1']:.3f})",
        f"* Best Critical-class detector (CV Critical recall): **{best_critical}** "
        f"(test Critical recall {test.loc[best_critical, 'Critical Recall']:.3f}, precision {test.loc[best_critical, 'Critical Precision']:.3f})",
        "* Most interpretable: **Logistic Regression** (linear in standardised inputs; coefficients readable directly)",
        f"* Best anomaly detector (validation PR-AUC, delayed): **{best_anomaly}** (test PR-AUC {test_ano.loc[best_anomaly, 'PR-AUC']:.3f})",
        f"* For reference, on the test period the highest macro-F1 is {picks['test_best_macro_f1']} and the highest Critical recall is "
        f"{picks['test_best_critical_recall']}.", ""
    ]
    (OUT / "model_comparison.md").write_text("\n".join(md))
    return picks
