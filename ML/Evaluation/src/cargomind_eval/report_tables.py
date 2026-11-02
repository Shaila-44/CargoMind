"""Generated markdown: model_comparison.md and reproducibility_checks.md (numbers come only from the CSVs)."""
import pandas as pd

from .common import OUT, md_table


def _ci(df, model, metric, scheme="day"):
    r = df[(df["model"] == model) & (df["metric"] == metric) & (df["scheme"] == scheme)].iloc[0]
    return f"{r['ci_low']:.3f}–{r['ci_high']:.3f}"


def write(n_blocks, versions):
    m = pd.read_csv(OUT["metrics"] / "supervised_metrics.csv")
    ci = pd.read_csv(OUT["metrics"] / "supervised_metric_ci.csv")
    pairs = pd.read_csv(OUT["metrics"] / "pairwise_comparisons.csv")
    ano = pd.read_csv(OUT["metrics"] / "anomaly_metrics.csv")
    ano_ci = pd.read_csv(OUT["metrics"] / "anomaly_metric_ci.csv")
    cost = pd.read_csv(OUT["metrics"] / "anomaly_cost.csv")
    stab = pd.read_csv(OUT["validation"] / "anomaly_seed_stability.csv")
    cal = pd.read_csv(OUT["metrics"] / "calibration_metrics.csv")
    ops = pd.read_csv(OUT["metrics"] / "operational_policies.csv")

    t = m[m["set"] == "test"]
    sup = t[["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "critical_recall"]].copy()
    sup.columns = ["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "Critical Recall"]
    models = [x for x in t["model"] if x != "Majority-class reference"]
    extra = pd.DataFrame([{
        "Model": x, "Macro F1 95% CI": _ci(ci, x, "macro_f1"), "Critical Recall 95% CI": _ci(ci, x, "critical_recall"),
        "Critical Precision": t.set_index("model").loc[x, "critical_precision"], "Critical F1": t.set_index("model").loc[x, "critical_f1"],
        "ROC-AUC (OvR macro)": t.set_index("model").loc[x, "roc_auc_ovr_macro"], "Log loss": t.set_index("model").loc[x, "log_loss"],
        "Brier (multiclass)": t.set_index("model").loc[x, "brier_multiclass"]} for x in models])
    crit = t[["model", "critical_support", "critical_false_negatives", "critical_pred_on_time", "critical_pred_at_risk",
              "on_time_pred_critical", "at_risk_pred_critical", "critical_alerts", "at_risk_pred_on_time"]].copy()
    crit.columns = ["Model", "Critical shipments", "Critical missed (FN)", "Critical→On-Time", "Critical→At-Risk",
                    "On-Time→Critical", "At-Risk→Critical", "Critical alerts raised", "At-Risk→On-Time"]

    rows = []
    for _, r in m[m["set"].isin(["cv_fold_0", "cv_fold_1", "cv_fold_2", "test"]) & (m["key"] != "dummy_prior")].iterrows():
        rows.append({"Model": r["model"], "Set": r["set"], "Critical share": r["critical_prevalence"], "Macro F1": r["macro_f1"],
                     "Critical Recall": r["critical_recall"], "Critical Precision": r["critical_precision"],
                     "Critical ROC-AUC": r["critical_roc_auc_ovr"]})
    temporal = pd.DataFrame(rows).sort_values(["Model", "Set"])

    pr = pairs[pairs["metric"].isin(["macro_f1", "critical_recall", "critical_precision", "log_loss", "roc_auc_ovr_macro",
                                     "delayed_pr_auc", "delayed_roc_auc", "critical_pr_auc", "delayed_precision", "delayed_recall",
                                     "on_time_flag_rate"])].copy()
    pr["95% CI"] = pr.apply(lambda r: f"{r['ci_low']:+.3f} to {r['ci_high']:+.3f}", axis=1)
    pr = pr[["model_a", "model_b", "metric", "scheme", "difference_a_minus_b", "95% CI", "share_of_resamples_a_better", "interval_excludes_zero"]]
    pr.columns = ["A", "B", "Metric", "Blocks", "A − B", "95% CI", "Share of resamples A better", "CI excludes 0"]

    a = ano[ano["set"] == "test"]
    anot = a[["model", "delayed_precision", "delayed_recall", "delayed_f1", "delayed_roc_auc", "delayed_pr_auc"]].copy()
    anot.columns = ["Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    anoc = a[["model", "critical_precision", "critical_recall", "critical_f1", "critical_roc_auc", "critical_pr_auc", "critical_prevalence"]].copy()
    anoc.columns = ["Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "PR-AUC baseline (prevalence)"]
    anox = a[["model", "flag_rate", "delayed_fpr", "critical_fpr", "flag_rate_On-Time", "flag_rate_At-Risk", "flag_rate_Critical",
              "share_of_flags_On-Time", "share_of_flags_At-Risk", "share_of_flags_Critical", "critical_lift"]].copy()
    anox.columns = ["Model", "Flag rate", "FPR (delayed)", "FPR (Critical)", "Flagged On-Time", "Flagged At-Risk", "Flagged Critical",
                    "Flags that are On-Time", "Flags that are At-Risk", "Flags that are Critical", "Critical lift"]
    aci = pd.DataFrame([{"Model": x, **{k: _ci(ano_ci, x, k) for k in ("delayed_pr_auc", "delayed_roc_auc", "critical_pr_auc", "delayed_recall")}}
                        for x in a["model"]])
    st = stab.groupby("model").agg(seeds=("seed", "count"), pr_auc_min=("delayed_pr_auc", "min"), pr_auc_max=("delayed_pr_auc", "max"),
                                   spearman_min=("spearman_vs_saved", "min"), flag_jaccard_min=("flag_jaccard_vs_saved", "min"),
                                   critical_recall_min=("critical_recall", "min"), critical_recall_max=("critical_recall", "max")).reset_index()

    calt = cal[cal["set"] == "test"][["model", "probabilities", "log_loss", "brier_multiclass", "ece_top_label", "ece_classwise_mean",
                                      "mean_p_Critical", "observed_Critical", "accuracy_conf_75_85", "n_conf_75_85", "critical_recall"]]
    opst = ops[["model", "policy", "threshold_t", "val_critical_recall", "macro_f1", "critical_recall", "critical_precision",
                "critical_alerts_per_1000", "missed_critical_per_1000", "critical_as_on_time_per_1000", "false_critical_alerts_per_1000",
                "expected_cost_per_shipment"]]

    md = ["# Model comparison - independent evaluation (generated by `report_tables.py`)", "",
          "Every number below is computed by `ml/evaluation/scripts/run_evaluation.py` from the saved models; nothing is typed in by hand. "
          "Supervised classifiers and anomaly detectors are compared **separately** and never ranked together.", "",
          f"Uncertainty: cluster bootstrap with 1,000 resamples. 'day' = whole departure days ({n_blocks['day']} clusters in the test period); "
          f"'port_day' = origin port x day ({n_blocks['port_day']} clusters). Paired: both models scored on the same resample.", "",
          "## 1. Supervised classifiers - test period (latest 25 %, never used for tuning)", "", md_table(sup), "",
          md_table(extra), "", "### Critical-class errors (counts, test period)", "", md_table(crit, 0), "",
          "### Across time: CV validation blocks (January) vs test (February)", "", md_table(temporal), "",
          "## 2. Paired differences (test period)", "", md_table(pr), "",
          "## 3. Probability quality (test period)", "",
          "`raw` = the models' probabilities. The two corrections are fitted without test data (prior correction uses training class "
          "shares only; vector scaling is fitted on pooled out-of-fold validation predictions).", "", md_table(calt), "",
          "## 4. Decision policies chosen on validation, applied once to the test period", "",
          "Policy: Critical if P(Critical) ≥ t, otherwise the more probable of On-Time / At-Risk. Expected cost uses an *assumed* "
          "cost matrix (see `operational.py`).", "", md_table(opst), "",
          "## 5. Anomaly detectors - test period", "", "Positive = delayed (At-Risk or Critical):", "", md_table(anot), "",
          "Positive = Critical only:", "", md_table(anoc), "", "Flag behaviour:", "", md_table(anox), "",
          "95 % day-block intervals:", "", md_table(aci), "",
          "Computational cost (this machine, evaluation-only refits):", "", md_table(cost), "",
          "Seed / subsample stability (6 refits each; Isolation Forest varies the random seed, One-Class SVM the 3,000-row subsample):", "",
          md_table(st), ""]
    (OUT["reports"] / "model_comparison.md").write_text("\n".join(md))

    chk = pd.read_csv(OUT["validation"] / "reproducibility_checks.csv")
    md = ["# Reproducibility and technical verification (generated)", "",
          f"{(chk['result'] == 'PASS').sum()} of {len(chk)} checks passed.", "",
          "Evaluation-only refits (cloned saved pipelines, identical hyperparameters) were used for: CV-fold out-of-fold predictions, "
          "the final-model refit check, the ablation re-runs and the anomaly seed-stability/cost runs. No saved model or stored "
          "result was modified.", "", "Environment: " + ", ".join(f"{k} {v}" for k, v in versions.items()), "",
          md_table(chk), ""]
    (OUT["validation"] / "reproducibility_checks.md").write_text("\n".join(md))
