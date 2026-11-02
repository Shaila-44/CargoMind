"""Independent evaluation of the saved CargoMind models.

Run from the CargoMind project root:

    .venv/bin/python ML/Evaluation/scripts/run_evaluation.py

Requires the project packages to be installed once (`pip install -e .` at the repository root).
Reads ML/ML/models/ and ML/ML/results/ (never writes to them) and rebuilds the modelling table from
ML/Dataset/Processed in memory (read-only). Writes every output under ML/Evaluation/.
"""
import json
import time

from cargomind_eval import anomaly, calibration, common, operational, refits, reproducibility, report_tables, robustness
from cargomind_eval import supervised, temporal, uncertainty


def step(msg, t0):
    print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)


def main():
    t0 = time.time()
    common.ensure_dirs()
    common.setup_plots()
    step("rebuilding modelling table and split (read-only)", t0)
    ctx = common.load_context()
    test, train = ctx["test"], ctx["train"]
    models, bundles = common.load_supervised(), common.load_anomaly()
    test_proba = common.predict_proba(models, test)

    step("out-of-fold validation predictions (evaluation-only refits)", t0)
    oof = refits.oof_validation(ctx, models)
    oof_check = refits.oof_check(oof, models)

    step("supervised metrics", t0)
    sets = supervised.evaluation_sets(ctx, oof, test_proba)
    table = supervised.metric_table(sets, supervised.dummy_test_proba(test))
    metric_check = supervised.verify_against_stored(table)
    supervised.per_class_table(table)
    supervised.confusion_outputs(sets)
    supervised.boundary_analysis(test, test_proba)
    supervised.curves(sets)

    step("calibration", t0)
    prior = train["y"].value_counts(normalize=True).sort_index().values
    calibration.run(sets, oof, prior)

    step("anomaly detectors", t0)
    val = train.iloc[ctx["folds"][ctx["val_fold"]][1]]
    sc_test, sc_val = common.anomaly_scores(bundles, test), common.anomaly_scores(bundles, val)
    ano_check = anomaly.verify_scores(test, sc_test)
    anomaly.run(ctx, test, sc_test, sc_val, test_proba["logistic_regression"])

    step("temporal validation", t0)
    temporal.verify_split(ctx)
    perf = temporal.performance_by_period(ctx, oof, test_proba)
    prev = temporal.prevalence_over_time(ctx)
    temporal.plot_temporal(perf, prev, ctx)
    temporal.feature_drift(ctx)
    shares = {"train_prevalence": (train["y"] == 2).mean(), "validation_prevalence": (val["y"] == 2).mean(),
              "test_prevalence": (test["y"] == 2).mean(), "prevalence_5pct": 0.05, "prevalence_20pct": 0.20}
    temporal.prevalence_adjusted_precision(table, shares)

    step("bootstrap uncertainty", t0)
    thresholds = {n: b["threshold"] for n, b in bundles.items()}
    _, _, _, n_blocks = uncertainty.run(test, test_proba, sc_test, thresholds)

    step("ablation / sensitivity re-runs and slices", t0)
    abl = refits.ablation_predictions(ctx, models)
    rob = robustness.ablation_analysis(abl, test)
    robustness.plot_ablation(rob)
    robustness.slices(test, test_proba)

    step("operational policies (thresholds chosen on validation)", t0)
    operational.run(oof, test, test_proba)

    step("reproducibility checks", t0)
    refit_proba = refits.final_refit(ctx, models)
    checks, versions = reproducibility.run(ctx, models, bundles, test_proba, refit_proba, oof_check, metric_check, ano_check, rob)

    step("report tables", t0)
    report_tables.write(n_blocks, versions)
    meta = {"n_test_day_blocks": n_blocks["day"], "n_test_port_day_blocks": n_blocks["port_day"], "versions": versions,
            "checks_failed": checks[checks["result"] == "FAIL"]["check"].tolist(), "seconds": round(time.time() - t0)}
    (common.OUT["validation"] / "run_info.json").write_text(json.dumps(meta, indent=2))
    step(f"done; {len(checks)} checks, {len(meta['checks_failed'])} failed", t0)


if __name__ == "__main__":
    main()
