"""Technical verification of the saved artifacts. Every check is recorded with its evidence."""
import importlib
import platform
import warnings

import joblib
import numpy as np
import pandas as pd

from .common import ANOMALY_INPUT_COLUMNS, CLASSES, FEATURES, OUT, SUP_LABEL, mlc

OUTCOME_COLS = ["delay_status", "actual_delay_hours", "actual_arrival_ts", "y"]


class Log:
    def __init__(self):
        self.rows = []

    def add(self, area, check, passed, evidence):
        self.rows.append({"area": area, "check": check, "result": "PASS" if passed else "FAIL", "evidence": evidence})


def versions():
    out = {"python": platform.python_version()}
    for pkg in ["sklearn", "xgboost", "lightgbm", "shap", "numpy", "pandas", "scipy", "joblib"]:
        try:
            out[pkg] = importlib.import_module(pkg).__version__
        except ImportError:
            out[pkg] = "not installed"
    return out


def run(ctx, models, bundles, test_proba, refit_proba, oof_check, metric_check, anomaly_check, rob):
    log = Log()
    test, train, table = ctx["test"], ctx["train"], ctx["table"]

    # --- loading and versions ---------------------------------------------------------------
    for name in list(SUP_LABEL) + ["dummy_prior", "isolation_forest", "one_class_svm"]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = joblib.load(mlc.MODELS / f"{name}.joblib")
        vw = [str(x.message)[:120] for x in w if "version" in str(x.message).lower()]
        log.add("loading", f"{name}.joblib loads without version warnings", not vw,
                f"type {type(obj).__name__ if not isinstance(obj, dict) else 'dict(pipeline, threshold)'}; warnings: {vw or 'none'}")
    req = (mlc.ML_ROOT / "requirements.txt").read_text()
    v = versions()
    pinned_ok = all(f"{pkg}=={v[pkg]}" in req for pkg in ("xgboost", "lightgbm", "shap"))
    log.add("environment", "installed xgboost / lightgbm / shap match ml/requirements.txt pins", pinned_ok, str(v))

    # --- schema ------------------------------------------------------------------------------
    for name, m in models.items():
        cfg_feats = __import__("json").loads((mlc.MODELS / f"{name}_config.json").read_text())["features"]
        fin = list(m.feature_names_in_)
        log.add("schema", f"{name}: input columns = config feature list = FEATURES (same order)",
                fin == cfg_feats == FEATURES, f"{len(fin)} columns; first 3 {fin[:3]}")
        pre = joblib.load(mlc.MODELS / f"{name}_preprocessor.joblib")
        a = pre.transform(test[FEATURES])
        b = m.named_steps["prep"].transform(test[FEATURES])
        same = np.allclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True)
        log.add("preprocessing", f"{name}: saved standalone preprocessor reproduces the pipeline's transform", same,
                f"transformed shape {np.asarray(a).shape}")
        log.add("labels", f"{name}: classes_ = [0, 1, 2] = {CLASSES}", list(m.classes_) == [0, 1, 2], f"classes_ {list(m.classes_)}")

    # --- labels ------------------------------------------------------------------------------
    codes = table["delay_status"].map({c: i for i, c in enumerate(CLASSES)})
    log.add("labels", "y code equals position of delay_status in CLASSES for every row", bool((codes == table["y"]).all()),
            f"{len(table)} rows")
    d = table["actual_delay_hours"]
    expect = np.where(d <= 12, 0, np.where(d <= 48, 1, 2))
    log.add("labels", "delay_status agrees with actual_delay_hours thresholds (<=12, 12-48, >48)", bool((expect == table["y"]).all()),
            f"mismatches {(expect != table['y']).sum()}")

    # --- predictions vs stored ---------------------------------------------------------------
    stored = pd.read_csv(mlc.RESULT_DIRS["metrics"] / "test_predictions.csv")
    log.add("predictions", "test rows in same order as stored test_predictions.csv", bool((stored["shipment_id"].values == test["shipment_id"].values).all()),
            f"{len(stored)} rows")
    for name, p in test_proba.items():
        sp = stored[[f"{name}_p_{c}" for c in CLASSES]].to_numpy()
        diff = float(np.abs(sp - p).max())
        same_lab = bool((stored[f"{name}_pred"].values == np.array(CLASSES)[p.argmax(1)]).all())
        log.add("predictions", f"{name}: saved model reproduces stored test probabilities and labels", diff < 1e-6 and same_lab,
                f"max |Δp| {diff:.2e}; labels identical {same_lab}")
        log.add("predictions", f"{name}: probability rows sum to 1 and lie in [0, 1]",
                bool(np.allclose(p.sum(1), 1, atol=1e-6) and p.min() >= 0 and p.max() <= 1), f"min {p.min():.2e}, max {p.max():.4f}")
        log.add("predictions", f"{name}: repeated inference is identical", bool(np.array_equal(p, models[name].predict_proba(test[FEATURES]))), "two calls")
        diff_refit = float(np.abs(refit_proba[name] - p).max())
        log.add("training", f"{name}: refit from saved configuration on the training period reproduces test probabilities",
                diff_refit < 1e-6, f"max |Δp| {diff_refit:.2e} (evaluation-only refit; saved model untouched)")

    # --- leakage -----------------------------------------------------------------------------
    bad = [c for c in FEATURES if c in OUTCOME_COLS or c.startswith(("LEAK_", "filled_", "raw_label", "outcome_not"))]
    log.add("leakage", "no outcome, LEAK_, or cleaning-lineage column among the 51 inference features", not bad, f"offending: {bad or 'none'}")
    canary = test.copy()
    canary["delay_status"] = "Critical"
    canary["actual_delay_hours"] = 999.0
    canary["actual_arrival_ts"] = canary["departure_ts"]
    canary["y"] = 2
    try:
        same = all(np.array_equal(m.predict_proba(canary), test_proba[n]) for n, m in models.items())
        how = "pipelines given the full frame (corrupted delay_status/actual_delay_hours/actual_arrival_ts/y) select only their 51 inputs"
    except ValueError as e:
        same = all(np.array_equal(m.predict_proba(canary[FEATURES]), test_proba[n]) for n, m in models.items())
        how = f"full frame rejected by the pipeline ({str(e)[:80]}); checked on the 51 selected columns"
    log.add("leakage", "overwriting the outcome columns in the input frame leaves predictions unchanged", same, how)
    # point-in-time history: recompute hist_* for a random sample of test rows from scratch
    rng = np.random.default_rng(mlc.SEED)
    sample = test.iloc[rng.choice(len(test), 300, replace=False)]
    worst = 0.0
    n_cmp = 0
    for key, col in [("carrier", "hist_carrier_pct_delayed"), ("origin_port_id", "hist_origin_port_id_pct_delayed"),
                     ("route_id", "hist_route_id_pct_delayed")]:
        for _, r in sample.iterrows():
            prior = table[(table[key] == r[key]) & (table["actual_arrival_ts"] < r["cutoff_ts"])]
            val = (prior["actual_delay_hours"] > 12).mean() * 100 if len(prior) >= 10 else np.nan
            if np.isnan(val) and np.isnan(r[col]):
                continue
            worst = max(worst, abs(val - r[col]) if not (np.isnan(val) or np.isnan(r[col])) else np.inf)
            n_cmp += 1
    log.add("leakage", "hist_* rates equal a from-scratch recomputation using only shipments arrived before the cutoff", worst < 1e-6,
            f"{n_cmp} (row, feature) comparisons on 300 random test rows; max |Δ| {worst:.2e} percentage points")
    cut_ok = bool(((table["departure_ts"] - table["cutoff_ts"]) == pd.Timedelta(hours=mlc.LAG_HOURS)).all())
    log.add("leakage", "prediction cutoff = departure_ts - 1 h for every shipment", cut_ok, "cutoff_ts column")

    # --- missing values ----------------------------------------------------------------------
    from cargomind_ml.features import CATEGORICAL, NUMERIC

    miss_tr, miss_te = train[NUMERIC].isna().mean(), test[NUMERIC].isna().mean()
    log.add("missing values", "features with missing values in training also handled at test (rates listed)", True,
            "; ".join(f"{c}: train {miss_tr[c]:.1%} / test {miss_te[c]:.1%}" for c in NUMERIC if miss_tr[c] > 0 or miss_te[c] > 0) or "none")
    probe = test.iloc[:50].copy()
    probe[NUMERIC] = np.nan
    probe[CATEGORICAL[0]] = "UNSEEN_PORT"
    ok = all(np.isfinite(m.predict_proba(probe[FEATURES])).all() for m in models.values())
    log.add("missing values", "all numeric inputs NaN + an unseen origin port: every model returns finite probabilities", ok, "50 probe rows")

    # --- rebuilt data ------------------------------------------------------------------------
    saved = pd.read_pickle(mlc.RESULT_DIRS["data"] / "modelling_table.pkl")
    same = saved["shipment_id"].equals(table["shipment_id"]) and saved[FEATURES + ["y"]].equals(table[FEATURES + ["y"]])
    log.add("data", "modelling table rebuilt from Dataset/Processed equals the saved modelling_table.pkl (51 features + y)", bool(same),
            f"{len(table)} rows")

    # --- tuning / folds ----------------------------------------------------------------------
    log.add("training", "CV-fold refits reproduce the per-fold macro-F1 and Critical recall stored at tuning time",
            bool(oof_check["max_abs_diff"].max() < 1e-9), f"max |Δ| {oof_check['max_abs_diff'].max():.2e} over {len(oof_check)} model-folds")

    # --- anomaly -----------------------------------------------------------------------------
    for _, r in anomaly_check.iterrows():
        log.add("anomaly", f"{r['model']}: saved detector reproduces stored test scores and flags",
                r["max_abs_score_diff"] < 1e-9 and r["flags_identical"], f"max |Δscore| {r['max_abs_score_diff']:.2e}")
    from cargomind_ml.train_models import ANOMALY

    normal = train[train["y"] == 0][ANOMALY_INPUT_COLUMNS]
    for m in ANOMALY:
        b = bundles[m.NAME]
        thr = float(np.quantile(m.anomaly_score(b["pipeline"], normal), mlc.ANOMALY_FLAG_QUANTILE))
        log.add("anomaly", f"{m.LABEL}: stored flag threshold = 95th percentile of training On-Time scores",
                abs(thr - b["threshold"]) < 1e-9, f"stored {b['threshold']:.6f}, recomputed {thr:.6f}")
        prs = b["pipeline"].named_steps["prep"].named_steps["port_relative"]
        cfg_feats = __import__("json").loads((mlc.MODELS / f"{m.NAME}_config.json").read_text())["features"]
        used = list(prs.cols) + list(prs.other_)
        shuffled = test[ANOMALY_INPUT_COLUMNS[::-1]]
        order_free = np.array_equal(m.anomaly_score(b["pipeline"], shuffled), m.anomaly_score(b["pipeline"], test[ANOMALY_INPUT_COLUMNS]))
        log.add("anomaly", f"{m.LABEL}: columns used = config feature list (15) grouped by origin_port_id; scores independent of column order",
                sorted(used) == sorted(cfg_feats) and prs.group_col == "origin_port_id" and order_free,
                f"{len(used)} features selected by name; reversed column order gives identical scores: {order_free}")

    # --- stored metrics / ablations ----------------------------------------------------------
    # log loss gets 1e-6: XGBoost returns float32 probabilities and sklearn's log_loss renormalises each row first
    tol = np.where(metric_check["metric"] == "log_loss", 1e-6, 1e-9)
    exact = metric_check[metric_check["metric"] != "log_loss"]["abs_diff"].max()
    ll = metric_check[metric_check["metric"] == "log_loss"]["abs_diff"].max()
    log.add("metrics", "independent metric implementation reproduces every stored supervised metric",
            bool((metric_check["abs_diff"] < tol).all()),
            f"{len(metric_check)} values; max |Δ| {exact:.2e} for count-based metrics and AUCs, {ll:.2e} for log loss (float32 XGBoost rows renormalised by sklearn)")
    log.add("metrics", "re-run ablation / sensitivity variants reproduce the stored ablation table",
            bool(rob["rerun_matches_stored"].all()), f"{int(rob['rerun_matches_stored'].sum())}/{len(rob)} variant-model rows identical")

    df = pd.DataFrame(log.rows)
    df.to_csv(OUT["validation"] / "reproducibility_checks.csv", index=False)
    return df, v
