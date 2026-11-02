"""Data preparation, hyperparameter tuning, fitting and saving of all six models.

Supervised models (LR, RF, XGBoost, LightGBM): RandomizedSearchCV over expanding-window,
purged time-series folds inside the training period, refit on the whole training period with the
best macro-F1 configuration. The test period is never touched here.

Anomaly models (Isolation Forest, One-Class SVM): fitted on On-Time training shipments only.
Hyperparameters are chosen on the validation block (the last CV fold) by PR-AUC for
"delayed" (At-Risk or Critical); labels are used for that selection only, never for fitting.
"""
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, f1_score, make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from . import config
from . import (
    isolation_forest_model,
    lightgbm_model,
    logistic_regression_model,
    one_class_svm_model,
    random_forest_model,
    xgboost_model,
)
from .features import ANOMALY_FEATURES, FEATURE_GROUPS, FEATURES, IDENTITY_FEATURES, build_modelling_table, feature_catalog
from .preprocessing import ANOMALY_INPUT_COLUMNS, chronological_split, make_preprocessor, time_series_folds

SUPERVISED = [logistic_regression_model, random_forest_model, xgboost_model, lightgbm_model]
ANOMALY = [isolation_forest_model, one_class_svm_model]

SCORING = {
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "accuracy": "accuracy",
    "recall_critical": make_scorer(recall_score, labels=[2], average="macro", zero_division=0),
    "recall_at_risk": make_scorer(recall_score, labels=[1], average="macro", zero_division=0),
    "neg_log_loss": "neg_log_loss",
}


def _json(obj, path):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        return repr(o)

    path.write_text(json.dumps(obj, indent=2, default=default))


def _describe_space(v):
    """Readable form of a search-space entry: a list, or a frozen scipy distribution with its args."""
    if hasattr(v, "dist"):
        args = ", ".join(f"{a:g}" for a in v.args)
        return f"{v.dist.name}({args})" + (f" {v.kwds}" if v.kwds else "")
    return repr(v)


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
def prepare_data(lag_hours=config.LAG_HOURS, save=True):
    table, log = build_modelling_table(lag_hours)
    split, info = chronological_split(table)
    table["split"] = split.values
    train = table[table["split"] == "train"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    folds = time_series_folds(train)

    train["cv_val_fold"] = -1
    for k, (_, val_idx) in enumerate(folds):
        train.loc[val_idx, "cv_val_fold"] = k
    val_fold = len(folds) - 1  # the most recent block of the training period = validation set

    ctx = {"table": table, "train": train, "test": test, "folds": folds, "split_info": info, "val_fold": val_fold, "data_log": log}
    if not save:
        return ctx

    cols = ["shipment_id", "departure_ts", "actual_arrival_ts", "delay_status", "shipping_mode"]
    splits = pd.concat([
        train[cols + ["cv_val_fold"]].assign(split="train"),
        table.loc[table["split"] == "purge", cols].assign(split="purge", cv_val_fold=-1),
        test[cols].assign(split="test", cv_val_fold=-1),
    ])
    splits.to_csv(config.RESULT_DIRS["data"] / "splits.csv", index=False)
    table.to_pickle(config.RESULT_DIRS["data"] / "modelling_table.pkl")
    feature_catalog(table).to_csv(config.RESULT_DIRS["data"] / "feature_catalog.csv")

    def describe(df):
        return {
            "n": int(len(df)),
            "departure_from": df["departure_ts"].min(),
            "departure_to": df["departure_ts"].max(),
            "class_counts": df["delay_status"].value_counts().reindex(config.CLASSES).to_dict(),
            "class_share_pct": (df["delay_status"].value_counts(normalize=True).reindex(config.CLASSES) * 100).round(1).to_dict(),
            "n_air": int((df["shipping_mode"] == "air").sum()),
        }

    summary = {
        **log,
        "prediction_cutoff": f"departure_ts - {lag_hours} h",
        "test_start": info["test_start"],
        "train_end_exclusive": info["train_end"],
        "purge_gap_hours": config.PURGE_GAP_HOURS,
        "train": describe(train),
        "purge (unused)": describe(table[table["split"] == "purge"]),
        "test": describe(test),
        "train_labels_not_yet_arrived_at_test_start": int((train["actual_arrival_ts"] >= info["test_start"]).sum()),
        "cv_folds": [
            {"fold": k, "n_train": len(a), "n_val": len(b),
             "train_to": train["departure_ts"].iloc[a].max(), "val_from": train["departure_ts"].iloc[b].min(),
             "val_to": train["departure_ts"].iloc[b].max(),
             "val_class_share_pct": (train["delay_status"].iloc[b].value_counts(normalize=True).reindex(config.CLASSES) * 100).round(1).to_dict()}
            for k, (a, b) in enumerate(folds)
        ],
        "validation_set": f"cv fold {val_fold} validation block",
        "n_features": len(FEATURES),
        "features": FEATURE_GROUPS,
    }
    _json(summary, config.RESULT_DIRS["data"] / "data_summary.json")
    return ctx


# ---------------------------------------------------------------------------------------------
# Supervised
# ---------------------------------------------------------------------------------------------
def tune_supervised(module, ctx):
    train, folds = ctx["train"], ctx["folds"]
    search = RandomizedSearchCV(
        module.build_pipeline(),
        module.param_distributions(),
        n_iter=config.N_ITER[module.NAME],
        scoring=SCORING,
        refit=config.TUNING_SCORING,
        cv=folds,
        n_jobs=config.N_JOBS,
        random_state=config.SEED,
        error_score="raise",
    )
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(train[FEATURES], train["y"])
    elapsed = time.time() - t0

    cv = pd.DataFrame(search.cv_results_)
    cv.to_csv(config.RESULT_DIRS["metrics"] / f"{module.NAME}_cv_results.csv", index=False)
    best = search.best_index_
    per_metric = {
        m: {"cv_mean": cv.loc[best, f"mean_test_{m}"], "cv_std": cv.loc[best, f"std_test_{m}"],
            "per_fold": [cv.loc[best, f"split{k}_test_{m}"] for k in range(len(folds))]}
        for m in SCORING
    }
    joblib.dump(search.best_estimator_, config.MODELS / f"{module.NAME}.joblib")
    joblib.dump(search.best_estimator_.named_steps["prep"], config.MODELS / f"{module.NAME}_preprocessor.joblib")
    cfg = {
        "model": module.LABEL,
        "estimator": type(search.best_estimator_.named_steps["model"]).__name__,
        "preprocessing": module.PREPROCESSING,
        "features": FEATURES,
        "best_params": search.best_params_,
        "search_space": {k: _describe_space(v) for k, v in module.param_distributions().items()},
        "n_iter": config.N_ITER[module.NAME],
        "tuning_metric": config.TUNING_SCORING,
        "cv_scores_best": per_metric,
        "n_train": len(train),
        "seed": config.SEED,
        "tuning_seconds": round(elapsed, 1),
    }
    _json(cfg, config.MODELS / f"{module.NAME}_config.json")
    print(f"  {module.LABEL}: CV macro-F1 {per_metric['f1_macro']['cv_mean']:.3f} "
          f"(folds {np.round(per_metric['f1_macro']['per_fold'], 3)}), {elapsed:.0f}s", flush=True)
    return search.best_estimator_, cfg


def fit_dummy(ctx):
    """Reference only: predicts the training class prior (argmax = always On-Time)."""
    model = DummyClassifier(strategy="prior").fit(ctx["train"][FEATURES], ctx["train"]["y"])
    joblib.dump(model, config.MODELS / "dummy_prior.joblib")
    return model


# ---------------------------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------------------------
def fit_anomaly(module, ctx):
    train, folds, k = ctx["train"], ctx["folds"], ctx["val_fold"]
    tr_idx, val_idx = folds[k]
    fold_train, val = train.iloc[tr_idx], train.iloc[val_idx]
    normal = fold_train[fold_train["y"] == 0][ANOMALY_INPUT_COLUMNS]
    val_pos = (val["y"] > 0).values  # anomaly-positive = delayed (At-Risk or Critical)

    rows = []
    t0 = time.time()
    for params in module.param_grid():
        pipe = module.build_pipeline(**params).fit(module.fit_rows(normal))
        s = module.anomaly_score(pipe, val[ANOMALY_INPUT_COLUMNS])
        rows.append({**params, "val_pr_auc_delayed": average_precision_score(val_pos, s),
                     "val_roc_auc_delayed": roc_auc_score(val_pos, s),
                     "val_pr_auc_critical": average_precision_score(val["y"].values == 2, s)})
    grid = pd.DataFrame(rows).sort_values("val_pr_auc_delayed", ascending=False)
    grid.to_csv(config.RESULT_DIRS["anomaly"] / f"{module.NAME}_tuning_grid.csv", index=False)
    best_params = {k: v for k, v in grid.iloc[0].items() if not k.startswith("val_") and not (isinstance(v, float) and np.isnan(v))}
    for key in ("n_estimators", "max_samples", "degree"):
        if key in best_params:
            best_params[key] = int(best_params[key])

    train_normal = train[train["y"] == 0][ANOMALY_INPUT_COLUMNS]
    fit_X = module.fit_rows(train_normal)
    final = module.build_pipeline(**best_params).fit(fit_X)
    threshold = float(np.quantile(module.anomaly_score(final, train_normal), config.ANOMALY_FLAG_QUANTILE))
    elapsed = time.time() - t0

    joblib.dump({"pipeline": final, "threshold": threshold}, config.MODELS / f"{module.NAME}.joblib")
    cfg = {
        "model": module.LABEL,
        "features": ANOMALY_FEATURES,
        "port_relative_features": list(final.named_steps["prep"].named_steps["port_relative"].cols),
        "fit_on": "On-Time shipments of the training period",
        "n_fit_rows": len(fit_X),
        "n_train_normal": len(train_normal),
        "best_params": best_params,
        "selection": "max PR-AUC (positive = At-Risk or Critical) on the validation block, model fitted on the fold's On-Time rows",
        "best_validation": grid.iloc[0][[c for c in grid.columns if c.startswith("val_")]].to_dict(),
        "threshold": threshold,
        "threshold_rule": f"{config.ANOMALY_FLAG_QUANTILE:.0%} quantile of anomaly scores of training On-Time shipments",
        "grid_size": len(grid),
        "seed": config.SEED,
        "seconds": round(elapsed, 1),
    }
    _json(cfg, config.MODELS / f"{module.NAME}_config.json")
    print(f"  {module.LABEL}: best {best_params}, val PR-AUC {cfg['best_validation']['val_pr_auc_delayed']:.3f}, {elapsed:.0f}s", flush=True)
    return final, cfg


# ---------------------------------------------------------------------------------------------
# Ablations and sensitivity checks (fixed tuned hyperparameters, refit only)
# ---------------------------------------------------------------------------------------------
def _refit(model, module, train, numeric, categorical):
    pipe = clone(model)
    pipe.set_params(prep=make_preprocessor(module.PREPROCESSING, numeric=numeric, categorical=categorical))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return pipe.fit(train[numeric + categorical], train["y"])


def _score(pipe, df, cols):
    pred = pipe.predict(df[cols])
    return {"macro_f1": f1_score(df["y"], pred, average="macro"),
            "critical_recall": recall_score(df["y"], pred, labels=[2], average="macro", zero_division=0),
            "at_risk_recall": recall_score(df["y"], pred, labels=[1], average="macro", zero_division=0),
            "accuracy": float((pred == df["y"].values).mean()), "n_train": None, "n_test": len(df)}


def ablations_and_sensitivity(models, ctx):
    """Each row: one supervised model re-fitted with its tuned hyperparameters under a variant."""
    from .features import CATEGORICAL, NUMERIC

    train, test, info = ctx["train"], ctx["test"], ctx["split_info"]
    state_groups = ["origin_congestion", "container_dwell", "anchor_queue", "weather", "disruptions", "destination_state", "hub_state"]
    state_cols = {c for g in state_groups for c in FEATURE_GROUPS[g]}
    filled_inputs = ["filled_route_id", "filled_carrier", "filled_shipping_mode", "filled_cargo_type",
                     "filled_container_teu", "filled_shipment_priority", "filled_expected_arrival_ts"]

    variants = {
        "full model (reference)": (train, test, NUMERIC, CATEGORICAL),
        "ablation: no port/carrier identity": (train, test, NUMERIC, [c for c in CATEGORICAL if c not in IDENTITY_FEATURES]),
        "ablation: no operational state (booking-time info only)": (train, test, [c for c in NUMERIC if c not in state_cols], CATEGORICAL),
        "ablation: no historical delay rates": (train, test, [c for c in NUMERIC if c not in FEATURE_GROUPS["historical_delay_rates"]], CATEGORICAL),
        "sensitivity: train only on labels available at test start": (
            train[train["actual_arrival_ts"] < info["test_start"]], test, NUMERIC, CATEGORICAL),
        "sensitivity: exclude shipments with filled inputs (train+test)": (
            train[~train[filled_inputs].any(axis=1)], test[~test[filled_inputs].any(axis=1)], NUMERIC, CATEGORICAL),
        "sensitivity: sea shipments only (train+test)": (
            train[train["shipping_mode"] == "sea"], test[test["shipping_mode"] == "sea"], NUMERIC, CATEGORICAL),
    }
    # Earlier prediction time: same shipments and split, features rebuilt with a 24 h cutoff
    ctx24 = prepare_data(lag_hours=24, save=False)
    ids_tr, ids_te = set(train["shipment_id"]), set(test["shipment_id"])
    t24 = ctx24["table"]
    variants["sensitivity: prediction 24 h before departure"] = (
        t24[t24["shipment_id"].isin(ids_tr)], t24[t24["shipment_id"].isin(ids_te)], NUMERIC, CATEGORICAL)

    rows = []
    for module in SUPERVISED:
        for variant, (tr, te, num, cat) in variants.items():
            pipe = _refit(models[module.NAME], module, tr, num, cat)
            r = _score(pipe, te, num + cat)
            r.update({"model": module.LABEL, "variant": variant, "n_train": len(tr), "n_features": len(num) + len(cat)})
            rows.append(r)
        print(f"  ablations done: {module.LABEL}", flush=True)
    out = pd.DataFrame(rows)[["model", "variant", "n_train", "n_test", "n_features", "macro_f1", "critical_recall", "at_risk_recall", "accuracy"]]
    out.to_csv(config.RESULT_DIRS["comparison"] / "ablation_and_sensitivity.csv", index=False)
    return out


def main():
    config.ensure_dirs()
    print("preparing data ...", flush=True)
    ctx = prepare_data()
    print(f"  train {len(ctx['train'])}, test {len(ctx['test'])}, test starts {ctx['split_info']['test_start']}", flush=True)
    print("tuning supervised models ...", flush=True)
    models, configs = {}, {}
    for module in SUPERVISED:
        models[module.NAME], configs[module.NAME] = tune_supervised(module, ctx)
    models["dummy_prior"] = fit_dummy(ctx)
    print("fitting anomaly detectors ...", flush=True)
    for module in ANOMALY:
        models[module.NAME], configs[module.NAME] = fit_anomaly(module, ctx)
    print("ablations and sensitivity checks ...", flush=True)
    ablations_and_sensitivity(models, ctx)
    return ctx, models, configs


if __name__ == "__main__":
    main()
