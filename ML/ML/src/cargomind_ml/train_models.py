"""Data preparation, hyperparameter tuning, fitting and saving of all six models.

Supervised models (LR, RF, XGBoost, LightGBM): RandomizedSearchCV over expanding-window,
purged time-series folds inside the training period, refit on the whole training period with the
best macro-F1 configuration. The test period is never touched here.

Anomaly models (Isolation Forest, One-Class SVM): fitted on On-Time training shipments only.
Hyperparameters are chosen on the validation block (the last CV fold) by PR-AUC for
"delayed" (At-Risk or Critical); labels are used for that selection only, never for fitting.
"""
# --- Standard library imports ---
import json        # For saving results as human-readable JSON files
import time        # For measuring how long each training step takes
import warnings    # For suppressing non-critical warning messages during training

# --- Third-party library imports ---
import joblib      # For saving trained models to disk so we can reload them later
import numpy as np # NumPy: the go-to library for numerical arrays and math operations
import pandas as pd # Pandas: the go-to library for working with tabular (spreadsheet-like) data

# scikit-learn imports (the most popular Python machine-learning library)
from sklearn.base import clone               # Makes a fresh copy of a model with the same settings
from sklearn.dummy import DummyClassifier    # A deliberately "dumb" model used as a baseline
from sklearn.exceptions import ConvergenceWarning  # Warning that fires when a model doesn't fully finish learning
# Scoring functions – each one measures a different aspect of how good predictions are:
#   average_precision_score : summarises the precision-recall curve into a single number (PR-AUC)
#   f1_score                : balances precision ("of everything I flagged, how many were right?")
#                             and recall ("of all the real positives, how many did I catch?")
#   make_scorer             : turns any metric function into a format scikit-learn's tuning tools expect
#   recall_score            : "of all the truly delayed shipments, what fraction did we catch?"
#   roc_auc_score           : area under the ROC curve – measures overall ranking quality
from sklearn.metrics import average_precision_score, f1_score, make_scorer, recall_score, roc_auc_score
# RandomizedSearchCV: the core hyperparameter-tuning engine.
# Instead of trying EVERY possible combination of settings (which can take forever),
# it randomly samples a fixed number of combinations and tests each one with
# cross-validation to find the best-performing settings.
from sklearn.model_selection import RandomizedSearchCV

# --- Internal project imports ---
from . import config  # Project-wide settings (paths, seeds, number of iterations, etc.)

# Each of these modules defines one type of model (its architecture, preprocessing, and search space)
from . import (
    isolation_forest_model,
    lightgbm_model,
    logistic_regression_model,
    one_class_svm_model,
    random_forest_model,
    xgboost_model,
)

# Feature definitions: which columns (variables) the models will use as inputs
# FEATURES          – the full list of input columns for supervised models
# FEATURE_GROUPS    – features grouped by category (e.g., "weather", "congestion")
# ANOMALY_FEATURES  – the smaller set of features the anomaly detectors use
# IDENTITY_FEATURES – columns that identify specific ports or carriers (can cause overfitting)
# build_modelling_table – the function that loads raw data and engineers all features
# feature_catalog   – utility that documents every feature for record-keeping
from .features import ANOMALY_FEATURES, FEATURE_GROUPS, FEATURES, IDENTITY_FEATURES, build_modelling_table, feature_catalog

# Preprocessing utilities:
# ANOMALY_INPUT_COLUMNS – raw input columns that anomaly models need
# chronological_split   – splits data into train/test by date (earlier = train, later = test)
# make_preprocessor     – builds the data-transformation pipeline (scaling, encoding, etc.)
# time_series_folds     – creates expanding-window cross-validation folds that respect time order
from .preprocessing import ANOMALY_INPUT_COLUMNS, chronological_split, make_preprocessor, time_series_folds

# -------------------------------------------------------------------------------------
# The two families of models we train:
#   SUPERVISED – models that learn from labelled examples (each shipment has a known outcome)
#   ANOMALY    – models that learn what "normal" (on-time) looks like, then flag anything unusual
# -------------------------------------------------------------------------------------
SUPERVISED = [logistic_regression_model, random_forest_model, xgboost_model, lightgbm_model]
ANOMALY = [isolation_forest_model, one_class_svm_model]

# -------------------------------------------------------------------------------------
# SCORING – the dictionary of metrics (report cards) used to evaluate every model.
#
# Think of each metric as grading the model on a different aspect:
#
# "f1_macro"         – The primary tuning metric. F1 is the harmonic mean of precision
#                      and recall. "macro" means we compute F1 for each class (On-Time,
#                      At-Risk, Critical) separately and then take the plain average,
#                      giving equal importance to every class regardless of how common it is.
#
# "f1_weighted"      – Same idea but the average is weighted by how many shipments each
#                      class has, so the majority class counts more.
#
# "accuracy"         – Simply "what percentage of predictions were correct?"
#                      This can be misleading when one class dominates.
#
# "recall_critical"  – "Of all truly Critical shipments, what fraction did the model
#                      catch?"  This is vital because missing a critical delay is costly.
#                      labels=[2] targets class 2 (Critical).
#
# "recall_at_risk"   – Same thing for At-Risk shipments (class 1).
#
# "neg_log_loss"     – Measures how confident the model is in the *correct* answer.
#                      Lower log-loss = better-calibrated probabilities. Stored as
#                      negative because scikit-learn maximises scores by convention.
# -------------------------------------------------------------------------------------
SCORING = {
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "accuracy": "accuracy",
    "recall_critical": make_scorer(recall_score, labels=[2], average="macro", zero_division=0),
    "recall_at_risk": make_scorer(recall_score, labels=[1], average="macro", zero_division=0),
    "neg_log_loss": "neg_log_loss",
}


# -------------------------------------------------------------------------------------
# Small helper: save a Python dictionary as a nicely-formatted JSON file.
# The inner `default` function handles special types (numpy ints, pandas timestamps)
# that the standard JSON encoder doesn't know how to convert.
# -------------------------------------------------------------------------------------
def _json(obj, path):
    def default(o):
        # Convert numpy integers (e.g., np.int64) to plain Python ints for JSON
        if isinstance(o, (np.integer,)):
            return int(o)
        # Convert numpy floats (e.g., np.float64) to plain Python floats for JSON
        if isinstance(o, (np.floating,)):
            return float(o)
        # Convert pandas Timestamps to ISO-format date strings (e.g., "2025-03-15T08:30:00")
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        # Anything else: fall back to its string representation
        return repr(o)

    path.write_text(json.dumps(obj, indent=2, default=default))


# -------------------------------------------------------------------------------------
# Helper to produce a human-readable description of a search-space entry.
# For example, a scipy distribution like "uniform(0.01, 0.1)" or a plain list [100, 200].
# This is used only for logging/documentation purposes.
# -------------------------------------------------------------------------------------
def _describe_space(v):
    """Readable form of a search-space entry: a list, or a frozen scipy distribution with its args."""
    # If the value is a scipy frozen distribution (like uniform, loguniform), format it nicely
    if hasattr(v, "dist"):
        args = ", ".join(f"{a:g}" for a in v.args)
        return f"{v.dist.name}({args})" + (f" {v.kwds}" if v.kwds else "")
    # Otherwise (e.g., a list of values), just show its repr
    return repr(v)


# =====================================================================================
# DATA PREPARATION
# =====================================================================================
# prepare_data() is the first step of the entire pipeline. It:
#   1. Builds the feature table from raw shipment data (build_modelling_table)
#   2. Splits shipments into TRAIN and TEST sets by date (chronological_split):
#      - Training set: all shipments departing before a cutoff date
#      - Test set: all shipments departing after the cutoff date
#      This time-based split is critical because in real life we can only predict the
#      future, never the past. A random split would let the model "peek" at future data.
#   3. Creates cross-validation folds within the training set (time_series_folds):
#      Cross-validation = testing the model multiple times on different held-out slices
#      of the training data. Each "fold" uses all data up to a date for training and
#      the next chunk of time for validation – like repeatedly pretending today is an
#      earlier date and checking if the model predicts the next period correctly.
#   4. Saves a summary of the data (class distributions, date ranges, fold details)
# =====================================================================================
def prepare_data(lag_hours=config.LAG_HOURS, save=True):
    # Build the full modelling table: load raw data, engineer features, and get a log
    # of what happened (e.g., how many rows were dropped and why).
    # lag_hours = how many hours before departure we "freeze" the feature snapshot
    #             (simulates making a prediction X hours in advance).
    table, log = build_modelling_table(lag_hours)

    # Split the table into train / purge / test based on departure timestamps.
    # "purge" is a small gap between train and test to avoid data leakage (overlap).
    split, info = chronological_split(table)
    table["split"] = split.values

    # Separate the train and test DataFrames for convenience
    train = table[table["split"] == "train"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)

    # Create expanding-window time-series cross-validation folds within the training set.
    # Each fold is a pair of (train_indices, validation_indices).
    # Fold 0 uses the earliest data for training and the next block for validation;
    # Fold 1 uses a bigger chunk (all of fold 0's data + more) for training, etc.
    folds = time_series_folds(train)

    # Record which validation fold each training row belongs to (-1 = used only for training)
    train["cv_val_fold"] = -1
    for k, (_, val_idx) in enumerate(folds):
        train.loc[val_idx, "cv_val_fold"] = k
    # The last (most recent) fold is treated as the dedicated validation set
    # for anomaly model hyperparameter selection.
    val_fold = len(folds) - 1  # the most recent block of the training period = validation set

    # Bundle everything into a context dictionary that downstream functions will use
    ctx = {"table": table, "train": train, "test": test, "folds": folds, "split_info": info, "val_fold": val_fold, "data_log": log}

    # If save=False, return early without writing files (used by sensitivity checks
    # that rebuild features with different lag_hours but don't need to save again)
    if not save:
        return ctx

    # ---------- Save diagnostic files ----------

    # Save a CSV showing every shipment's split assignment and CV fold
    cols = ["shipment_id", "departure_ts", "actual_arrival_ts", "delay_status", "shipping_mode"]
    splits = pd.concat([
        train[cols + ["cv_val_fold"]].assign(split="train"),
        table.loc[table["split"] == "purge", cols].assign(split="purge", cv_val_fold=-1),
        test[cols].assign(split="test", cv_val_fold=-1),
    ])
    splits.to_csv(config.RESULT_DIRS["data"] / "splits.csv", index=False)

    # Save the entire modelling table as a pickle file (fast binary format)
    table.to_pickle(config.RESULT_DIRS["data"] / "modelling_table.pkl")

    # Save a CSV that documents every feature (name, type, description, group)
    feature_catalog(table).to_csv(config.RESULT_DIRS["data"] / "feature_catalog.csv")

    # Inner helper: create a compact summary dict for a subset of data
    def describe(df):
        return {
            "n": int(len(df)),
            "departure_from": df["departure_ts"].min(),
            "departure_to": df["departure_ts"].max(),
            # How many shipments in each class (On-Time / At-Risk / Critical)
            "class_counts": df["delay_status"].value_counts().reindex(config.CLASSES).to_dict(),
            # Same counts but as percentages
            "class_share_pct": (df["delay_status"].value_counts(normalize=True).reindex(config.CLASSES) * 100).round(1).to_dict(),
            "n_air": int((df["shipping_mode"] == "air").sum()),
        }

    # Build a comprehensive JSON summary of the data and cross-validation setup
    summary = {
        **log,
        "prediction_cutoff": f"departure_ts - {lag_hours} h",
        "test_start": info["test_start"],
        "train_end_exclusive": info["train_end"],
        "purge_gap_hours": config.PURGE_GAP_HOURS,
        "train": describe(train),
        "purge (unused)": describe(table[table["split"] == "purge"]),
        "test": describe(test),
        # Count how many training labels hadn't actually arrived yet when the test period started
        # (important for understanding if any labels were still "in flight")
        "train_labels_not_yet_arrived_at_test_start": int((train["actual_arrival_ts"] >= info["test_start"]).sum()),
        # Detailed per-fold statistics: sizes, date ranges, class distributions
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


# =====================================================================================
# SUPERVISED MODEL TUNING
# =====================================================================================
# tune_supervised() is the heart of the training process for supervised models.
# It uses RandomizedSearchCV to find the best hyperparameters for a given model.
#
# What are "hyperparameters"?
#   Every ML model has knobs you can turn before training – like how deep a decision
#   tree can grow, or how fast the model should learn. These are called hyperparameters.
#   Unlike regular parameters (which the model learns from data), hyperparameters must
#   be set by the practitioner, and their values can dramatically affect performance.
#
# What is RandomizedSearchCV?
#   "Search" = try many different hyperparameter combinations to see which works best.
#   "Randomized" = instead of exhaustively trying all possible combos (could be millions),
#                  randomly pick N combinations. This is much faster and often finds
#                  equally good settings.
#   "CV" = cross-validation. For each combination of settings, train the model on the
#          training folds and score it on the held-out validation fold, repeating for
#          each fold. The average score across folds tells us how well those settings
#          generalise to unseen data.
#
# After finding the best settings, RandomizedSearchCV automatically re-trains ("refits")
# the model on ALL training data using those best settings (giving the model the most
# data possible for its final version).
# =====================================================================================
def tune_supervised(module, ctx):
    train, folds = ctx["train"], ctx["folds"]

    # Create the randomized search object:
    search = RandomizedSearchCV(
        # The model pipeline to tune (includes preprocessing + the classifier)
        module.build_pipeline(),
        # The range of hyperparameter values to sample from (defined per model module)
        module.param_distributions(),
        # How many random combinations to try (more = better chance of finding a great
        # combo, but takes longer). Configured per model in config.N_ITER.
        n_iter=config.N_ITER[module.NAME],
        # Which metrics to compute for every combination (our SCORING dictionary above)
        scoring=SCORING,
        # Which single metric to use for picking the "best" combination and refitting
        # (e.g., "f1_macro" – the balanced F1 score across all classes)
        refit=config.TUNING_SCORING,
        # The time-series cross-validation fold indices
        cv=folds,
        # How many CPU cores to use in parallel (-1 = all available)
        n_jobs=config.N_JOBS,
        # Random seed for reproducibility (same seed → same random combos every time)
        random_state=config.SEED,
        # If any combination causes an error, raise it immediately instead of silently
        error_score="raise",
    )

    # Start the clock so we know how long tuning takes
    t0 = time.time()
    # Suppress convergence warnings (e.g., Logistic Regression sometimes doesn't fully
    # converge with certain hyperparameters, but that's OK – it just means that combo
    # isn't great and the search will move on to others)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        # Run the actual search: try n_iter random combos, evaluate each with CV
        # train[FEATURES] = input columns; train["y"] = the target labels (0/1/2)
        search.fit(train[FEATURES], train["y"])
    elapsed = time.time() - t0

    # --- Save detailed cross-validation results ---
    # cv_results_ is a table with one row per hyperparameter combo and columns for
    # each metric's mean, std, and per-fold scores.
    cv = pd.DataFrame(search.cv_results_)
    cv.to_csv(config.RESULT_DIRS["metrics"] / f"{module.NAME}_cv_results.csv", index=False)

    # Extract the row index of the best-performing combination
    best = search.best_index_
    # Build a summary of every metric for the best combination
    per_metric = {
        m: {"cv_mean": cv.loc[best, f"mean_test_{m}"], "cv_std": cv.loc[best, f"std_test_{m}"],
            "per_fold": [cv.loc[best, f"split{k}_test_{m}"] for k in range(len(folds))]}
        for m in SCORING
    }

    # Save the best trained model to disk using joblib (a binary format optimised for
    # large numpy arrays inside sklearn models)
    joblib.dump(search.best_estimator_, config.MODELS / f"{module.NAME}.joblib")
    # Also save just the preprocessor (useful if we want to transform data without predicting)
    joblib.dump(search.best_estimator_.named_steps["prep"], config.MODELS / f"{module.NAME}_preprocessor.joblib")

    # Save a JSON config file documenting exactly how this model was trained:
    # which hyperparameters won, what the search space was, and how it scored
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

    # Print a one-line progress summary to the console
    print(f"  {module.LABEL}: CV macro-F1 {per_metric['f1_macro']['cv_mean']:.3f} "
          f"(folds {np.round(per_metric['f1_macro']['per_fold'], 3)}), {elapsed:.0f}s", flush=True)
    return search.best_estimator_, cfg


# -------------------------------------------------------------------------------------
# DUMMY BASELINE MODEL
# -------------------------------------------------------------------------------------
# fit_dummy() trains a deliberately naive model that serves as a sanity-check baseline.
#
# What is a DummyClassifier?
#   It ignores all input features and simply predicts based on the class distribution
#   in the training data. With strategy="prior", it always predicts the most common
#   class (which is "On-Time" in our data).
#
# Why do we need this?
#   If our real models can't beat the dummy, something is seriously wrong. It sets
#   the absolute floor: any useful model MUST do better than "just guess On-Time".
# -------------------------------------------------------------------------------------
def fit_dummy(ctx):
    """Reference only: predicts the training class prior (argmax = always On-Time)."""
    model = DummyClassifier(strategy="prior").fit(ctx["train"][FEATURES], ctx["train"]["y"])
    joblib.dump(model, config.MODELS / "dummy_prior.joblib")
    return model


# =====================================================================================
# ANOMALY DETECTION MODEL TRAINING
# =====================================================================================
# fit_anomaly() trains unsupervised anomaly detection models.
#
# How anomaly detection works here:
#   1. We train the model ONLY on "normal" (On-Time) shipments – it learns what a
#      typical on-time shipment looks like.
#   2. When we show it a new shipment, it assigns an "anomaly score" – how different
#      this shipment is from what it learned as normal.
#   3. High anomaly score → the shipment looks unusual → might be delayed.
#
# The model never sees the labels during training (that's why it's "unsupervised").
# We only use labels afterwards to EVALUATE which hyperparameter settings produced
# the best anomaly scores (measured by PR-AUC on the validation set).
#
# PR-AUC = Precision-Recall Area Under the Curve. It tells us how well the anomaly
# scores rank truly delayed shipments above on-time ones.
# =====================================================================================
def fit_anomaly(module, ctx):
    # Unpack the training data, CV folds, and the validation fold index
    train, folds, k = ctx["train"], ctx["folds"], ctx["val_fold"]

    # Use the last fold: train on its training portion, evaluate on its validation portion
    tr_idx, val_idx = folds[k]
    fold_train, val = train.iloc[tr_idx], train.iloc[val_idx]

    # Filter to ONLY on-time shipments (y == 0) for model fitting.
    # The model should learn what "normal" looks like, not what "delayed" looks like.
    normal = fold_train[fold_train["y"] == 0][ANOMALY_INPUT_COLUMNS]

    # Create binary labels for evaluation: True = delayed (At-Risk or Critical), False = On-Time
    val_pos = (val["y"] > 0).values  # anomaly-positive = delayed (At-Risk or Critical)

    # --- Grid search over all hyperparameter combinations ---
    # (Anomaly models use a full grid search because the search space is small enough)
    rows = []
    t0 = time.time()
    for params in module.param_grid():
        # Build and fit the anomaly pipeline on normal-only data
        pipe = module.build_pipeline(**params).fit(module.fit_rows(normal))
        # Score every validation shipment: higher score = more anomalous
        s = module.anomaly_score(pipe, val[ANOMALY_INPUT_COLUMNS])
        # Record this combination's PR-AUC and ROC-AUC for delayed and critical classes
        rows.append({**params, "val_pr_auc_delayed": average_precision_score(val_pos, s),
                     "val_roc_auc_delayed": roc_auc_score(val_pos, s),
                     "val_pr_auc_critical": average_precision_score(val["y"].values == 2, s)})

    # Collect all results into a DataFrame and sort by best PR-AUC (delayed)
    grid = pd.DataFrame(rows).sort_values("val_pr_auc_delayed", ascending=False)
    grid.to_csv(config.RESULT_DIRS["anomaly"] / f"{module.NAME}_tuning_grid.csv", index=False)

    # Extract the best hyperparameters (top row of the sorted grid)
    best_params = {k: v for k, v in grid.iloc[0].items() if not k.startswith("val_") and not (isinstance(v, float) and np.isnan(v))}
    # Ensure integer-typed hyperparameters are actually ints (they may be floats from pandas)
    for key in ("n_estimators", "max_samples", "degree"):
        if key in best_params:
            best_params[key] = int(best_params[key])

    # --- Refit the final model on ALL on-time training data (not just the fold's portion) ---
    train_normal = train[train["y"] == 0][ANOMALY_INPUT_COLUMNS]
    fit_X = module.fit_rows(train_normal)
    final = module.build_pipeline(**best_params).fit(fit_X)

    # Compute a threshold for flagging shipments as anomalous.
    # We use a high quantile (e.g., 95th percentile) of anomaly scores on the
    # normal training data. Anything scoring above this threshold is "suspicious".
    threshold = float(np.quantile(module.anomaly_score(final, train_normal), config.ANOMALY_FLAG_QUANTILE))
    elapsed = time.time() - t0

    # Save the fitted model and threshold together
    joblib.dump({"pipeline": final, "threshold": threshold}, config.MODELS / f"{module.NAME}.joblib")

    # Save a JSON configuration file documenting how the anomaly model was trained
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


# =====================================================================================
# ABLATION STUDIES AND SENSITIVITY CHECKS
# =====================================================================================
#
# What is an "ablation"?
#   In ML, an ablation study means deliberately REMOVING a group of features (inputs)
#   and re-training the model to see how much performance drops. If removing weather
#   features causes a big drop in F1, weather features are clearly important. If
#   removing them barely changes anything, they might not be adding much value.
#
# What is a "sensitivity check"?
#   Sensitivity checks test whether the model is robust under different conditions:
#   - What if we only train on labels that were truly known at the time?
#   - What if we exclude rows with imputed (filled-in) data?
#   - What if we only look at sea shipments?
#   - What if we predict 24 hours ahead instead of the default lag?
#   If performance holds up, the model is robust. If it collapses, we've found a
#   fragile assumption we should worry about.
# =====================================================================================

# -------------------------------------------------------------------------------------
# _refit: Helper that re-trains a model with its already-tuned hyperparameters on a
# different subset of features. It clones (copies) the model so the original stays
# untouched, swaps in a new preprocessor that only knows about the requested features,
# and fits on the given training data.
# -------------------------------------------------------------------------------------
def _refit(model, module, train, numeric, categorical):
    # clone() creates a fresh copy of the pipeline with the same hyperparameters
    pipe = clone(model)
    # Replace the preprocessor with one built for the current feature subset
    pipe.set_params(prep=make_preprocessor(module.PREPROCESSING, numeric=numeric, categorical=categorical))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        # Train the cloned model on only the specified features
        return pipe.fit(train[numeric + categorical], train["y"])


# -------------------------------------------------------------------------------------
# _score: Helper that runs a trained model on a test set and returns a dictionary
# of performance metrics.
# -------------------------------------------------------------------------------------
def _score(pipe, df, cols):
    # Generate predictions for every row in the test set
    pred = pipe.predict(df[cols])
    return {
        # macro_f1: balanced F1 across all three classes
        "macro_f1": f1_score(df["y"], pred, average="macro"),
        # critical_recall: what fraction of truly Critical shipments did we catch?
        "critical_recall": recall_score(df["y"], pred, labels=[2], average="macro", zero_division=0),
        # at_risk_recall: what fraction of truly At-Risk shipments did we catch?
        "at_risk_recall": recall_score(df["y"], pred, labels=[1], average="macro", zero_division=0),
        # Plain accuracy: overall percentage of correct predictions
        "accuracy": float((pred == df["y"].values).mean()), "n_train": None, "n_test": len(df)}


def ablations_and_sensitivity(models, ctx):
    """Each row: one supervised model re-fitted with its tuned hyperparameters under a variant."""
    # Import the lists of numeric and categorical feature names
    from .features import CATEGORICAL, NUMERIC

    train, test, info = ctx["train"], ctx["test"], ctx["split_info"]

    # Define which feature groups represent "operational state" – i.e., features that
    # capture the current real-time situation at ports, hubs, and routes.
    # Removing these simulates "what if we only had booking-time information?"
    state_groups = ["origin_congestion", "container_dwell", "anchor_queue", "weather", "disruptions", "destination_state", "hub_state"]
    state_cols = {c for g in state_groups for c in FEATURE_GROUPS[g]}

    # "filled_*" columns are flags indicating that a value was imputed (filled in
    # because the original was missing). We test sensitivity to these imputations.
    filled_inputs = ["filled_route_id", "filled_carrier", "filled_shipping_mode", "filled_cargo_type",
                     "filled_container_teu", "filled_shipment_priority", "filled_expected_arrival_ts"]

    # -------------------------------------------------------------------------------------
    # Define all the experimental variants to run.
    # Each variant is: (training_data, test_data, numeric_features, categorical_features)
    # -------------------------------------------------------------------------------------
    variants = {
        # Baseline: the full model with all features and all data (for comparison)
        "full model (reference)": (train, test, NUMERIC, CATEGORICAL),

        # ABLATION: Remove port/carrier identity features.
        # Tests whether the model relies on memorising specific ports/carriers
        # rather than learning general patterns.
        "ablation: no port/carrier identity": (train, test, NUMERIC, [c for c in CATEGORICAL if c not in IDENTITY_FEATURES]),

        # ABLATION: Remove all operational state features (congestion, weather, etc.)
        # Tests how much real-time information contributes beyond static booking info.
        "ablation: no operational state (booking-time info only)": (train, test, [c for c in NUMERIC if c not in state_cols], CATEGORICAL),

        # ABLATION: Remove historical delay rate features.
        # Tests whether past delay patterns for routes/carriers add predictive power.
        "ablation: no historical delay rates": (train, test, [c for c in NUMERIC if c not in FEATURE_GROUPS["historical_delay_rates"]], CATEGORICAL),

        # SENSITIVITY: Only train on shipments whose actual arrival was known before
        # the test period started. This simulates a stricter "no future information" rule.
        "sensitivity: train only on labels available at test start": (
            train[train["actual_arrival_ts"] < info["test_start"]], test, NUMERIC, CATEGORICAL),

        # SENSITIVITY: Exclude any shipments that had missing values filled in.
        # Tests whether imputations are distorting the model's predictions.
        "sensitivity: exclude shipments with filled inputs (train+test)": (
            train[~train[filled_inputs].any(axis=1)], test[~test[filled_inputs].any(axis=1)], NUMERIC, CATEGORICAL),

        # SENSITIVITY: Train and test on sea shipments only.
        # Tests whether the model works for the dominant shipping mode on its own.
        "sensitivity: sea shipments only (train+test)": (
            train[train["shipping_mode"] == "sea"], test[test["shipping_mode"] == "sea"], NUMERIC, CATEGORICAL),
    }

    # SENSITIVITY: Predict 24 hours before departure instead of the default lag.
    # This rebuilds the entire feature table with a 24-hour cutoff to see if
    # the model still works when it has to predict further in advance.
    ctx24 = prepare_data(lag_hours=24, save=False)
    # Use the same shipment IDs as in the original train/test so results are comparable
    ids_tr, ids_te = set(train["shipment_id"]), set(test["shipment_id"])
    t24 = ctx24["table"]
    variants["sensitivity: prediction 24 h before departure"] = (
        t24[t24["shipment_id"].isin(ids_tr)], t24[t24["shipment_id"].isin(ids_te)], NUMERIC, CATEGORICAL)

    # --- Run every variant for every supervised model and collect results ---
    rows = []
    for module in SUPERVISED:
        for variant, (tr, te, num, cat) in variants.items():
            # Re-train the model (same tuned hyperparameters) on this variant's data/features
            pipe = _refit(models[module.NAME], module, tr, num, cat)
            # Score it on the variant's test set
            r = _score(pipe, te, num + cat)
            # Record the model name, variant description, and sizes alongside the scores
            r.update({"model": module.LABEL, "variant": variant, "n_train": len(tr), "n_features": len(num) + len(cat)})
            rows.append(r)
        print(f"  ablations done: {module.LABEL}", flush=True)

    # Combine all results into a single DataFrame and save to CSV
    out = pd.DataFrame(rows)[["model", "variant", "n_train", "n_test", "n_features", "macro_f1", "critical_recall", "at_risk_recall", "accuracy"]]
    out.to_csv(config.RESULT_DIRS["comparison"] / "ablation_and_sensitivity.csv", index=False)
    return out


# =====================================================================================
# MAIN ENTRY POINT – ORCHESTRATES THE ENTIRE TRAINING PIPELINE
# =====================================================================================
# When you run this file (or call main()), it executes the full training pipeline
# in order:
#   1. Create output directories
#   2. Prepare data (load, feature-engineer, split, set up cross-validation)
#   3. Tune and train all 4 supervised models (LR, RF, XGBoost, LightGBM)
#   4. Train a dummy baseline for comparison
#   5. Train 2 anomaly detection models (Isolation Forest, One-Class SVM)
#   6. Run ablation studies and sensitivity checks
# =====================================================================================
def main():
    # Create all output directories (models/, metrics/, data/, etc.) if they don't exist
    config.ensure_dirs()

    # --- Step 1: Data preparation ---
    print("preparing data ...", flush=True)
    ctx = prepare_data()
    print(f"  train {len(ctx['train'])}, test {len(ctx['test'])}, test starts {ctx['split_info']['test_start']}", flush=True)

    # --- Step 2: Tune and train supervised models ---
    print("tuning supervised models ...", flush=True)
    models, configs = {}, {}
    for module in SUPERVISED:
        # For each model type, run hyperparameter tuning with cross-validation
        # and save the best model
        models[module.NAME], configs[module.NAME] = tune_supervised(module, ctx)

    # Train the dummy baseline (always predicts the most common class)
    models["dummy_prior"] = fit_dummy(ctx)

    # --- Step 3: Train anomaly detection models ---
    print("fitting anomaly detectors ...", flush=True)
    for module in ANOMALY:
        # Train each anomaly model on on-time data only, select best params by PR-AUC
        models[module.NAME], configs[module.NAME] = fit_anomaly(module, ctx)

    # --- Step 4: Run ablations and sensitivity checks ---
    print("ablations and sensitivity checks ...", flush=True)
    ablations_and_sensitivity(models, ctx)

    # Return everything so callers (e.g., notebooks) can inspect the results
    return ctx, models, configs


# Standard Python idiom: only run main() if this file is executed directly
# (not when it's imported as a module by another file)
if __name__ == "__main__":
    main()
