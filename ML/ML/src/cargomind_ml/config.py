"""Paths, seeds and every modelling decision that is a *parameter* rather than code.

Changing a value here and re-running `ML/ML/scripts/run_ml_pipeline.py` reproduces the whole
ML stage (features -> splits -> tuning -> evaluation -> comparison) end to end.
"""
# 'Path' is a handy library that lets us construct file/folder paths in a way
# that works on any operating system (Windows, Mac, Linux).
from pathlib import Path

# =====================================================================================
# DIRECTORY LAYOUT
# These variables point to important folders in the project so the rest of the code
# can find data files, save trained models, and store results without hard-coding paths.
# =====================================================================================

# __file__ is the path to THIS file (config.py).  .resolve() makes it absolute,
# and .parents[2] goes up two directories to reach the "ML/ML" folder —
# the root of the machine-learning modelling stage.
ML_ROOT = Path(__file__).resolve().parents[2]  # ML/ML: the ML modelling stage

# One more directory up gives us the broader ML workflow folder
# (which also contains Dataset, EDA, and Evaluation stages).
ML_WORKFLOW = ML_ROOT.parent  # ML/: Dataset -> EDA -> ML -> Evaluation

# One more level up reaches the very top of the entire Git repository.
PROJECT_ROOT = ML_WORKFLOW.parent  # repository root

# Path to the Exploratory Data Analysis (EDA) source code.
# We reuse some feature-engineering functions from the EDA stage (read-only).
EDA_SRC = ML_WORKFLOW / "EDA" / "src"  # reused (read-only) EDA feature engineering (installed as cargomind_eda)

# Where we save trained model files (e.g., serialised model objects).
MODELS = ML_ROOT / "models"

# Where we save summary reports (e.g., PDF or HTML write-ups).
RESULTS = ML_ROOT / "results"

# Where we save human-readable report documents.
REPORTS = ML_ROOT / "reports"

# A dictionary of sub-folders inside RESULTS, one for each kind of output.
# For example, "metrics" stores accuracy numbers, "confusion_matrices" stores
# tables showing how often the model confused one class for another, etc.
RESULT_DIRS = {
    k: RESULTS / k
    for k in ("data", "metrics", "classification_reports", "confusion_matrices", "feature_importance", "anomaly", "comparison", "figures")
}

# =====================================================================================
# RANDOM SEED
# =====================================================================================
# A "seed" is a starting number for the random-number generator.  Setting it to a
# fixed value (here 42 — a popular convention from "The Hitchhiker's Guide to the
# Galaxy") means that every time you run the code you get the EXACT same random
# numbers, which makes your results perfectly reproducible.  Without a seed,
# re-running the pipeline could give slightly different results each time.
SEED = 42

# =====================================================================================
# TARGET VARIABLE (what the model is trying to predict)
# =====================================================================================

# --- Target (unchanged from the project specification / EDA) ---

# TARGET is the column name in our data that tells us whether a shipment was
# delayed.  This is what the ML model learns to predict.
TARGET = "delay_status"

# OUTCOME is the column with the raw number (actual hours of delay).
# It is used to compute the TARGET labels but is NOT given to the model as input
# (that would be cheating — see "leakage" comments in features.py).
OUTCOME = "actual_delay_hours"

# CLASSES lists the three categories our model predicts:
#   - "On-Time"  (code 0): shipment arrived within 12 hours of schedule
#   - "At-Risk"  (code 1): delayed between 12 and 48 hours — needs attention
#   - "Critical" (code 2): delayed more than 48 hours — serious problem
# The numbers 0, 1, 2 are an "ordinal encoding" — they preserve the natural
# order from best to worst.
CLASSES = ["On-Time", "At-Risk", "Critical"]  # encoded 0 / 1 / 2 (ordinal order)

# THRESHOLDS define the boundaries between the three classes above.
# A delay of 0–12 h → On-Time, 12–48 h → At-Risk, >48 h → Critical.
THRESHOLDS = {"on_time_max": 12.0, "at_risk_max": 48.0}

# =====================================================================================
# PREDICTION TIMING
# =====================================================================================

# --- Prediction timestamp ---
# Pre-departure prediction at departure_ts - LAG_HOURS. Only hourly port/weather records with
# ts_hour <= cutoff, disruptions already started at the cutoff, and outcomes of shipments that
# had *arrived* before the cutoff are used. 1 h is the EDA convention and is safe whether
# ts_hour marks the start or the end of the hour (EDA notebook 09, I3-2).

# LAG_HOURS controls how far BEFORE a ship departs we make our prediction.
# A value of 1 means "predict 1 hour before departure."  We only allow the
# model to see data that was available at that point in time — no peeking at
# things that happen after the ship leaves!
LAG_HOURS = 1

# =====================================================================================
# TRAIN / TEST SPLIT SETTINGS
# =====================================================================================

# --- Chronological split ---

# TEST_FRACTION = 0.25 means the newest 25 % of shipments (sorted by departure
# time) become the "test set."  The test set is data the model has NEVER seen
# during training; we use it to measure how well the model would work in the
# real world on future shipments.
TEST_FRACTION = 0.25  # latest 25 % of observed shipments (by departure time) form the test set

# PURGE_GAP_HOURS = 72 creates a 3-day "no-man's-land" between the training
# data and the test data.  Shipments that departed in this 72-hour gap are
# thrown away — they are too close in time and could let information leak
# from the test period into training.
PURGE_GAP_HOURS = 72  # departures in the 72 h before the test start are used for neither train nor test

# CV_FOLDS = 3 is the number of "cross-validation" rounds used to tune model
# settings.  Cross-validation means we split the training data into parts,
# train on some parts and validate on the held-out part, repeating this
# multiple times to get a more reliable estimate of model quality.
# "Expanding-window" means each fold uses all earlier data for training
# and a later chunk for validation (respecting time order).
CV_FOLDS = 3  # expanding-window folds inside the training period for hyperparameter tuning

# CV_VAL_FRACTION = 0.15 means each validation block (the held-out chunk in
# each fold) contains about 15 % of the training-period shipments.
CV_VAL_FRACTION = 0.15  # each validation block = 15 % of the training period's shipments

# TUNING_SCORING tells the search what metric to optimise.
# "f1_macro" is the F1-score averaged equally across all three classes.
# We don't use simple "accuracy" because our classes are imbalanced — there
# are many more On-Time shipments than Critical ones.  Accuracy would let the
# model get a high score by always predicting "On-Time," ignoring the rare
# but important delays.  F1-macro forces the model to care equally about
# predicting At-Risk and Critical shipments correctly.
TUNING_SCORING = "f1_macro"  # never accuracy: treats At-Risk and Critical as equally important as On-Time

# =====================================================================================
# ANOMALY DETECTION SETTINGS
# =====================================================================================
# Anomaly detection is a separate model that flags shipments whose port
# conditions look "abnormal" compared to typical On-Time departures.
# Think of it as an alarm system: "something unusual is going on at this port."

# --- Anomaly detection ---

# ANOMALY_FLAG_QUANTILE = 0.95 means we flag the top 5 % most unusual-looking
# shipments.  95th percentile = the score above which only 5 % of normal
# (On-Time) training shipments fall.  This gives us a 5 % false-alarm budget:
# roughly 1 in 20 normal shipments will be incorrectly flagged, which is an
# acceptable trade-off for catching truly anomalous conditions.
ANOMALY_FLAG_QUANTILE = 0.95  # flag scores above the 95th percentile of *training On-Time* scores (5 % false-alarm budget on normals)

# OCSVM_MAX_TRAIN limits how many rows we feed into the One-Class SVM anomaly
# detector during training.  One-Class SVM with an RBF kernel has a
# computational cost that grows very fast with the number of rows (somewhere
# between n² and n³).  Capping at 3 000 rows keeps training fast while still
# giving the model enough examples to learn what "normal" looks like.
OCSVM_MAX_TRAIN = 3000  # RBF One-Class SVM is O(n^2)-O(n^3); subsample normal training rows

# =====================================================================================
# HYPERPARAMETER TUNING BUDGET
# =====================================================================================
# "Hyperparameters" are settings of a machine-learning model that you choose
# BEFORE training (e.g., how deep a decision tree can grow, how fast the model
# learns).  "Randomised search" tries random combinations of these settings and
# picks the best one.  N_ITER controls how many random combinations to try for
# each model type — more iterations = longer search but potentially a better
# configuration.

# --- Randomised search budget ---
N_ITER = {"logistic_regression": 20, "random_forest": 30, "xgboost": 40, "lightgbm": 40}

# N_JOBS = -1 means "use ALL available CPU cores" for parallel processing.
# This makes training and hyperparameter search much faster on multi-core
# machines.
N_JOBS = -1

# BOOTSTRAP_REPS = 1000 is the number of bootstrap resampling rounds used to
# compute confidence intervals for our test-set metrics.  Bootstrapping
# randomly re-samples the test results many times to estimate how much our
# metric (e.g., F1-score) would vary if we had a slightly different test set.
# 1 000 rounds is a standard choice that gives stable intervals.
BOOTSTRAP_REPS = 1000  # day-block bootstrap for test-metric confidence intervals


def ensure_dirs():
    """Create all output directories if they don't already exist.

    This is called at the start of the pipeline to make sure the folders for
    models, reports, and every kind of result are in place before we try to
    save anything.  'parents=True' creates any missing intermediate folders,
    and 'exist_ok=True' means it won't complain if the folder already exists.
    """
    for p in [MODELS, REPORTS, *RESULT_DIRS.values()]:
        p.mkdir(parents=True, exist_ok=True)
