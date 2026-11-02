"""Paths, seeds and every modelling decision that is a *parameter* rather than code.

Changing a value here and re-running `ML/ML/scripts/run_ml_pipeline.py` reproduces the whole
ML stage (features -> splits -> tuning -> evaluation -> comparison) end to end.
"""
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[2]  # ML/ML: the ML modelling stage
ML_WORKFLOW = ML_ROOT.parent  # ML/: Dataset -> EDA -> ML -> Evaluation
PROJECT_ROOT = ML_WORKFLOW.parent  # repository root
EDA_SRC = ML_WORKFLOW / "EDA" / "src"  # reused (read-only) EDA feature engineering (installed as cargomind_eda)
MODELS = ML_ROOT / "models"
RESULTS = ML_ROOT / "results"
REPORTS = ML_ROOT / "reports"

RESULT_DIRS = {
    k: RESULTS / k
    for k in ("data", "metrics", "classification_reports", "confusion_matrices", "feature_importance", "anomaly", "comparison", "figures")
}

SEED = 42

# --- Target (unchanged from the project specification / EDA) ---
TARGET = "delay_status"
OUTCOME = "actual_delay_hours"
CLASSES = ["On-Time", "At-Risk", "Critical"]  # encoded 0 / 1 / 2 (ordinal order)
THRESHOLDS = {"on_time_max": 12.0, "at_risk_max": 48.0}

# --- Prediction timestamp ---
# Pre-departure prediction at departure_ts - LAG_HOURS. Only hourly port/weather records with
# ts_hour <= cutoff, disruptions already started at the cutoff, and outcomes of shipments that
# had *arrived* before the cutoff are used. 1 h is the EDA convention and is safe whether
# ts_hour marks the start or the end of the hour (EDA notebook 09, I3-2).
LAG_HOURS = 1

# --- Chronological split ---
TEST_FRACTION = 0.25  # latest 25 % of observed shipments (by departure time) form the test set
PURGE_GAP_HOURS = 72  # departures in the 72 h before the test start are used for neither train nor test
CV_FOLDS = 3  # expanding-window folds inside the training period for hyperparameter tuning
CV_VAL_FRACTION = 0.15  # each validation block = 15 % of the training period's shipments
TUNING_SCORING = "f1_macro"  # never accuracy: treats At-Risk and Critical as equally important as On-Time

# --- Anomaly detection ---
ANOMALY_FLAG_QUANTILE = 0.95  # flag scores above the 95th percentile of *training On-Time* scores (5 % false-alarm budget on normals)
OCSVM_MAX_TRAIN = 3000  # RBF One-Class SVM is O(n^2)-O(n^3); subsample normal training rows

# --- Randomised search budget ---
N_ITER = {"logistic_regression": 20, "random_forest": 30, "xgboost": 40, "lightgbm": 40}
N_JOBS = -1

BOOTSTRAP_REPS = 1000  # day-block bootstrap for test-metric confidence intervals


def ensure_dirs():
    for p in [MODELS, REPORTS, *RESULT_DIRS.values()]:
        p.mkdir(parents=True, exist_ok=True)
