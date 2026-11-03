# =========================================================================================
# INFERENCE MODULE — Using Trained Models to Make Predictions on New Shipments
# =========================================================================================
#
# "Inference" means taking a model that has already been trained (i.e., it has already
# learned patterns from historical data) and using it to make predictions on brand-new,
# never-before-seen data. Think of it like a student who has finished studying (training)
# and is now taking the exam (inference) — they apply what they learned to new questions.
#
# This module is the "prediction engine" of CargoMind. Given information about a new
# shipment (where it's going, what carrier is used, the weather at the port, etc.),
# it produces two outputs:
#
#   1. CLASSIFICATION — Will this shipment be "On-Time", "At-Risk", or "Critical"?
#      (Uses an XGBoost model, which is a powerful tree-based machine learning algorithm.)
#
#   2. ANOMALY DETECTION — Are the conditions at the origin port unusual/abnormal?
#      (Uses an Isolation Forest model, which spots data points that look "weird"
#       compared to what the model saw during training.)
#
# These two predictions are kept SEPARATE — they answer different questions and are
# never combined into a single score.
# =========================================================================================

"""Inference for the selected ML models: XGBoost delay-class classifier + Isolation Forest anomaly detector.

    from cargomind_ml import inference as mli
    feats = mli.build_features_for_shipments(shipments_df)     # shipment attributes -> the 51 model features
    results = mli.predict(feats)                                # list of result dicts (contract below)

Selection (product decision, recorded in ML/ML/models/selection.json):
* classifier: XGBoost (`xgboost`, provisional primary; Logistic Regression is the research challenger)
                                              - On-Time / At-Risk / Critical, argmax of the 3 class probabilities;
* anomaly detector: Isolation Forest (`isolation_forest`) - unusual origin operating state, score + flag.
The two outputs are kept separate. They are never combined into one prediction. The other trained models
(logistic_regression, random_forest, lightgbm, one_class_svm) stay loadable by name for research/comparison.

Feature construction (`build_features_for_shipments`) reuses the training-time feature code unchanged
(`cargomind_eda.features.build_shipment_features` + the ML-stage steps of `features.build_modelling_table`) and reads
port telemetry, weather, disruptions, holidays and historical outcomes from `ML/Dataset/Processed`. It can only
build features for departures whose cutoff hour (departure_ts - 1 h) is covered by those tables
(2025-01-01 .. 2025-02-11 15:00 UTC). There is no live data feed; see docs/architecture.md, "Known gaps".

Result contract (one dict per input row, `CONTRACT_VERSION`):
    shipment_id                str | None
    prediction_timestamp_utc   ISO time at which the prediction was produced
    feature_cutoff_utc         ISO cutoff the features describe (departure_ts - 1 h), or None if not supplied
    classification             {model_name, model_label, model_version, predicted_class, class_probabilities,
                                decision_rule, probabilities_note}
    anomaly                    {model_name, model_label, model_version, anomaly_score, anomaly_flag, threshold,
                                score_definition}   (None when anomaly=False)
    warnings                   list[str] (missing-data and unseen-category notes; empty when none)
"""
from __future__ import annotations

# --- Standard library imports ---
import hashlib       # Used to create a unique "fingerprint" (hash) of model files, so we can track versions
import json          # For reading/writing JSON configuration files
from dataclasses import dataclass   # A convenient way to create simple classes that hold data
from datetime import datetime, timezone   # For working with dates/times and UTC timezone
from functools import lru_cache    # A "cache" that remembers results of expensive function calls so they only run once
from pathlib import Path           # A modern way to work with file paths on any operating system

# --- Third-party library imports ---
import joblib        # Used to save and load trained ML models to/from disk (like saving a file)
import numpy as np   # The fundamental library for numerical computing in Python (arrays, math, etc.)
import pandas as pd  # The go-to library for working with tabular data (spreadsheet-like DataFrames)

# --- Internal project imports ---
# Import project configuration (paths, constants, etc.)
from . import config
# Import feature definitions from the features module:
#   ANOMALY_FEATURES — the subset of features used by the anomaly detector
#   CATEGORICAL — list of categorical (text/label) feature columns
#   FEATURES — the full list of 51 feature column names the classifier needs
#   HIST_MIN_N — minimum number of historical shipments needed to compute route history
#   NUMERIC — list of numeric (number) feature columns
#   _hub_holidays_in_voyage — helper that counts public holidays at the hub port during a voyage
from .features import ANOMALY_FEATURES, CATEGORICAL, FEATURES, HIST_MIN_N, NUMERIC, _hub_holidays_in_voyage

# Import from the EDA (Exploratory Data Analysis) package:
#   features — functions that build features from raw data
#   io — functions that load processed datasets from disk
#   lineage — functions that track which data came from where (data provenance)
from cargomind_eda import features as eda_features, io as eda_io, lineage as eda_lineage  # noqa: E402

# =========================================================================================
# CONSTANTS — Fixed values used throughout this module
# =========================================================================================

# A version string for the output format. If we ever change what the prediction results
# look like, we bump this version so downstream consumers know which format to expect.
CONTRACT_VERSION = "cargomind-ml-inference/1.0"

# Path to the file that records which models were officially "selected" (chosen) for use
SELECTION_PATH = config.MODELS / "selection.json"

# The primary (default) classifier model — XGBoost was chosen as the best performer
PRIMARY_CLASSIFIER = "xgboost"

# The primary (default) anomaly detector — Isolation Forest was chosen for spotting unusual port conditions
ANOMALY_DETECTOR = "isolation_forest"

# Other models that were trained during research but are NOT the primary choice.
# They can still be loaded for comparison or experimentation.
RESEARCH_CLASSIFIERS = ("logistic_regression", "random_forest", "lightgbm")
RESEARCH_DETECTORS = ("one_class_svm",)

# Human-readable labels for each model (for display in results)
LABELS = {"xgboost": "XGBoost", "logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
          "lightgbm": "LightGBM", "isolation_forest": "Isolation Forest", "one_class_svm": "One-Class SVM"}

# Descriptions of what the anomaly score means for each detector type.
# "Higher = more unusual" means a bigger number indicates a stranger/more abnormal situation.
SCORE_DEFINITION = {"isolation_forest": "-score_samples(X); higher = more unusual origin operating state",
                    "one_class_svm": "-decision_function(X); higher = more unusual origin operating state"}

# ------------------------------------------------------------------------------------------
# KNOWN CATEGORIES — The specific text values that each categorical feature can take.
# These are the values the model saw during training. If a new shipment has a value the
# model has never seen (e.g., a brand-new port code), it will still work but the model
# essentially treats that unknown value as a neutral "I don't know this" baseline.
# The technical term is "one-hot encoding with handle_unknown='ignore'" — unseen values
# get encoded as all zeros, meaning "none of the known categories."
# ------------------------------------------------------------------------------------------
KNOWN_CATEGORIES = {
    "origin_port_id": ["AEJEA", "CNSHA", "DEHAM", "INMAA", "INNSA", "LKCMB", "MYPKG", "NLRTM", "SGSIN", "USLAX"],
    "destination_port_id": ["AEJEA", "CNSHA", "DEHAM", "INMAA", "INNSA", "LKCMB", "MYPKG", "NLRTM", "SGSIN", "USLAX"],
    "hub_port_id": ["AEJEA", "DIRECT", "LKCMB", "MYPKG", "SGSIN"],
    "route_type": ["direct", "transshipment"],
    "carrier": ["BlueWave Shipping", "Coral Bay Logistics", "Meridian Cargo", "Northgate Maritime", "Oceanic Lines",
                "Pacific Star Line"],
    "cargo_type": ["general", "hazardous", "reefer"],
    "shipping_mode": ["air", "sea"],
}

# ------------------------------------------------------------------------------------------
# STRUCTURAL NaN PREFIXES — Some features are EXPECTED to be missing (NaN = "Not a Number")
# in certain situations, and that's perfectly normal:
#   - "h_" features: hub port data — missing when the route is direct (no hub port)
#   - "wx_h_" features: weather at hub — also missing for direct routes
#   - "dis_h_" features: disruptions at hub — also missing for direct routes
#   - "hol_h_" features: holidays at hub — also missing for direct routes
#   - "hist_" features: historical statistics — missing when fewer than HIST_MIN_N past
#     shipments have arrived on this route (not enough data to compute averages)
# The trained model pipelines know how to handle these missing values (XGBoost handles NaN
# natively; Isolation Forest uses median imputation, i.e., it fills in the middle value).
# ------------------------------------------------------------------------------------------
STRUCTURAL_NAN_PREFIXES = ("h_", "wx_h_", "dis_h_", "hol_h_", "hist_")

# The columns you must provide about each shipment in order to make a prediction.
# These describe the shipment BEFORE it departs — things you know ahead of time.
SHIPMENT_INPUT_COLUMNS = ["shipment_id", "route_id", "origin_port_id", "destination_port_id", "carrier", "shipping_mode",
                          "cargo_type", "container_teu", "shipment_priority", "departure_ts", "expected_arrival_ts"]

# OUTCOME COLUMNS — These describe what ACTUALLY happened (did it arrive late? by how much?).
# These must NEVER be provided when making predictions, because that would be cheating:
# you can't know the actual arrival time before the shipment departs! The model needs to
# predict the future, not peek at the answer.
OUTCOME_COLUMNS = ["actual_arrival_ts", "actual_delay_hours", "delay_status"]


# =========================================================================================
# CUSTOM ERROR CLASSES — Specific error types so callers can catch and handle problems
# =========================================================================================

class FeatureValidationError(ValueError):
    """The feature rows cannot be scored. The message says why."""


class ShipmentInputError(ValueError):
    """Shipment attributes cannot be turned into model features. The message says why."""


class ModelArtifactError(RuntimeError):
    """A model artifact or its configuration is missing, altered or incompatible."""


# =========================================================================================
# HELPER FUNCTIONS
# =========================================================================================

def _sha256(path: Path) -> str:
    """Compute a SHA-256 hash (a unique fingerprint) of a file's contents.

    This is used to create a version identifier for model files — if even one byte
    of the model file changes, the hash will be completely different. This helps us
    track exactly which version of a model produced a given prediction.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_selection(path: Path = SELECTION_PATH) -> dict:
    """Load the model selection file that records which models were officially chosen.

    This JSON file documents the product decision about which models to use in production.
    """
    if not Path(path).exists():
        raise ModelArtifactError(f"model selection file not found: {path}")
    return json.loads(Path(path).read_text())


# =============================================================================================
# MODEL LOADING — Loading trained models from disk into memory so they can make predictions
# =============================================================================================

# A "dataclass" is a simple Python class that mainly holds data (like a struct or record).
# "frozen=True" means once created, its fields cannot be changed — this prevents accidental
# modification of a loaded model's metadata.

@dataclass(frozen=True)
class LoadedClassifier:
    """Holds everything about a loaded classification model.

    Think of this as a container that bundles together:
      - name: the model's short name (e.g., "xgboost")
      - label: a human-readable name (e.g., "XGBoost")
      - pipeline: the actual trained model object that can make predictions.
                  A "pipeline" is a chain of steps: first preprocess the data (scale numbers,
                  encode categories), then feed it to the model. Having it as one object means
                  you can't accidentally forget a preprocessing step.
      - config: the model's configuration (what features it expects, what algorithm it uses, etc.)
      - version: a unique identifier based on the file path and hash, so you know exactly which
                 model file produced a prediction
    """
    name: str
    label: str
    pipeline: object
    config: dict
    version: str


@dataclass(frozen=True)
class LoadedDetector:
    """Holds everything about a loaded anomaly detection model.

    Similar to LoadedClassifier, but also includes:
      - threshold: a cutoff value — if the anomaly score is above this number,
                   we flag the shipment as anomalous (unusual). Think of it like
                   a "weirdness limit": anything above the threshold is considered
                   abnormally different from what the model learned as normal.
    """
    name: str
    label: str
    pipeline: object
    threshold: float
    config: dict
    version: str


def _version(path: Path) -> str:
    """Create a version string for a model file, combining its relative path and a short hash.

    Example output: "models/xgboost.joblib@a1b2c3d4e5f6"
    The hash portion (after @) changes if the file contents change, acting as a fingerprint.
    """
    return f"{path.relative_to(config.PROJECT_ROOT).as_posix()}@{_sha256(path)[:12]}"


# @lru_cache(maxsize=None) is a "decorator" that makes this function remember its results.
# The first time you call load_classifier("xgboost"), it does all the work of loading the
# model from disk. The second time you call it with the same argument, it instantly returns
# the cached result without re-reading the file. This is important because loading a model
# from disk is slow, and we don't want to do it every time we make a prediction.
@lru_cache(maxsize=None)
def load_classifier(name: str = PRIMARY_CLASSIFIER) -> LoadedClassifier:
    """Load a saved supervised pipeline and check it against its config and the current feature schema.

    This function loads a trained classification model from disk and performs many safety checks
    to make sure the model file hasn't been corrupted, altered, or become incompatible with the
    current code. It's like opening a sealed envelope and verifying the contents match the label.

    "Supervised" means this model was trained with labeled examples — it was shown shipments
    AND told whether each one was On-Time, At-Risk, or Critical, so it could learn the patterns.
    """
    # --- Safety check 1: Is this a model name we recognize? ---
    if name not in (PRIMARY_CLASSIFIER, *RESEARCH_CLASSIFIERS):
        raise ModelArtifactError(f"unknown classifier {name!r}; available: {[PRIMARY_CLASSIFIER, *RESEARCH_CLASSIFIERS]}")

    # Build paths to the model file (.joblib) and its config file (.json)
    path, cfg_path = config.MODELS / f"{name}.joblib", config.MODELS / f"{name}_config.json"

    # --- Safety check 2: Do both files actually exist on disk? ---
    for p in (path, cfg_path):
        if not p.exists():
            raise ModelArtifactError(f"model artifact missing: {p}")

    # Read the configuration (a JSON file describing what the model expects)
    cfg = json.loads(cfg_path.read_text())

    # Load the actual trained model from the .joblib file
    # joblib is a library that can save and load Python objects (including trained models) to/from files
    pipe = joblib.load(path)

    # --- Safety check 3: Is the loaded object actually a scikit-learn Pipeline with the right steps? ---
    # A Pipeline should have exactly two steps: "prep" (preprocessing) and "model" (the actual algorithm)
    if not hasattr(pipe, "named_steps") or set(pipe.named_steps) != {"prep", "model"}:
        raise ModelArtifactError(f"{path} is not a (prep, model) Pipeline")

    # --- Safety check 4: Does the config's feature list match the current code's feature list? ---
    # If someone changed the features in the code but didn't retrain the model, this would catch it.
    if cfg["features"] != FEATURES:
        raise ModelArtifactError(f"{cfg_path} feature list differs from cargomind_ml.features.FEATURES")

    # --- Safety check 5: Does the model inside the pipeline match what the config says it should be? ---
    # e.g., if the config says "XGBClassifier" but the file contains a RandomForest, something is wrong.
    if type(pipe.named_steps["model"]).__name__ != cfg["estimator"]:
        raise ModelArtifactError(f"{path} holds {type(pipe.named_steps['model']).__name__}, config says {cfg['estimator']}")

    # --- Safety check 6: Was the preprocessor fitted on exactly the same columns as FEATURES? ---
    # "Fitted" means the preprocessor learned statistics (like the mean and standard deviation of
    # each numeric column) from the training data. It must have been fitted on the same columns
    # we're going to give it now.
    fitted_cols = [c for _, _, cols in pipe.named_steps["prep"].transformers_ if cols != "drop" for c in cols]
    if sorted(fitted_cols) != sorted(FEATURES):
        raise ModelArtifactError(f"{path} preprocessor was fitted on different columns than FEATURES")

    # --- Safety check 7: Does the model know about exactly 3 classes (On-Time=0, At-Risk=1, Critical=2)? ---
    if list(pipe.classes_) != [0, 1, 2]:
        raise ModelArtifactError(f"{path} classes {list(pipe.classes_)} are not the encoded {config.CLASSES}")

    # All checks passed! Return the loaded model bundled into a LoadedClassifier dataclass.
    return LoadedClassifier(name, LABELS[name], pipe, cfg, _version(path))


# Same caching strategy as load_classifier — load once, reuse forever.
@lru_cache(maxsize=None)
def load_detector(name: str = ANOMALY_DETECTOR) -> LoadedDetector:
    """Load a saved anomaly bundle {pipeline, threshold} and check it against its config.

    An anomaly detector is an "unsupervised" model — it was trained on "normal" data without
    being told what's normal or abnormal. Instead, it learned the shape of normal data, and
    anything that doesn't fit that shape is flagged as an anomaly (like spotting the odd one
    out in a crowd).

    The "bundle" is a dictionary containing:
      - pipeline: the trained anomaly detection model
      - threshold: the cutoff score above which a data point is considered anomalous
    """
    # --- Safety check 1: Is this a detector name we recognize? ---
    if name not in (ANOMALY_DETECTOR, *RESEARCH_DETECTORS):
        raise ModelArtifactError(f"unknown anomaly detector {name!r}; available: {[ANOMALY_DETECTOR, *RESEARCH_DETECTORS]}")

    # Build paths to the model file and its config file
    path, cfg_path = config.MODELS / f"{name}.joblib", config.MODELS / f"{name}_config.json"

    # --- Safety check 2: Do both files exist? ---
    for p in (path, cfg_path):
        if not p.exists():
            raise ModelArtifactError(f"model artifact missing: {p}")

    # Load the config and the model bundle from disk
    cfg = json.loads(cfg_path.read_text())
    bundle = joblib.load(path)

    # --- Safety check 3: Is the loaded object a dictionary with exactly "pipeline" and "threshold" keys? ---
    if not isinstance(bundle, dict) or set(bundle) != {"pipeline", "threshold"}:
        raise ModelArtifactError(f"{path} is not a {{pipeline, threshold}} bundle")

    # --- Safety check 4: Does the config's feature list match the current anomaly feature list? ---
    if cfg["features"] != ANOMALY_FEATURES:
        raise ModelArtifactError(f"{cfg_path} feature list differs from cargomind_ml.features.ANOMALY_FEATURES")

    # --- Safety check 5: Is the threshold a valid number, and does it match the config? ---
    # np.isfinite checks it's not infinity or NaN; np.isclose checks it matches the config value
    threshold = float(bundle["threshold"])
    if not np.isfinite(threshold) or not np.isclose(threshold, cfg["threshold"], rtol=0, atol=1e-12):
        raise ModelArtifactError(f"{path} threshold {threshold} does not match its config ({cfg['threshold']})")

    # All checks passed! Return the loaded detector.
    return LoadedDetector(name, LABELS[name], bundle["pipeline"], threshold, cfg, _version(path))


# =============================================================================================
# FEATURE VALIDATION — Making sure the input data is clean and well-formed before prediction
# =============================================================================================

def validate_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[list[str]]]:
    """Check the 51 model features. Returns the feature frame (model column order) and per-row warnings.

    Raises FeatureValidationError for: a non-DataFrame, an empty frame, missing columns, non-numeric values in numeric
    columns, infinite values, or missing (NaN) categorical values. Nothing is imputed or repaired here; the saved
    pipelines handle NaN in numeric columns exactly as during training.
    """
    # --- Check 1: Is the input actually a pandas DataFrame? ---
    if not isinstance(features, pd.DataFrame):
        raise FeatureValidationError(f"features must be a pandas DataFrame with columns {FEATURES}; got {type(features).__name__}")

    # --- Check 2: Does it have at least one row of data? ---
    if len(features) == 0:
        raise FeatureValidationError("features has no rows")

    # --- Check 3: Are all 51 required feature columns present? ---
    missing = [c for c in FEATURES if c not in features.columns]
    if missing:
        raise FeatureValidationError(f"missing feature columns: {missing}")

    # Make a copy of just the feature columns in the correct order so we don't modify the original
    X = features[FEATURES].copy()

    # --- Validate each NUMERIC column ---
    for c in NUMERIC:
        # Convert boolean columns to float (True→1.0, False→0.0) since the model expects numbers
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(float)
        # Try to convert the column to numbers; if it contains text like "abc", this will fail
        try:
            X[c] = pd.to_numeric(X[c], errors="raise").astype(float)
        except (ValueError, TypeError) as e:
            raise FeatureValidationError(f"feature {c!r} must be numeric: {e}") from e
        # Check for infinite values (which can break calculations and are never valid input)
        if np.isinf(X[c].to_numpy()).any():
            raise FeatureValidationError(f"feature {c!r} contains infinite values")

    # --- Validate each CATEGORICAL column ---
    for c in CATEGORICAL:
        # Categorical features (like port names, carrier names) must NOT be missing
        # (unlike numeric features where NaN can be handled by imputation)
        if X[c].isna().any():
            raise FeatureValidationError(f"categorical feature {c!r} has missing values at rows {np.flatnonzero(X[c].isna()).tolist()}")
        # Convert to string type for consistency
        X[c] = X[c].astype(str)

    # --- Generate per-row warnings (non-fatal issues) ---
    # Warnings don't stop the prediction, but they alert the caller to potential issues.
    warnings: list[list[str]] = [[] for _ in range(len(X))]

    # Warn about any categorical values the model hasn't seen during training.
    # The model will still work (it treats unknown categories as all-zeros), but the prediction
    # may be less reliable because the model has no learned information about this category.
    for c in CATEGORICAL:
        unseen = ~X[c].isin(KNOWN_CATEGORIES[c])
        for i in np.flatnonzero(unseen.to_numpy()):
            warnings[i].append(f"{c}={X[c].iloc[i]!r} was not seen in training; encoded as the all-zero baseline category")

    # Warn about unexpected missing values in numeric columns.
    # "Unexpected" means missing in columns where we DON'T normally expect NaN (i.e., not the
    # structural-NaN columns like hub features on direct routes).
    unexpected_nan = [c for c in NUMERIC if not c.startswith(STRUCTURAL_NAN_PREFIXES)]
    nan_mask = X[unexpected_nan].isna().to_numpy()
    for i in np.flatnonzero(nan_mask.any(axis=1)):
        cols = [unexpected_nan[j] for j in np.flatnonzero(nan_mask[i])]
        warnings[i].append(f"missing input values {cols}; the model's training-time missing-value handling is applied")

    return X, warnings


# =============================================================================================
# PREDICTION — Actually running the models to produce results
# =============================================================================================

def _iso(ts) -> str | None:
    """Convert a timestamp to ISO 8601 format string (the standard way to write dates/times).

    Returns None if the timestamp is missing. Handles both timezone-aware and naive timestamps.
    Example output: "2025-01-15T08:00:00+00:00"
    """
    if ts is None or (not isinstance(ts, str) and pd.isna(ts)):
        return None
    return pd.Timestamp(ts).tz_convert("UTC").isoformat() if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).isoformat()


def classify(X: pd.DataFrame, model: str = PRIMARY_CLASSIFIER) -> tuple[np.ndarray, LoadedClassifier]:
    """Run the classification model to get delay-risk probabilities for each shipment.

    This is the core classification step. It takes the prepared features and asks the model:
    "For each shipment, what's the probability it will be On-Time? At-Risk? Critical?"

    Returns:
      - proba: a 2D array of shape (num_shipments, 3) where each row contains three
               probabilities that sum to 1.0 (e.g., [0.7, 0.2, 0.1] means 70% chance
               On-Time, 20% At-Risk, 10% Critical)
      - clf: the loaded classifier object (for metadata like version info)

    "predict_proba" is a scikit-learn method that returns class probabilities instead of
    just a single predicted class. This is more useful because it tells you HOW confident
    the model is, not just what it thinks the answer is.
    """
    # Load the trained classifier (or get it from cache if already loaded)
    clf = load_classifier(model)

    # Ask the model to produce probability estimates for each row
    # predict_proba returns an array like [[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], ...]
    proba = clf.pipeline.predict_proba(X[FEATURES])

    # --- Sanity check the model's output ---
    # Shape must be (number_of_rows, 3) — three probabilities per shipment
    # All values must be finite (not infinity or NaN)
    # Each row must sum to approximately 1.0 (since they're probabilities)
    if proba.shape != (len(X), 3) or not np.isfinite(proba).all() or not np.allclose(proba.sum(1), 1.0, atol=1e-5):
        raise ModelArtifactError(f"{model} returned invalid class probabilities (shape {proba.shape})")

    return proba, clf


def score_anomaly(X: pd.DataFrame, detector: str = ANOMALY_DETECTOR) -> tuple[np.ndarray, np.ndarray, LoadedDetector]:
    """Run the anomaly detector to check if port conditions are unusual for each shipment.

    An "anomaly" is something that looks different from the normal pattern. For example,
    if a port usually processes 100 containers per hour but today it's only processing 10,
    that's anomalous. The Isolation Forest algorithm works by trying to "isolate" each data
    point — points that are easy to isolate (far from the rest) are considered anomalies.

    Returns:
      - s: anomaly scores (higher = more unusual). One score per shipment.
      - s > det.threshold: boolean flags (True = this shipment's port conditions are anomalous)
      - det: the loaded detector object (for metadata)

    The scores are negated (-score_samples or -decision_function) so that HIGHER values
    mean MORE anomalous, which is more intuitive.
    """
    # Load the trained anomaly detector (or get it from cache)
    det = load_detector(detector)

    # columns the detector was fitted on: ANOMALY_FEATURES + origin_port_id (port-relative scaling)
    # Compute anomaly scores — we negate them so higher = more unusual
    # Isolation Forest uses score_samples(); One-Class SVM uses decision_function()
    s = -det.pipeline.score_samples(X) if detector == "isolation_forest" else -det.pipeline.decision_function(X)

    # Make sure all scores are valid numbers (not infinity or NaN)
    if not np.isfinite(s).all():
        raise ModelArtifactError(f"{detector} returned non-finite anomaly scores")

    # Return scores, boolean flags (True if score exceeds threshold), and the detector object
    return s, s > det.threshold, det


def predict(features: pd.DataFrame, classifier: str = PRIMARY_CLASSIFIER, detector: str = ANOMALY_DETECTOR,
            anomaly: bool = True) -> list[dict]:
    """Score feature rows. One result dict per row (see the module docstring for the contract).

    This is the main prediction function that orchestrates everything:
    1. Validates the input features (checks for missing columns, bad data, etc.)
    2. Runs the classifier to get delay-risk probabilities
    3. Optionally runs the anomaly detector to check for unusual port conditions
    4. Packages everything into a clean, standardised output format (one dictionary per shipment)

    The output "contract" is a fixed structure that downstream systems can rely on.
    Each result dictionary contains: shipment_id, timestamps, classification details,
    anomaly details, and any warnings about data quality issues.
    """
    # Step 1: Validate the feature data and collect any warnings
    X, warnings = validate_features(features)

    # Step 2: Run the classification model to get probabilities
    proba, clf = classify(X, classifier)

    # Step 3: Optionally run the anomaly detector
    if anomaly:
        scores, flags, det = score_anomaly(X, detector)

    # Record the current time in UTC — this is when the prediction was made
    now = datetime.now(timezone.utc).isoformat()

    # Extract shipment IDs and feature cutoff timestamps if they exist in the input
    ids = features["shipment_id"].tolist() if "shipment_id" in features.columns else [None] * len(X)
    cutoffs = features["cutoff_ts"].tolist() if "cutoff_ts" in features.columns else [None] * len(X)

    # Step 4: Build one result dictionary per shipment
    out = []
    for i in range(len(X)):
        # Convert the probability array for this row into a dictionary like
        # {"On-Time": 0.7, "At-Risk": 0.2, "Critical": 0.1}
        p = {c: float(proba[i, k]) for k, c in enumerate(config.CLASSES)}

        # Build the result dictionary following the output contract
        res = {
            "contract_version": CONTRACT_VERSION,
            # The shipment this prediction is for (None if not provided)
            "shipment_id": None if ids[i] is None or pd.isna(ids[i]) else str(ids[i]),
            # When this prediction was generated
            "prediction_timestamp_utc": now,
            # The point in time that the features describe (1 hour before departure)
            "feature_cutoff_utc": _iso(cutoffs[i]),
            "classification": {
                "model_name": clf.name, "model_label": clf.label, "model_version": clf.version,
                # The predicted class is whichever class has the HIGHEST probability
                # "argmax" means "index of the maximum value" — so if probabilities are
                # [0.1, 0.7, 0.2], argmax = 1, which maps to "At-Risk"
                "predicted_class": config.CLASSES[int(proba[i].argmax())], "class_probabilities": p,
                "decision_rule": "argmax of the three class probabilities",
                "probabilities_note": "raw model probabilities from a class-balanced model. Not recalibrated: sigmoid and "
                                      "isotonic calibration were not justified on validation data "
                                      "(ML/Evaluation/calibration_study/decision.json). Not an alert threshold.",
            },
            # Anomaly result starts as None; filled in below if anomaly detection is enabled
            "anomaly": None,
            # Any warnings about data quality (missing values, unseen categories, etc.)
            "warnings": list(warnings[i]),
        }

        # If anomaly detection was run, add the anomaly results
        if anomaly:
            res["anomaly"] = {
                "model_name": det.name, "model_label": det.label, "model_version": det.version,
                # The raw anomaly score (higher = more unusual)
                "anomaly_score": float(scores[i]),
                # True if this shipment's port conditions are flagged as anomalous
                "anomaly_flag": bool(flags[i]),
                # The threshold above which a score is considered anomalous
                "threshold": det.threshold,
                # A human-readable explanation of what the score means
                "score_definition": SCORE_DEFINITION[det.name],
                # How the threshold was determined (e.g., "95th percentile of training scores")
                "threshold_rule": det.config.get("threshold_rule"),
            }
        out.append(res)

    return out


# =============================================================================================
# FEATURE CONSTRUCTION — Converting raw shipment info into the 51 numbers the model needs
# =============================================================================================
#
# The model can't understand text like "CNSHA" or "BlueWave Shipping" directly. It needs
# everything converted into numbers. This process is called "feature engineering" or
# "feature construction." For example:
#   - Port utilisation at departure time → a number like 0.85 (85% busy)
#   - Weather at origin → temperature, wind speed, etc. as numbers
#   - Carrier name → converted to multiple 0/1 columns (one-hot encoding)
#   - Historical delay rate on this route → a percentage like 0.23 (23% of past shipments delayed)
#
# The model needs exactly 51 such features, in a specific order, to make its prediction.
# =============================================================================================

@lru_cache(maxsize=1)
def _reference_data():
    """Processed tables + the observed shipments with their outcomes (unobserved outcomes excluded, as in training).

    This function loads all the processed reference data from disk (port telemetry, weather,
    disruptions, holidays, ship records, etc.) and identifies which shipments have known
    outcomes (i.e., we know whether they arrived on time or not). Only shipments with
    observed outcomes are used to compute historical statistics, exactly as was done
    during training.

    The result is cached (maxsize=1) so we only load this data once, even if this
    function is called many times.
    """
    # Load all processed data tables (port data, weather, disruptions, holidays, etc.)
    tables = eda_io.load_all()

    # Get lineage flags that tell us which shipments have actually observed outcomes
    # (some shipments in the dataset have synthetically filled outcomes — we exclude those,
    # just like training did)
    flags = eda_lineage.shipment_fill_flags()

    # Merge the flags onto the shipment records to identify which ones have real outcomes
    ship = tables["shipment_records"].merge(flags[["shipment_id", "outcome_not_observed_in_raw"]], on="shipment_id", how="left",
                                            validate="1:1")

    # Keep only the shipments whose outcomes were actually observed (not synthetic)
    observed = ship[~ship["outcome_not_observed_in_raw"]].drop(columns="outcome_not_observed_in_raw")

    return tables, observed.reset_index(drop=True)


def _validate_shipments(shipments: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """Validate the raw shipment input before building features.

    This function performs extensive checks to make sure the shipment data you're asking
    predictions for is valid and self-consistent. Think of it as a bouncer at the door
    who checks everyone's ID before letting them in. It verifies:

    1. The input is a non-empty DataFrame with all required columns
    2. NO outcome columns are present (you can't peek at the answer!)
    3. Shipment IDs are present and unique
    4. Timestamps are valid and make sense (arrival must be after departure)
    5. Departure times are on the hour (port data is recorded hourly)
    6. Route IDs exist in the port network and match the stated origin/destination
    7. Shipping mode, cargo type, and priority have valid values
    8. Container TEU (a measure of container size) is a positive number
    9. Port telemetry data exists for the cutoff hour (departure - 1 hour)
    """
    # --- Check: Is it a non-empty DataFrame? ---
    if not isinstance(shipments, pd.DataFrame) or len(shipments) == 0:
        raise ShipmentInputError(f"shipments must be a non-empty DataFrame with columns {SHIPMENT_INPUT_COLUMNS}")

    # --- Check: Are all required input columns present? ---
    missing = [c for c in SHIPMENT_INPUT_COLUMNS if c not in shipments.columns]
    if missing:
        raise ShipmentInputError(f"missing shipment columns: {missing}")

    # --- CRITICAL CHECK: Are outcome columns present? They must NOT be! ---
    # This is the "no cheating" rule. If someone includes actual_arrival_ts,
    # actual_delay_hours, or delay_status, it means they're trying to give the model
    # the answer — which defeats the purpose of prediction. You can't know the actual
    # arrival time BEFORE the shipment departs!
    leaked = [c for c in OUTCOME_COLUMNS if c in shipments.columns]
    if leaked:
        raise ShipmentInputError(f"outcome columns {leaked} must not be supplied for prediction")

    # Extract just the required columns and reset the row index
    s = shipments[SHIPMENT_INPUT_COLUMNS].copy().reset_index(drop=True)

    # --- Check: Shipment IDs must be present and unique (no duplicates) ---
    if s["shipment_id"].isna().any() or not s["shipment_id"].astype(str).is_unique:
        raise ShipmentInputError("shipment_id must be present and unique")
    s["shipment_id"] = s["shipment_id"].astype(str)

    # --- Check: Departure and arrival timestamps must be valid dates/times ---
    for c in ("departure_ts", "expected_arrival_ts"):
        try:
            # Parse the timestamps and convert to UTC timezone
            s[c] = pd.to_datetime(s[c], utc=True)
        except (ValueError, TypeError) as e:
            raise ShipmentInputError(f"{c} could not be parsed as timestamps: {e}") from e
        if s[c].isna().any():
            raise ShipmentInputError(f"{c} has missing values")

    # --- Check: Departure must be exactly on the hour (e.g., 08:00, not 08:15) ---
    # This is because port telemetry data is recorded hourly, so we need exact hour alignment
    if ((s["departure_ts"].dt.minute != 0) | (s["departure_ts"].dt.second != 0)).any():
        raise ShipmentInputError("departure_ts must be on the hour (port data is hourly)")

    # --- Check: Expected arrival must be AFTER departure (can't arrive before you leave!) ---
    if (s["expected_arrival_ts"] <= s["departure_ts"]).any():
        raise ShipmentInputError("expected_arrival_ts must be after departure_ts")

    # --- Check: Route IDs must exist in the port network and match origin/destination ---
    # The port network table defines which routes exist and connects which ports
    net = tables["port_network"].set_index("route_id")
    unknown_routes = sorted(set(s["route_id"]) - set(net.index))
    if unknown_routes:
        raise ShipmentInputError(f"unknown route_id {unknown_routes}")

    # Look up each route's origin and destination and verify they match what was provided
    r = net.loc[s["route_id"]]
    if not ((r["origin_port_id"].values == s["origin_port_id"].values) & (r["destination_port_id"].values == s["destination_port_id"].values)).all():
        raise ShipmentInputError("origin_port_id / destination_port_id disagree with route_id")

    # --- Check: Shipping mode and cargo type must be valid known values ---
    for c, allowed in [("shipping_mode", ["air", "sea"]), ("cargo_type", KNOWN_CATEGORIES["cargo_type"])]:
        bad = sorted(set(s[c]) - set(allowed))
        if bad:
            raise ShipmentInputError(f"{c} has unknown values {bad}; expected one of {allowed}")

    # --- Check: Shipment priority must be low, medium, or high ---
    bad = sorted(set(s["shipment_priority"].astype(str)) - {"low", "medium", "high"})
    if bad:
        raise ShipmentInputError(f"shipment_priority has unknown values {bad}; expected low / medium / high")
    # Convert to an ordered categorical type (low < medium < high), which allows
    # the model to understand that "high" priority is greater than "low"
    s["shipment_priority"] = pd.Categorical(s["shipment_priority"].astype(str), categories=["low", "medium", "high"], ordered=True)

    # --- Check: Container TEU must be a positive finite number ---
    # TEU = "Twenty-foot Equivalent Unit" — the standard measure of container size
    try:
        s["container_teu"] = pd.to_numeric(s["container_teu"], errors="raise").astype(float)
    except (ValueError, TypeError) as e:
        raise ShipmentInputError(f"container_teu must be numeric: {e}") from e
    if (~np.isfinite(s["container_teu"]) | (s["container_teu"] <= 0)).any():
        raise ShipmentInputError("container_teu must be a positive finite number")

    # --- Check: Port telemetry data must exist for the cutoff hour ---
    # The "cutoff" is departure_ts minus LAG_HOURS (typically 1 hour). We need port data
    # at that exact hour. If the departure is outside the date range of our processed data,
    # we can't build features for it.
    hours = tables["port_hourly_telemetry"]["ts_hour"]
    cutoff = s["departure_ts"] - pd.Timedelta(hours=config.LAG_HOURS)
    outside = (cutoff < hours.min()) | (cutoff > hours.max())
    if outside.any():
        raise ShipmentInputError(
            f"no port data for the cutoff hour of {s.loc[outside, 'shipment_id'].tolist()}: departure_ts - {config.LAG_HOURS} h must lie "
            f"within {hours.min().isoformat()} .. {hours.max().isoformat()} (coverage of ML/Dataset/Processed; no live feed exists)")

    return s


def build_features_for_shipments(shipments: pd.DataFrame) -> pd.DataFrame:
    """Shipment attributes known before departure -> one row per shipment with the 51 model features.

    Uses the same point-in-time rules as training (cutoff = departure_ts - LAG_HOURS): port state, weather and
    disruptions are read as of the cutoff from ML/Dataset/Processed, and history uses only observed shipments that
    had arrived before the cutoff (at least HIST_MIN_N). The shipments themselves are treated as having no known outcome.
    Returns shipment_id, cutoff_ts, the FEATURES columns and a `feature_warnings` column (list per row).
    """
    # --- Step 1: Load all reference data (port telemetry, weather, etc.) and observed shipments ---
    tables, observed = _reference_data()

    # --- Step 2: Validate the input shipments (all the checks described above) ---
    s_new = _validate_shipments(shipments, tables)

    # --- Step 3: Combine the new shipments with historical observed shipments ---
    # We need the historical shipments to compute features like "what percentage of past
    # shipments on this route were delayed?" But we must NOT let the new shipments' outcomes
    # leak into history (they don't have outcomes — they haven't departed yet!).
    # First, remove any overlap (in case a historical shipment has the same ID)
    base = observed[~observed["shipment_id"].isin(s_new["shipment_id"])]

    # Give the new shipments dummy/empty outcome columns (NaT = "Not a Time", NaN = "Not a Number")
    # This makes their column structure match the historical data so they can be combined.
    # Crucially, these dummy outcomes mean the new shipments will NEVER contribute to
    # historical statistics — they have no known outcome to contribute.
    new_rows = s_new.assign(actual_arrival_ts=pd.Series(pd.NaT, index=s_new.index, dtype=base["actual_arrival_ts"].dtype), actual_delay_hours=np.nan,
                            delay_status=pd.Categorical([None] * len(s_new), categories=config.CLASSES, ordered=True),
                            data_source="inference_request", is_synthetic=False)

    # Stack the historical shipments and new shipments into one big table
    combined = pd.concat([base, new_rows[base.columns]], ignore_index=True)

    # Ensure categorical columns have the right types (needed for proper feature engineering)
    combined["shipment_priority"] = pd.Categorical(combined["shipment_priority"].astype(str), categories=["low", "medium", "high"], ordered=True)
    combined["delay_status"] = pd.Categorical(combined["delay_status"], categories=config.CLASSES, ordered=True)

    # --- Step 4: Build the core shipment features ---
    # This calls the EDA package's feature builder, which computes things like:
    #   - Port utilisation at departure time
    #   - Weather conditions (temperature, wind, etc.)
    #   - Active disruptions at the port
    #   - Holiday information
    # It uses the same "point-in-time" rules as training: features are computed as of
    # the cutoff time (departure minus LAG_HOURS), so no future information leaks in.
    s = eda_features.build_shipment_features(lag_hours=config.LAG_HOURS, tables={**tables, "shipment_records": combined})

    # --- Step 5: Compute historical outcome statistics ---
    # ML-stage steps of features.build_modelling_table, unchanged: history on observed shipments only (min HIST_MIN_N);
    # the requested shipments have no outcome, so they never contribute to any history.
    # First, drop any existing history columns (they'll be recomputed)
    s = s.drop(columns=[c for c in s.columns if c.startswith("hist_")])

    # Compute historical statistics grouped by route, carrier, origin port, and destination port.
    # For example, "what percentage of past shipments by this carrier on this route were delayed?"
    # This requires at least HIST_MIN_N past shipments; if fewer exist, the feature is left as NaN.
    for key in ["route_id", "carrier", "origin_port_id", "destination_port_id"]:
        s = pd.concat([s, eda_features._historical_outcomes(s, key, min_n=HIST_MIN_N)], axis=1)

    # --- Step 6: Extract only the new shipments' rows (not the historical ones) ---
    q = s[s["shipment_id"].isin(s_new["shipment_id"])].copy()

    # --- Step 7: Compute a few final derived features ---
    # the caller supplies the planned ETA, so it is used as is (no schedule-ETA reconstruction is needed)
    # Count how many hub-port holidays fall within the planned voyage duration
    q["hol_h_days_in_voyage_sched"] = _hub_holidays_in_voyage(q, q["expected_arrival_ts"], tables)

    # Set hub_port_id: if the route has a transshipment port, use it; otherwise mark as "DIRECT"
    q["hub_port_id"] = q["transshipment_port_id"].fillna("DIRECT")

    # Convert the ordered priority ("low"=0, "medium"=1, "high"=2) to a numeric rank
    q["priority_rank"] = q["shipment_priority"].cat.codes.astype(float).replace(-1, np.nan)

    # Cap the "days to nearest holiday" feature at 14 days (anything beyond 14 is treated the same)
    q["hol_o_days_to_nearest_cap14"] = q["hol_o_days_to_nearest"].clip(upper=14)

    # Fill missing disruption hours with 0 (no active disruption = 0 hours elapsed)
    q["dis_o_active_hours_elapsed"] = q["dis_o_active_hours_elapsed"].fillna(0)

    # --- Step 8: Reorder rows to match the original input order ---
    q = q.set_index("shipment_id").loc[s_new["shipment_id"]].reset_index()

    # --- Step 9: Generate feature-level warnings ---
    hol_end = tables["public_holidays"]["holiday_date"].max()
    warn = []
    for _, row in q.iterrows():
        w = []
        # Warn if hub port state data is unavailable for transshipment routes
        if row["route_type"] == "transshipment" and pd.isna(row["h_util_mean24"]):
            w.append("hub state unavailable at the cutoff")
        # Warn if there aren't enough historical shipments on this route
        if pd.isna(row["hist_route_id_pct_delayed"]):
            w.append(f"fewer than {HIST_MIN_N} earlier arrivals on this route: route history left missing (as in training)")
        # Warn if the expected arrival is beyond the holiday calendar's coverage
        if not row["hol_calendar_covers_exp_arr"]:
            w.append(f"expected arrival is after the end of the holiday calendar ({hol_end:%Y-%m-%d}); hub holidays during the "
                     "voyage are counted only up to that date")
        warn.append(w)

    # --- Step 10: Build the final output DataFrame ---
    # Include shipment_id, cutoff timestamp, all 51 feature columns, and the warnings
    out = q[["shipment_id", "cutoff_ts"] + FEATURES].copy()
    out["feature_warnings"] = warn
    return out


def predict_shipments(shipments: pd.DataFrame, **kwargs) -> list[dict]:
    """build_features_for_shipments + predict; feature-construction warnings are merged into each result.

    This is a convenience function that does everything in one call:
    1. Takes raw shipment attributes (origin, destination, carrier, etc.)
    2. Builds the 51 model features from those attributes (build_features_for_shipments)
    3. Runs both the classifier and anomaly detector (predict)
    4. Merges any warnings from feature construction into the final results

    This is the simplest way to go from "I have shipment info" to "I have predictions."
    """
    # Step 1: Convert raw shipment attributes into the 51 model features
    feats = build_features_for_shipments(shipments)

    # Step 2: Run the models to produce predictions
    results = predict(feats, **kwargs)

    # Step 3: Merge feature-construction warnings into each result's warning list
    # Feature warnings (e.g., "not enough history on this route") come first,
    # followed by any prediction-level warnings (e.g., "unseen category")
    for r, w in zip(results, feats["feature_warnings"]):
        r["warnings"] = list(w) + r["warnings"]

    return results


# =========================================================================================
# PUBLIC API — The names that other modules can import from this module
# =========================================================================================
# __all__ defines the "public interface" — the list of names that are considered part of
# this module's official API. When someone does `from cargomind_ml.inference import *`,
# only these names are imported.
__all__ = ["CONTRACT_VERSION", "PRIMARY_CLASSIFIER", "ANOMALY_DETECTOR", "RESEARCH_CLASSIFIERS", "RESEARCH_DETECTORS",
           "FeatureValidationError", "ShipmentInputError", "ModelArtifactError", "load_classifier", "load_detector",
           "load_selection", "validate_features", "classify", "score_anomaly", "predict", "build_features_for_shipments",
           "predict_shipments"]
