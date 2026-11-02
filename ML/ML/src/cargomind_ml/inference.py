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

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import ANOMALY_FEATURES, CATEGORICAL, FEATURES, HIST_MIN_N, NUMERIC, _hub_holidays_in_voyage

from cargomind_eda import features as eda_features, io as eda_io, lineage as eda_lineage  # noqa: E402

CONTRACT_VERSION = "cargomind-ml-inference/1.0"
SELECTION_PATH = config.MODELS / "selection.json"
PRIMARY_CLASSIFIER = "xgboost"
ANOMALY_DETECTOR = "isolation_forest"
RESEARCH_CLASSIFIERS = ("logistic_regression", "random_forest", "lightgbm")
RESEARCH_DETECTORS = ("one_class_svm",)
LABELS = {"xgboost": "XGBoost", "logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
          "lightgbm": "LightGBM", "isolation_forest": "Isolation Forest", "one_class_svm": "One-Class SVM"}
SCORE_DEFINITION = {"isolation_forest": "-score_samples(X); higher = more unusual origin operating state",
                    "one_class_svm": "-decision_function(X); higher = more unusual origin operating state"}

# Category levels seen in training (ML/ML/results/data/modelling_table.pkl). Unseen levels are one-hot encoded as all
# zeros by the saved pipelines (handle_unknown="ignore"); they are accepted with a warning, never remapped.
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
# Missing values that occur by construction (no hub on direct routes; fewer than HIST_MIN_N prior arrivals). The
# saved pipelines handle NaN in every numeric column (XGBoost natively, Isolation Forest by median imputation), but
# NaN in any other column means an input was not observed and is reported as a warning.
STRUCTURAL_NAN_PREFIXES = ("h_", "wx_h_", "dis_h_", "hol_h_", "hist_")
SHIPMENT_INPUT_COLUMNS = ["shipment_id", "route_id", "origin_port_id", "destination_port_id", "carrier", "shipping_mode",
                          "cargo_type", "container_teu", "shipment_priority", "departure_ts", "expected_arrival_ts"]
OUTCOME_COLUMNS = ["actual_arrival_ts", "actual_delay_hours", "delay_status"]


class FeatureValidationError(ValueError):
    """The feature rows cannot be scored. The message says why."""


class ShipmentInputError(ValueError):
    """Shipment attributes cannot be turned into model features. The message says why."""


class ModelArtifactError(RuntimeError):
    """A model artifact or its configuration is missing, altered or incompatible."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_selection(path: Path = SELECTION_PATH) -> dict:
    if not Path(path).exists():
        raise ModelArtifactError(f"model selection file not found: {path}")
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedClassifier:
    name: str
    label: str
    pipeline: object
    config: dict
    version: str


@dataclass(frozen=True)
class LoadedDetector:
    name: str
    label: str
    pipeline: object
    threshold: float
    config: dict
    version: str


def _version(path: Path) -> str:
    return f"{path.relative_to(config.PROJECT_ROOT).as_posix()}@{_sha256(path)[:12]}"


@lru_cache(maxsize=None)
def load_classifier(name: str = PRIMARY_CLASSIFIER) -> LoadedClassifier:
    """Load a saved supervised pipeline and check it against its config and the current feature schema."""
    if name not in (PRIMARY_CLASSIFIER, *RESEARCH_CLASSIFIERS):
        raise ModelArtifactError(f"unknown classifier {name!r}; available: {[PRIMARY_CLASSIFIER, *RESEARCH_CLASSIFIERS]}")
    path, cfg_path = config.MODELS / f"{name}.joblib", config.MODELS / f"{name}_config.json"
    for p in (path, cfg_path):
        if not p.exists():
            raise ModelArtifactError(f"model artifact missing: {p}")
    cfg = json.loads(cfg_path.read_text())
    pipe = joblib.load(path)
    if not hasattr(pipe, "named_steps") or set(pipe.named_steps) != {"prep", "model"}:
        raise ModelArtifactError(f"{path} is not a (prep, model) Pipeline")
    if cfg["features"] != FEATURES:
        raise ModelArtifactError(f"{cfg_path} feature list differs from cargomind_ml.features.FEATURES")
    if type(pipe.named_steps["model"]).__name__ != cfg["estimator"]:
        raise ModelArtifactError(f"{path} holds {type(pipe.named_steps['model']).__name__}, config says {cfg['estimator']}")
    fitted_cols = [c for _, _, cols in pipe.named_steps["prep"].transformers_ if cols != "drop" for c in cols]
    if sorted(fitted_cols) != sorted(FEATURES):
        raise ModelArtifactError(f"{path} preprocessor was fitted on different columns than FEATURES")
    if list(pipe.classes_) != [0, 1, 2]:
        raise ModelArtifactError(f"{path} classes {list(pipe.classes_)} are not the encoded {config.CLASSES}")
    return LoadedClassifier(name, LABELS[name], pipe, cfg, _version(path))


@lru_cache(maxsize=None)
def load_detector(name: str = ANOMALY_DETECTOR) -> LoadedDetector:
    """Load a saved anomaly bundle {pipeline, threshold} and check it against its config."""
    if name not in (ANOMALY_DETECTOR, *RESEARCH_DETECTORS):
        raise ModelArtifactError(f"unknown anomaly detector {name!r}; available: {[ANOMALY_DETECTOR, *RESEARCH_DETECTORS]}")
    path, cfg_path = config.MODELS / f"{name}.joblib", config.MODELS / f"{name}_config.json"
    for p in (path, cfg_path):
        if not p.exists():
            raise ModelArtifactError(f"model artifact missing: {p}")
    cfg = json.loads(cfg_path.read_text())
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or set(bundle) != {"pipeline", "threshold"}:
        raise ModelArtifactError(f"{path} is not a {{pipeline, threshold}} bundle")
    if cfg["features"] != ANOMALY_FEATURES:
        raise ModelArtifactError(f"{cfg_path} feature list differs from cargomind_ml.features.ANOMALY_FEATURES")
    threshold = float(bundle["threshold"])
    if not np.isfinite(threshold) or not np.isclose(threshold, cfg["threshold"], rtol=0, atol=1e-12):
        raise ModelArtifactError(f"{path} threshold {threshold} does not match its config ({cfg['threshold']})")
    return LoadedDetector(name, LABELS[name], bundle["pipeline"], threshold, cfg, _version(path))


# ---------------------------------------------------------------------------------------------
# Feature validation
# ---------------------------------------------------------------------------------------------
def validate_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[list[str]]]:
    """Check the 51 model features. Returns the feature frame (model column order) and per-row warnings.

    Raises FeatureValidationError for: a non-DataFrame, an empty frame, missing columns, non-numeric values in numeric
    columns, infinite values, or missing (NaN) categorical values. Nothing is imputed or repaired here; the saved
    pipelines handle NaN in numeric columns exactly as during training.
    """
    if not isinstance(features, pd.DataFrame):
        raise FeatureValidationError(f"features must be a pandas DataFrame with columns {FEATURES}; got {type(features).__name__}")
    if len(features) == 0:
        raise FeatureValidationError("features has no rows")
    missing = [c for c in FEATURES if c not in features.columns]
    if missing:
        raise FeatureValidationError(f"missing feature columns: {missing}")
    X = features[FEATURES].copy()
    for c in NUMERIC:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(float)
        try:
            X[c] = pd.to_numeric(X[c], errors="raise").astype(float)
        except (ValueError, TypeError) as e:
            raise FeatureValidationError(f"feature {c!r} must be numeric: {e}") from e
        if np.isinf(X[c].to_numpy()).any():
            raise FeatureValidationError(f"feature {c!r} contains infinite values")
    for c in CATEGORICAL:
        if X[c].isna().any():
            raise FeatureValidationError(f"categorical feature {c!r} has missing values at rows {np.flatnonzero(X[c].isna()).tolist()}")
        X[c] = X[c].astype(str)

    warnings: list[list[str]] = [[] for _ in range(len(X))]
    for c in CATEGORICAL:
        unseen = ~X[c].isin(KNOWN_CATEGORIES[c])
        for i in np.flatnonzero(unseen.to_numpy()):
            warnings[i].append(f"{c}={X[c].iloc[i]!r} was not seen in training; encoded as the all-zero baseline category")
    unexpected_nan = [c for c in NUMERIC if not c.startswith(STRUCTURAL_NAN_PREFIXES)]
    nan_mask = X[unexpected_nan].isna().to_numpy()
    for i in np.flatnonzero(nan_mask.any(axis=1)):
        cols = [unexpected_nan[j] for j in np.flatnonzero(nan_mask[i])]
        warnings[i].append(f"missing input values {cols}; the model's training-time missing-value handling is applied")
    return X, warnings


# ---------------------------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------------------------
def _iso(ts) -> str | None:
    if ts is None or (not isinstance(ts, str) and pd.isna(ts)):
        return None
    return pd.Timestamp(ts).tz_convert("UTC").isoformat() if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).isoformat()


def classify(X: pd.DataFrame, model: str = PRIMARY_CLASSIFIER) -> tuple[np.ndarray, LoadedClassifier]:
    clf = load_classifier(model)
    proba = clf.pipeline.predict_proba(X[FEATURES])
    if proba.shape != (len(X), 3) or not np.isfinite(proba).all() or not np.allclose(proba.sum(1), 1.0, atol=1e-5):
        raise ModelArtifactError(f"{model} returned invalid class probabilities (shape {proba.shape})")
    return proba, clf


def score_anomaly(X: pd.DataFrame, detector: str = ANOMALY_DETECTOR) -> tuple[np.ndarray, np.ndarray, LoadedDetector]:
    det = load_detector(detector)
    # columns the detector was fitted on: ANOMALY_FEATURES + origin_port_id (port-relative scaling)
    s = -det.pipeline.score_samples(X) if detector == "isolation_forest" else -det.pipeline.decision_function(X)
    if not np.isfinite(s).all():
        raise ModelArtifactError(f"{detector} returned non-finite anomaly scores")
    return s, s > det.threshold, det


def predict(features: pd.DataFrame, classifier: str = PRIMARY_CLASSIFIER, detector: str = ANOMALY_DETECTOR,
            anomaly: bool = True) -> list[dict]:
    """Score feature rows. One result dict per row (see the module docstring for the contract)."""
    X, warnings = validate_features(features)
    proba, clf = classify(X, classifier)
    if anomaly:
        scores, flags, det = score_anomaly(X, detector)
    now = datetime.now(timezone.utc).isoformat()
    ids = features["shipment_id"].tolist() if "shipment_id" in features.columns else [None] * len(X)
    cutoffs = features["cutoff_ts"].tolist() if "cutoff_ts" in features.columns else [None] * len(X)
    out = []
    for i in range(len(X)):
        p = {c: float(proba[i, k]) for k, c in enumerate(config.CLASSES)}
        res = {
            "contract_version": CONTRACT_VERSION,
            "shipment_id": None if ids[i] is None or pd.isna(ids[i]) else str(ids[i]),
            "prediction_timestamp_utc": now,
            "feature_cutoff_utc": _iso(cutoffs[i]),
            "classification": {
                "model_name": clf.name, "model_label": clf.label, "model_version": clf.version,
                "predicted_class": config.CLASSES[int(proba[i].argmax())], "class_probabilities": p,
                "decision_rule": "argmax of the three class probabilities",
                "probabilities_note": "raw model probabilities from a class-balanced model. Not recalibrated: sigmoid and "
                                      "isotonic calibration were not justified on validation data "
                                      "(ML/Evaluation/calibration_study/decision.json). Not an alert threshold.",
            },
            "anomaly": None,
            "warnings": list(warnings[i]),
        }
        if anomaly:
            res["anomaly"] = {
                "model_name": det.name, "model_label": det.label, "model_version": det.version,
                "anomaly_score": float(scores[i]), "anomaly_flag": bool(flags[i]), "threshold": det.threshold,
                "score_definition": SCORE_DEFINITION[det.name],
                "threshold_rule": det.config.get("threshold_rule"),
            }
        out.append(res)
    return out


# ---------------------------------------------------------------------------------------------
# Shipment attributes -> model features
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _reference_data():
    """Processed tables + the observed shipments with their outcomes (unobserved outcomes excluded, as in training)."""
    tables = eda_io.load_all()
    flags = eda_lineage.shipment_fill_flags()
    ship = tables["shipment_records"].merge(flags[["shipment_id", "outcome_not_observed_in_raw"]], on="shipment_id", how="left",
                                            validate="1:1")
    observed = ship[~ship["outcome_not_observed_in_raw"]].drop(columns="outcome_not_observed_in_raw")
    return tables, observed.reset_index(drop=True)


def _validate_shipments(shipments: pd.DataFrame, tables: dict) -> pd.DataFrame:
    if not isinstance(shipments, pd.DataFrame) or len(shipments) == 0:
        raise ShipmentInputError(f"shipments must be a non-empty DataFrame with columns {SHIPMENT_INPUT_COLUMNS}")
    missing = [c for c in SHIPMENT_INPUT_COLUMNS if c not in shipments.columns]
    if missing:
        raise ShipmentInputError(f"missing shipment columns: {missing}")
    leaked = [c for c in OUTCOME_COLUMNS if c in shipments.columns]
    if leaked:
        raise ShipmentInputError(f"outcome columns {leaked} must not be supplied for prediction")
    s = shipments[SHIPMENT_INPUT_COLUMNS].copy().reset_index(drop=True)
    if s["shipment_id"].isna().any() or not s["shipment_id"].astype(str).is_unique:
        raise ShipmentInputError("shipment_id must be present and unique")
    s["shipment_id"] = s["shipment_id"].astype(str)
    for c in ("departure_ts", "expected_arrival_ts"):
        try:
            s[c] = pd.to_datetime(s[c], utc=True)
        except (ValueError, TypeError) as e:
            raise ShipmentInputError(f"{c} could not be parsed as timestamps: {e}") from e
        if s[c].isna().any():
            raise ShipmentInputError(f"{c} has missing values")
    if ((s["departure_ts"].dt.minute != 0) | (s["departure_ts"].dt.second != 0)).any():
        raise ShipmentInputError("departure_ts must be on the hour (port data is hourly)")
    if (s["expected_arrival_ts"] <= s["departure_ts"]).any():
        raise ShipmentInputError("expected_arrival_ts must be after departure_ts")
    net = tables["port_network"].set_index("route_id")
    unknown_routes = sorted(set(s["route_id"]) - set(net.index))
    if unknown_routes:
        raise ShipmentInputError(f"unknown route_id {unknown_routes}")
    r = net.loc[s["route_id"]]
    if not ((r["origin_port_id"].values == s["origin_port_id"].values) & (r["destination_port_id"].values == s["destination_port_id"].values)).all():
        raise ShipmentInputError("origin_port_id / destination_port_id disagree with route_id")
    for c, allowed in [("shipping_mode", ["air", "sea"]), ("cargo_type", KNOWN_CATEGORIES["cargo_type"])]:
        bad = sorted(set(s[c]) - set(allowed))
        if bad:
            raise ShipmentInputError(f"{c} has unknown values {bad}; expected one of {allowed}")
    bad = sorted(set(s["shipment_priority"].astype(str)) - {"low", "medium", "high"})
    if bad:
        raise ShipmentInputError(f"shipment_priority has unknown values {bad}; expected low / medium / high")
    s["shipment_priority"] = pd.Categorical(s["shipment_priority"].astype(str), categories=["low", "medium", "high"], ordered=True)
    try:
        s["container_teu"] = pd.to_numeric(s["container_teu"], errors="raise").astype(float)
    except (ValueError, TypeError) as e:
        raise ShipmentInputError(f"container_teu must be numeric: {e}") from e
    if (~np.isfinite(s["container_teu"]) | (s["container_teu"] <= 0)).any():
        raise ShipmentInputError("container_teu must be a positive finite number")
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
    tables, observed = _reference_data()
    s_new = _validate_shipments(shipments, tables)
    base = observed[~observed["shipment_id"].isin(s_new["shipment_id"])]
    new_rows = s_new.assign(actual_arrival_ts=pd.Series(pd.NaT, index=s_new.index, dtype=base["actual_arrival_ts"].dtype), actual_delay_hours=np.nan,
                            delay_status=pd.Categorical([None] * len(s_new), categories=config.CLASSES, ordered=True),
                            data_source="inference_request", is_synthetic=False)
    combined = pd.concat([base, new_rows[base.columns]], ignore_index=True)
    combined["shipment_priority"] = pd.Categorical(combined["shipment_priority"].astype(str), categories=["low", "medium", "high"], ordered=True)
    combined["delay_status"] = pd.Categorical(combined["delay_status"], categories=config.CLASSES, ordered=True)

    s = eda_features.build_shipment_features(lag_hours=config.LAG_HOURS, tables={**tables, "shipment_records": combined})
    # ML-stage steps of features.build_modelling_table, unchanged: history on observed shipments only (min HIST_MIN_N);
    # the requested shipments have no outcome, so they never contribute to any history.
    s = s.drop(columns=[c for c in s.columns if c.startswith("hist_")])
    for key in ["route_id", "carrier", "origin_port_id", "destination_port_id"]:
        s = pd.concat([s, eda_features._historical_outcomes(s, key, min_n=HIST_MIN_N)], axis=1)
    q = s[s["shipment_id"].isin(s_new["shipment_id"])].copy()
    # the caller supplies the planned ETA, so it is used as is (no schedule-ETA reconstruction is needed)
    q["hol_h_days_in_voyage_sched"] = _hub_holidays_in_voyage(q, q["expected_arrival_ts"], tables)
    q["hub_port_id"] = q["transshipment_port_id"].fillna("DIRECT")
    q["priority_rank"] = q["shipment_priority"].cat.codes.astype(float).replace(-1, np.nan)
    q["hol_o_days_to_nearest_cap14"] = q["hol_o_days_to_nearest"].clip(upper=14)
    q["dis_o_active_hours_elapsed"] = q["dis_o_active_hours_elapsed"].fillna(0)

    q = q.set_index("shipment_id").loc[s_new["shipment_id"]].reset_index()
    hol_end = tables["public_holidays"]["holiday_date"].max()
    warn = []
    for _, row in q.iterrows():
        w = []
        if row["route_type"] == "transshipment" and pd.isna(row["h_util_mean24"]):
            w.append("hub state unavailable at the cutoff")
        if pd.isna(row["hist_route_id_pct_delayed"]):
            w.append(f"fewer than {HIST_MIN_N} earlier arrivals on this route: route history left missing (as in training)")
        if not row["hol_calendar_covers_exp_arr"]:
            w.append(f"expected arrival is after the end of the holiday calendar ({hol_end:%Y-%m-%d}); hub holidays during the "
                     "voyage are counted only up to that date")
        warn.append(w)
    out = q[["shipment_id", "cutoff_ts"] + FEATURES].copy()
    out["feature_warnings"] = warn
    return out


def predict_shipments(shipments: pd.DataFrame, **kwargs) -> list[dict]:
    """build_features_for_shipments + predict; feature-construction warnings are merged into each result."""
    feats = build_features_for_shipments(shipments)
    results = predict(feats, **kwargs)
    for r, w in zip(results, feats["feature_warnings"]):
        r["warnings"] = list(w) + r["warnings"]
    return results


__all__ = ["CONTRACT_VERSION", "PRIMARY_CLASSIFIER", "ANOMALY_DETECTOR", "RESEARCH_CLASSIFIERS", "RESEARCH_DETECTORS",
           "FeatureValidationError", "ShipmentInputError", "ModelArtifactError", "load_classifier", "load_detector",
           "load_selection", "validate_features", "classify", "score_anomaly", "predict", "build_features_for_shipments",
           "predict_shipments"]
