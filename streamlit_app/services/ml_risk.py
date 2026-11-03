"""ML service: shipment delay-risk classification (XGBoost) and unusual-conditions detection (Isolation Forest).

A thin adapter over the ML workflow's inference module (`cargomind_ml.inference`, ML/ML/src/cargomind_ml/inference.py).
It adds no model, feature or validation logic of its own.

    from services import ml_risk
    results = ml_risk.predict_shipments(shipments_df)   # shipment attributes -> features -> both models
    results = ml_risk.predict_features(feature_df)      # rows that already hold the 51 model features

Each result keeps the classifier output (`classification`) and the anomaly output (`anomaly`) separate; see the
contract in cargomind_ml.inference. Feature construction reads port, weather, disruption and holiday data from
ML/Dataset/Processed, so it only works for departures inside that data's coverage (no live feed; docs/architecture.md).
"""
from __future__ import annotations

import pandas as pd

from cargomind_eda.config import PROCESSED
from cargomind_ml import inference as _ml
from cargomind_ml.inference import (ANOMALY_DETECTOR, CONTRACT_VERSION, PRIMARY_CLASSIFIER, SHIPMENT_INPUT_COLUMNS,
                                    FeatureValidationError, ModelArtifactError, ShipmentInputError)

DEFAULT_CLASSIFIER = PRIMARY_CLASSIFIER  # "xgboost" (ML/ML/models/selection.json)
DEFAULT_DETECTOR = ANOMALY_DETECTOR      # "isolation_forest"

__all__ = ["DEFAULT_CLASSIFIER", "DEFAULT_DETECTOR", "CONTRACT_VERSION", "SHIPMENT_INPUT_COLUMNS", "FeatureValidationError",
           "ModelArtifactError", "ShipmentInputError", "predict_shipments", "predict_features", "build_features", "selection",
           "supported_shipments", "SUPPORTED_DEPARTURE_START", "SUPPORTED_DEPARTURE_END"]


def predict_shipments(shipments: pd.DataFrame, classifier: str = DEFAULT_CLASSIFIER, detector: str = DEFAULT_DETECTOR) -> list[dict]:
    """Shipment attributes (SHIPMENT_INPUT_COLUMNS, no outcome columns) -> one result dict per shipment."""
    return _ml.predict_shipments(shipments, classifier=classifier, detector=detector)


def predict_features(features: pd.DataFrame, classifier: str = DEFAULT_CLASSIFIER, detector: str = DEFAULT_DETECTOR) -> list[dict]:
    """Pre-built rows holding the 51 model features (optionally shipment_id and cutoff_ts) -> result dicts."""
    return _ml.predict(features, classifier=classifier, detector=detector)


def build_features(shipments: pd.DataFrame) -> pd.DataFrame:
    """Shipment attributes -> model features (for display or caching)."""
    return _ml.build_features_for_shipments(shipments)


def selection() -> dict:
    """The production model selection (ML/ML/models/selection.json)."""
    return _ml.load_selection()


# Departures the feature tables cover (cutoff = departure - 1 h must fall in the coverage; see cargomind_ml.inference).
SUPPORTED_DEPARTURE_START = pd.Timestamp("2025-01-02 00:00", tz="UTC")
SUPPORTED_DEPARTURE_END = pd.Timestamp("2025-02-11 15:00", tz="UTC")


def supported_shipments() -> pd.DataFrame:
    """Synthetic shipments whose departure the ML service can score, as shipment attributes (SHIPMENT_INPUT_COLUMNS)."""
    records = pd.read_csv(PROCESSED / "shipment_records.csv", usecols=SHIPMENT_INPUT_COLUMNS)
    departures = pd.to_datetime(records["departure_ts"], utc=True, format="ISO8601")
    return records[departures.between(SUPPORTED_DEPARTURE_START, SUPPORTED_DEPARTURE_END)].reset_index(drop=True)
