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

from cargomind_ml import inference as _ml
from cargomind_ml.inference import (ANOMALY_DETECTOR, CONTRACT_VERSION, PRIMARY_CLASSIFIER, SHIPMENT_INPUT_COLUMNS,
                                    FeatureValidationError, ModelArtifactError, ShipmentInputError)

DEFAULT_CLASSIFIER = PRIMARY_CLASSIFIER  # "xgboost" (ML/ML/models/selection.json)
DEFAULT_DETECTOR = ANOMALY_DETECTOR      # "isolation_forest"

__all__ = ["DEFAULT_CLASSIFIER", "DEFAULT_DETECTOR", "CONTRACT_VERSION", "SHIPMENT_INPUT_COLUMNS", "FeatureValidationError",
           "ModelArtifactError", "ShipmentInputError", "predict_shipments", "predict_features", "build_features", "selection"]


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
