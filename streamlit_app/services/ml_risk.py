"""ML service: shipment delay-risk classification (On-Time / At-Risk / Critical).

A thin adapter over the ML workflow's saved artifacts (ML/ML/models) and feature definitions
(`cargomind_ml.features`). It adds no feature engineering: callers supply rows that already contain the
ML workflow's model features (as produced by `cargomind_ml.features.build_modelling_table`).

Note: the ML workflow does not yet provide a raw-shipment -> features inference function. Until it does,
this service can only score pre-built feature rows (see docs/architecture.md, "Known gaps").
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from cargomind_ml import config as ml_config
from cargomind_ml.features import FEATURES

DEFAULT_CLASSIFIER = "logistic_regression"  # selected by the ML workflow (ML/ML/reports/ML_model_report.md)
CLASSES = list(ml_config.CLASSES)


@lru_cache(maxsize=None)
def load_classifier(name: str = DEFAULT_CLASSIFIER):
    path = ml_config.MODELS / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"ML model {name!r} not found at {path}")
    return joblib.load(path)


def predict_delay_risk(features: pd.DataFrame, model: str = DEFAULT_CLASSIFIER) -> pd.DataFrame:
    """Class probabilities and predicted class for rows of ML model features."""
    missing = [c for c in FEATURES if c not in features.columns]
    if missing:
        raise ValueError(f"missing ML feature columns: {missing}")
    proba = load_classifier(model).predict_proba(features[FEATURES])
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in CLASSES], index=features.index)
    out["predicted_class"] = [CLASSES[i] for i in proba.argmax(1)]
    return out
