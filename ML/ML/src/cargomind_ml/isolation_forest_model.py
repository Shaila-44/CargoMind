"""Model 2 - Isolation Forest: unsupervised anomaly detection on operational state.

Fitted on On-Time training shipments only ("normal"). Inputs are the origin port's operating
state at the cutoff (congestion, dwell, anchor queue, weather, disruptions), standardised within
each origin port so that "unusual for this port" is what gets isolated. No global scaling is
needed: isolation splits are scale-invariant per feature.
Anomaly score = -score_samples (higher = more anomalous). Not a delay-class predictor.
"""
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import make_anomaly_preprocessor

NAME = "isolation_forest"
LABEL = "Isolation Forest"


def build_pipeline(**params):
    model = IsolationForest(random_state=config.SEED, n_jobs=config.N_JOBS, **params)
    return Pipeline([("prep", make_anomaly_preprocessor(scale_globally=False)), ("model", model)])


def param_grid():
    return list(ParameterGrid({"n_estimators": [200, 500], "max_samples": [128, 256, 1024], "max_features": [0.5, 0.75, 1.0]}))


def anomaly_score(pipeline, X):
    return -pipeline.score_samples(X)


def fit_rows(X_normal):
    return X_normal
