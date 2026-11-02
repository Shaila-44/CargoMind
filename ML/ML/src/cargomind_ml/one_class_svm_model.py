"""Model 3 - One-Class SVM: boundary-based anomaly detection on operational state.

Same inputs as Isolation Forest, plus global standard scaling after the per-port standardisation
(RBF/sigmoid/poly kernels are distance-based). Kernel, nu and gamma are tuned. Training uses a
seeded random subsample of at most OCSVM_MAX_TRAIN normal (On-Time) rows because kernel SVM
training cost grows between quadratically and cubically with n.
Anomaly score = -decision_function (higher = more anomalous). Not a delay-class predictor.
"""
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

from . import config
from .preprocessing import make_anomaly_preprocessor

NAME = "one_class_svm"
LABEL = "One-Class SVM"


def build_pipeline(**params):
    model = OneClassSVM(cache_size=1000, **params)
    return Pipeline([("prep", make_anomaly_preprocessor(scale_globally=True)), ("model", model)])


def param_grid():
    grid = list(ParameterGrid({"kernel": ["rbf"], "nu": [0.02, 0.05, 0.1, 0.2], "gamma": ["scale", 0.01, 0.03, 0.1, 0.3]}))
    grid += list(ParameterGrid({"kernel": ["sigmoid"], "nu": [0.05, 0.1, 0.2], "gamma": ["scale", 0.01, 0.03]}))
    grid += list(ParameterGrid({"kernel": ["poly"], "degree": [2, 3], "nu": [0.05, 0.1, 0.2], "gamma": ["scale"]}))
    return grid


def anomaly_score(pipeline, X):
    return -pipeline.decision_function(X)


def fit_rows(X_normal):
    if len(X_normal) <= config.OCSVM_MAX_TRAIN:
        return X_normal
    return X_normal.sample(config.OCSVM_MAX_TRAIN, random_state=config.SEED)
