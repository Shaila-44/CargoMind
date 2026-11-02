"""Model 6 - LightGBM: histogram-based, leaf-wise gradient boosting.

Objective multiclass (softmax), metric multi_logloss. No scaling or imputation (native missing
handling). Class imbalance: class_weight="balanced", computed by LightGBM on the fitted rows.
num_leaves / min_child_samples are tuned to keep leaf-wise growth from overfitting ~7k rows.
"""
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import make_preprocessor

NAME = "lightgbm"
LABEL = "LightGBM"
PREPROCESSING = "boosting"


def build_pipeline():
    model = LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        subsample_freq=1,
        n_jobs=1,
        random_state=config.SEED,
        verbose=-1,
    )
    return Pipeline([("prep", make_preprocessor(PREPROCESSING)), ("model", model)])


def param_distributions():
    return {
        "model__n_estimators": randint(150, 800),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__num_leaves": [7, 15, 31, 63],
        "model__max_depth": [-1, 4, 6, 8],
        "model__min_child_samples": [10, 20, 40, 80, 160],
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.4, 0.6),
        "model__reg_lambda": loguniform(1e-3, 20),
        "model__reg_alpha": loguniform(1e-3, 5),
    }
