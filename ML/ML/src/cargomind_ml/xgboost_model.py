"""Model 4 - XGBoost: gradient-boosted trees, primary candidate.

Objective multi:softprob (calibrated class probabilities), eval metric mlogloss. No scaling or
imputation: XGBoost learns a default direction for missing values (structural NA on direct routes).
Class imbalance: balanced sample weights, recomputed on whichever rows are being fitted (so each
CV fold uses weights from its own training rows only).
"""
from scipy.stats import loguniform, randint, uniform
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config
from .preprocessing import balanced_sample_weight, make_preprocessor

NAME = "xgboost"
LABEL = "XGBoost"
PREPROCESSING = "boosting"


class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier that applies inverse-class-frequency sample weights at fit time."""

    def fit(self, X, y, **kwargs):
        if kwargs.get("sample_weight") is None:
            kwargs["sample_weight"] = balanced_sample_weight(y)
        return super().fit(X, y, **kwargs)


def build_pipeline():
    model = BalancedXGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=1,
        random_state=config.SEED,
    )
    return Pipeline([("prep", make_preprocessor(PREPROCESSING)), ("model", model)])


def param_distributions():
    return {
        "model__n_estimators": randint(150, 800),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": randint(2, 8),
        "model__min_child_weight": [1, 3, 5, 10, 20],
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.4, 0.6),
        "model__reg_lambda": loguniform(0.1, 20),
        "model__gamma": [0, 0.1, 0.5, 1.0],
    }
