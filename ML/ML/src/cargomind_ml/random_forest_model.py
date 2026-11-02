"""Model 5 - Random Forest: robust bagged-tree comparison model.

No scaling (trees are invariant to monotone transforms); median imputation with missing-value
indicators (structural NA for hub features on direct routes); balanced class weights.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import make_preprocessor

NAME = "random_forest"
LABEL = "Random Forest"
PREPROCESSING = "forest"


def build_pipeline():
    model = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=config.SEED, class_weight="balanced")
    return Pipeline([("prep", make_preprocessor(PREPROCESSING)), ("model", model)])


def param_distributions():
    return {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [None, 8, 12, 16, 24],
        "model__min_samples_leaf": [1, 2, 5, 10, 20, 40],
        "model__max_features": ["sqrt", 0.2, 0.35, 0.5],
        "model__class_weight": ["balanced", "balanced_subsample"],
    }
