"""Model 1 - multinomial Logistic Regression: the interpretable baseline.

Scaled numeric inputs (coefficients are per standard deviation), one-hot categoricals, balanced
class weights, L1 or L2 regularisation tuned with C. Interpretation via coefficients.
"""
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import make_preprocessor

NAME = "logistic_regression"
LABEL = "Logistic Regression"
PREPROCESSING = "linear"


def build_pipeline():
    model = LogisticRegression(
        solver="saga", max_iter=5000, tol=1e-3, class_weight="balanced", random_state=config.SEED
    )  # 3 classes -> multinomial softmax
    return Pipeline([("prep", make_preprocessor(PREPROCESSING)), ("model", model)])


def param_distributions():
    return {"model__C": loguniform(1e-3, 10), "model__penalty": ["l1", "l2"]}


def coefficients(pipeline, classes=config.CLASSES):
    """Coefficient per (encoded feature, class). Numeric features are standardised, so
    magnitudes are comparable across numeric features (log-odds per 1 SD)."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coef = pipeline.named_steps["model"].coef_
    df = pd.DataFrame(coef.T, index=names, columns=classes)
    df["max_abs"] = np.abs(coef).max(axis=0)
    return df.sort_values("max_abs", ascending=False)
