"""Model 1 - multinomial Logistic Regression: the interpretable baseline.

Scaled numeric inputs (coefficients are per standard deviation), one-hot categoricals, balanced
class weights, L1 or L2 regularisation tuned with C. Interpretation via coefficients.
"""
# ---------------------------------------------------------------------------
# WHAT IS LOGISTIC REGRESSION?
# ---------------------------------------------------------------------------
# Logistic Regression is one of the simplest ML models for classification
# (sorting things into categories). Despite the word "regression", it does NOT
# predict a number — it predicts a *category*.
#
# Imagine drawing a straight line (or a flat plane) through your data to
# separate groups. On one side of the line you predict "category A", on the
# other side "category B". When there are 3+ categories (like here), the model
# draws *multiple* such boundaries and picks the most likely category — this
# variant is called "multinomial" or "softmax" logistic regression.
#
# Why use it? It's fast, easy to understand, and the coefficients tell you
# exactly which features push the prediction toward which category. That makes
# it a great *baseline* — a simple model you compare fancier models against.
# ---------------------------------------------------------------------------

# NumPy is the standard library for fast math on arrays of numbers
import numpy as np
# Pandas gives us DataFrames — spreadsheet-like tables in Python
import pandas as pd
# loguniform produces random numbers spread evenly on a *logarithmic* scale
# (good for values that can range from 0.001 to 10 — you want to try many
# small values AND a few big ones, not just big ones)
from scipy.stats import loguniform
# LogisticRegression is scikit-learn's implementation of the model described above
from sklearn.linear_model import LogisticRegression
# A Pipeline chains pre-processing steps and the model together so you can
# treat the whole sequence as one object — no risk of accidentally applying
# different transformations to training vs. test data
from sklearn.pipeline import Pipeline

# Import project-wide settings (random seed, class names, etc.)
from . import config
# Import the function that builds the data-cleaning / feature-engineering steps
from .preprocessing import make_preprocessor

# A short machine-readable name for this model (used in file names, logs, etc.)
NAME = "logistic_regression"
# A human-friendly label for reports and charts
LABEL = "Logistic Regression"
# Which flavour of preprocessing to use — "linear" means: scale numbers to
# mean=0/std=1 (StandardScaler) and one-hot-encode categories, because
# Logistic Regression needs all features on a comparable numeric scale
PREPROCESSING = "linear"


def build_pipeline():
    """Build and return a complete sklearn Pipeline (preprocessing + model).

    A Pipeline bundles the data-preparation step ("prep") and the model step
    ("model") into a single object.  When you call pipeline.fit(X, y), it
    first transforms the raw data through "prep", then trains the model on the
    transformed result.  This guarantees that train and test data are always
    processed identically.
    """
    # Create the Logistic Regression model with specific settings:
    model = LogisticRegression(
        # "saga" is the solver (optimisation algorithm). It supports both L1
        # and L2 penalties and works well with multi-class problems.
        solver="saga",
        # max_iter = maximum number of times the algorithm loops through the
        # data trying to improve. 5000 is generous — we want it to converge
        # (find a good answer) even on tricky data.
        max_iter=5000,
        # tol = tolerance. The algorithm stops when improvement between
        # iterations is smaller than this tiny number (0.001). Smaller = more
        # precise but slower.
        tol=1e-3,
        # class_weight="balanced" means: if one category has far fewer
        # examples than another (imbalanced data), automatically give those
        # rare examples MORE importance so the model doesn't just ignore them.
        # Without this, a model might predict the common category almost every
        # time and still look accurate overall.
        class_weight="balanced",
        # random_state locks down randomness so results are reproducible —
        # run the same code twice, get the same answer.
        random_state=config.SEED,
    )  # 3 classes -> multinomial softmax
    # Chain preprocessing and model into one Pipeline and return it
    return Pipeline([("prep", make_preprocessor(PREPROCESSING)), ("model", model)])


def param_distributions():
    """Return the hyperparameter search space for RandomizedSearchCV.

    Hyperparameters are "settings knobs" that you, the human, choose BEFORE
    training.  The computer can't learn them from the data on its own, so we
    try many different combinations and keep whichever works best
    (this process is called *hyperparameter tuning*).

    Returns a dictionary mapping each knob name to the values we want to try.
    """
    return {
        # ── C (inverse regularisation strength) ──
        # "C" controls how much the model is allowed to bend itself to fit the
        # training data.
        #   • Small C → heavy regularisation → simpler model (may underfit)
        #   • Large C → light regularisation → more complex model (may overfit)
        # We sample C from a log-uniform distribution between 0.001 and 10,
        # meaning the computer tries values like 0.002, 0.05, 0.3, 2, 8, etc.
        "model__C": loguniform(1e-3, 10),

        # ── penalty (type of regularisation) ──
        # Regularisation = a technique that penalises overly complex models to
        # prevent overfitting (memorising the training data instead of learning
        # general patterns).
        #   • "l1" (Lasso): can shrink unimportant feature weights all the way
        #     to zero, effectively doing automatic feature selection.
        #   • "l2" (Ridge): shrinks weights toward zero but rarely makes them
        #     exactly zero. Good when all features contribute a little.
        "model__penalty": ["l1", "l2"],
    }


def coefficients(pipeline, classes=config.CLASSES):
    """Coefficient per (encoded feature, class). Numeric features are standardised, so
    magnitudes are comparable across numeric features (log-odds per 1 SD)."""
    # Get the list of feature names that came out of the preprocessor
    # (e.g., "weight_kg", "origin_US", "origin_EU", …)
    names = pipeline.named_steps["prep"].get_feature_names_out()

    # Retrieve the learned coefficients (weights) from the trained model.
    # Each coefficient tells you: "for a 1-standard-deviation increase in this
    # feature, how much more (or less) likely is each class?"
    # Shape: (number_of_classes, number_of_features)
    coef = pipeline.named_steps["model"].coef_

    # Build a nice table: rows = features, columns = class names
    df = pd.DataFrame(coef.T, index=names, columns=classes)

    # Add a helper column: the largest absolute coefficient across classes.
    # This lets us see at a glance which features matter most overall.
    df["max_abs"] = np.abs(coef).max(axis=0)

    # Sort so the most influential features appear first
    return df.sort_values("max_abs", ascending=False)
