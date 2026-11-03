"""Model 5 - Random Forest: robust bagged-tree comparison model.

No scaling (trees are invariant to monotone transforms); median imputation with missing-value
indicators (structural NA for hub features on direct routes); balanced class weights.
"""
# ---------------------------------------------------------------------------
# WHAT IS A RANDOM FOREST?
# ---------------------------------------------------------------------------
# A Random Forest is a collection ("ensemble") of many Decision Trees that
# vote together to make a prediction.
#
# A single Decision Tree works like a flowchart of yes/no questions:
#   "Is the weight > 50 kg?"  →  yes  →  "Is the distance > 1000 km?"  →  …
# Each question splits the data, and at the bottom you get a prediction.
#
# One tree alone can be fragile — it might memorise quirks in the training
# data (overfitting).  A Random Forest fixes this by:
#   1. Building MANY trees (e.g., 500), each trained on a slightly different
#      random subset of the data (this is called "bagging").
#   2. At every split, each tree only considers a random subset of the
#      available features.
#   3. For classification, the final answer is whichever category gets the
#      most votes across all trees (majority voting).
#
# Because each tree sees a different slice of the data and features, the
# forest as a whole is much more robust and accurate than any single tree.
# ---------------------------------------------------------------------------

# RandomForestClassifier is scikit-learn's implementation of the algorithm
from sklearn.ensemble import RandomForestClassifier
# Pipeline chains preprocessing and model into one object (see logistic_regression_model.py
# for a detailed explanation)
from sklearn.pipeline import Pipeline

# Project-wide settings (random seed, class names, etc.)
from . import config
# The function that builds data-cleaning / feature-engineering steps
from .preprocessing import make_preprocessor

# Machine-readable identifier for this model
NAME = "random_forest"
# Human-friendly label for reports and charts
LABEL = "Random Forest"
# Which flavour of preprocessing to use — "forest" means: NO scaling (trees
# don't care about the scale of numbers) but still handles missing values and
# encodes categorical features
PREPROCESSING = "forest"


def build_pipeline():
    """Build and return a complete sklearn Pipeline (preprocessing + Random Forest).

    The Pipeline ensures that raw data is first cleaned and encoded by the
    preprocessor, then fed into the Random Forest model — all in one step.
    """
    # Create a Random Forest model with the following settings:
    model = RandomForestClassifier(
        # n_estimators = how many individual decision trees to build.
        # 500 trees → each one votes, and the majority wins.  More trees
        # usually means better accuracy, but also takes longer to train.
        n_estimators=500,
        # n_jobs=1 means use a single CPU core.  Set to -1 to use all cores,
        # but here we keep it at 1 so outer parallelism (e.g., cross-validation
        # folds) doesn't compete for resources.
        n_jobs=1,
        # random_state locks down randomness for reproducibility
        random_state=config.SEED,
        # class_weight="balanced" — if some categories are rare (imbalanced
        # data), the model automatically increases the importance of those
        # rare examples so they aren't drowned out by the common ones.
        class_weight="balanced",
    )
    # Bundle preprocessing and model into a Pipeline
    return Pipeline([("prep", make_preprocessor(PREPROCESSING)), ("model", model)])


def param_distributions():
    """Return the hyperparameter search space for RandomizedSearchCV.

    Hyperparameters are "settings knobs" you turn BEFORE training.  The
    computer tries many combinations and picks whichever scores best on
    held-out validation data.  Below are the knobs we let it explore.
    """
    return {
        # ── n_estimators (number of trees) ──
        # More trees = potentially better accuracy but slower training.
        # We try 300, 500, and 800 trees.
        "model__n_estimators": [300, 500, 800],

        # ── max_depth (maximum depth of each tree) ──
        # "Depth" = how many levels of yes/no questions a single tree can ask.
        #   • None → no limit; the tree grows until every leaf is pure or tiny.
        #   • A small number (e.g., 8) → simpler trees → less overfitting but
        #     may miss complex patterns (underfitting).
        #   • A larger number (e.g., 24) → more complex trees → captures more
        #     patterns but risks overfitting.
        "model__max_depth": [None, 8, 12, 16, 24],

        # ── min_samples_leaf (minimum samples in a leaf) ──
        # A "leaf" is the end node of a tree where a prediction is made.
        # This knob says: "Every leaf must contain at least X training
        # examples."  Bigger values → smoother, more conservative predictions;
        # smaller values → the tree can specialise on tiny groups (risk of
        # overfitting).
        "model__min_samples_leaf": [1, 2, 5, 10, 20, 40],

        # ── max_features (how many features each tree looks at per split) ──
        # At every split in a tree, only a random subset of features is
        # considered.  This adds diversity across trees.
        #   • "sqrt" → use √(total features) — a common default
        #   • 0.2, 0.35, 0.5 → use that fraction of all features
        #   • Fewer features per split → more diversity between trees but each
        #     tree is weaker.  More features → stronger individual trees but
        #     they start to look alike (less benefit from the "forest").
        "model__max_features": ["sqrt", 0.2, 0.35, 0.5],

        # ── class_weight (how to handle imbalanced classes) ──
        # "balanced" reweights once using the full training set.
        # "balanced_subsample" recalculates weights for each tree's bootstrap
        # sample, which can be slightly better when data is very imbalanced.
        "model__class_weight": ["balanced", "balanced_subsample"],
    }
