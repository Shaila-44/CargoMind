"""Model 4 - XGBoost: gradient-boosted trees, primary candidate.

Objective multi:softprob (calibrated class probabilities), eval metric mlogloss. No scaling or
imputation: XGBoost learns a default direction for missing values (structural NA on direct routes).
Class imbalance: balanced sample weights, recomputed on whichever rows are being fitted (so each
CV fold uses weights from its own training rows only).
"""
# =============================================================================
# WHAT IS GRADIENT BOOSTING & XGBOOST?
# =============================================================================
# Imagine you are practicing archery with a coach:
#   1. You take your first shot (Tree 1). You miss by 10 inches to the left.
#   2. The coach doesn't say "start over from scratch." Instead, they say:
#      "Adjust your aim 10 inches to the right" (Tree 2 learns from the error).
#   3. You shoot again, and now you are only 2 inches too high.
#   4. Another adjustment is made (Tree 3).
#
# Unlike Random Forest (where many trees vote independently at the same time),
# Gradient Boosting builds trees ONE AFTER ANOTHER in a sequence. Each new tree
# specifically focuses on fixing the mistakes (residuals) made by all prior trees.
#
# "XGBoost" stands for "Extreme Gradient Boosting". It is one of the most
# successful algorithms in competitive machine learning because:
#   - It is blazingly fast.
#   - It has built-in math to prevent overfitting (regularisation).
#   - It handles missing data naturally without needing manual imputation.
# =============================================================================

# --- Third-party library imports ---
# Scipy distributions generate random numbers within chosen ranges during hyperparameter tuning:
#   loguniform: picks numbers on a logarithmic scale (great for learning rates like 0.01 vs 0.1)
#   randint: picks random whole numbers (integers), e.g. tree depth 3, 4, 5
#   uniform: picks numbers evenly between a min and max
from scipy.stats import loguniform, randint, uniform

# Pipeline chains preprocessing steps and the model into a single unified object
from sklearn.pipeline import Pipeline

# XGBClassifier is the official Python class for classification using XGBoost
from xgboost import XGBClassifier

# --- Internal project imports ---
from . import config
# make_preprocessor builds the feature preparation pipeline
# balanced_sample_weight calculates higher weights for rare delay classes
from .preprocessing import balanced_sample_weight, make_preprocessor

# Machine-readable identifier for this model
NAME = "xgboost"
# Human-friendly label for charts and comparison tables
LABEL = "XGBoost"
# "boosting" preprocessing flavour:
# Trees do not require number scaling (like standard deviations).
# Furthermore, XGBoost handles missing values internally, so we don't impute missing numbers!
PREPROCESSING = "boosting"


class BalancedXGBClassifier(XGBClassifier):
    """Custom XGBClassifier that automatically balances classes during training.

    WHY DO WE NEED THIS?
    In cargo logistics, most shipments arrive "On-Time" (e.g. 70-80%), while severe
    delays ("Critical") might only be 5-10% of shipments.
    If a model simply guesses "On-Time" every time, it gets an 80% score while being
    completely useless at catching disasters!

    To fix this, we give "sample weights" to each row. A rare Critical delay gets a high
    weight (e.g. 5.0), while a common On-Time shipment gets a low weight (e.g. 0.5).
    This forces the model to pay serious attention to rare delay events.

    By putting this inside .fit(), the weights are automatically recalculated for whatever
    training slice or cross-validation fold is being trained, preventing data leakage.
    """

    def fit(self, X, y, **kwargs):
        # If the caller didn't supply custom sample weights, calculate balanced weights
        if kwargs.get("sample_weight") is None:
            kwargs["sample_weight"] = balanced_sample_weight(y)
        # Call the standard XGBoost fit method with our weights included
        return super().fit(X, y, **kwargs)


def build_pipeline():
    """Build and return the complete Scikit-Learn Pipeline for XGBoost.

    Combines:
      Step 1 ('prep'): Preprocesses inputs (one-hot encoding categorical variables, etc.)
      Step 2 ('model'): The BalancedXGBClassifier model ready to learn.
    """
    model = BalancedXGBClassifier(
        # 'multi:softprob' means predicting probabilities for each class (summing to 100%)
        # e.g., [On-Time: 70%, At-Risk: 20%, Critical: 10%]
        objective="multi:softprob",

        # 'mlogloss' (multiclass logarithmic loss) grades how confident and accurate probabilities are
        eval_metric="mlogloss",

        # 'hist' uses fast histogram binning to speed up tree construction dramatically
        tree_method="hist",

        # Single CPU thread for inner training, so outer cross-validation can parallelise cleanly
        n_jobs=1,

        # Fixed random seed so training produces exact reproducible results every run
        random_state=config.SEED,
    )
    # Bundle preprocessing and model together
    return Pipeline([("prep", make_preprocessor(PREPROCESSING)), ("model", model)])


def param_distributions():
    """Defines the hyperparameter search space for tuning.

    Hyperparameters are the control knobs of the algorithm. Randomized search will try
    different combinations of these settings to discover which combination achieves the
    highest accuracy and recall on held-out validation folds.
    """
    return {
        # ── n_estimators: Total number of sequential trees to grow ──
        # More trees allow the model to learn subtle patterns, but too many can lead to overfitting.
        # We test anywhere from 150 to 800 trees.
        "model__n_estimators": randint(150, 800),

        # ── learning_rate (shrinkage / step size) ──
        # Controls how much each new tree contributes to the final prediction.
        # A smaller learning rate (e.g. 0.02) means each tree makes conservative adjustments,
        # requiring more trees but usually resulting in better generalisation.
        "model__learning_rate": loguniform(0.01, 0.2),

        # ── max_depth: How deep each tree can grow ──
        # A depth of 2 or 3 creates very simple trees ("stumps") capturing simple interactions.
        # A depth of 6 to 8 creates detailed trees capturing complex multi-variable interactions.
        "model__max_depth": randint(2, 8),

        # ── min_child_weight: Minimum weight required in a leaf node ──
        # Higher values prevent trees from creating leaves that only match a handful of noisy data points.
        # Acts like a safety brake against memorising flukes in the data.
        "model__min_child_weight": [1, 3, 5, 10, 20],

        # ── subsample: Fraction of rows sampled to train each tree ──
        # e.g. 0.8 means each tree only sees 80% of training shipments randomly selected.
        # Randomising data rows adds diversity and reduces overfitting.
        "model__subsample": uniform(0.6, 0.4),

        # ── colsample_bytree: Fraction of features sampled for each tree ──
        # e.g. 0.7 means each tree only gets to look at 70% of available columns.
        # This prevents one dominant feature (like distance) from overshadowing everything else.
        "model__colsample_bytree": uniform(0.4, 0.6),

        # ── reg_lambda: L2 regularisation penalty on leaf weights ──
        # Discourages individual trees from giving overly extreme or aggressive predictions.
        "model__reg_lambda": loguniform(0.1, 20),

        # ── gamma: Minimum loss reduction required to make another split ──
        # If splitting a node doesn't improve the model by at least gamma, the tree stops splitting.
        # Higher values lead to more conservative, simpler trees.
        "model__gamma": [0, 0.1, 0.5, 1.0],
    }
