"""Model 6 - LightGBM: histogram-based, leaf-wise gradient boosting.

Objective multiclass (softmax), metric multi_logloss. No scaling or imputation (native missing
handling). Class imbalance: class_weight="balanced", computed by LightGBM on the fitted rows.
num_leaves / min_child_samples are tuned to keep leaf-wise growth from overfitting ~7k rows.
"""
# =============================================================================
# WHAT IS LIGHTGBM?
# =============================================================================
# LightGBM stands for "Light Gradient Boosted Machine", originally created by Microsoft.
# Like XGBoost, it is a gradient boosting framework that trains decision trees
# in a sequence, with each new tree correcting the mistakes of earlier trees.
#
# So how is it different?
# 1. "Leaf-Wise" Tree Growth:
#    Most decision tree algorithms grow level-by-level (depth-wise), splitting all
#    nodes at a given depth evenly.
#    LightGBM grows "leaf-wise" — it finds the SINGLE leaf that will yield the biggest
#    drop in error and splits ONLY that one. This often produces higher accuracy faster.
# 2. Histogram-Based Binning:
#    Instead of sorting millions of continuous numbers, LightGBM groups numbers into
#    discrete buckets ("bins", e.g. 256 bins). This makes it exceptionally fast and
#    memory-friendly.
# 3. Built-in Missing Data & Imbalance Handling:
#    It automatically figures out the best branch for missing values and has native
#    support for balanced class weighting.
# =============================================================================

# --- Third-party library imports ---
# LGBMClassifier is the Scikit-Learn compatible classifier from the LightGBM library
from lightgbm import LGBMClassifier
# Scipy distributions generate values for hyperparameter searching
from scipy.stats import loguniform, randint, uniform
# Pipeline links preprocessing steps and the model together
from sklearn.pipeline import Pipeline

# --- Internal project imports ---
from . import config
from .preprocessing import make_preprocessor

# Machine-readable identifier for LightGBM
NAME = "lightgbm"
# Human-friendly display label
LABEL = "LightGBM"
# Uses the "boosting" preprocessing flavour (no number scaling or imputation needed)
PREPROCESSING = "boosting"


def build_pipeline():
    """Build and return the Scikit-Learn Pipeline for LightGBM.

    The pipeline takes raw features, runs them through categorical encoding,
    and feeds them directly to the LGBMClassifier.
    """
    model = LGBMClassifier(
        # 'multiclass': predicts probabilities across the 3 categories (On-Time, At-Risk, Critical)
        objective="multiclass",

        # 'balanced': automatically calculates weights inversely proportional to class frequencies,
        # ensuring the model pays equal attention to rare delay events.
        class_weight="balanced",

        # 'subsample_freq=1': enables subsampling at every single tree iteration
        subsample_freq=1,

        # 'n_jobs=1': uses a single CPU core here so the outer cross-validation can parallelise cleanly
        n_jobs=1,

        # Fixed random seed to ensure reproducible training runs
        random_state=config.SEED,

        # Suppress verbose training log outputs so terminal logs remain tidy
        verbose=-1,
    )
    # Assemble the preprocessor and the model into one cohesive pipeline
    return Pipeline([("prep", make_preprocessor(PREPROCESSING)), ("model", model)])


def param_distributions():
    """Defines the hyperparameter search space for LightGBM tuning.

    These parameters act as tuning knobs. Randomized search samples combinations
    from this dictionary to find the best settings for classifying cargo shipment risks.
    """
    return {
        # ── n_estimators: Total number of boosting iterations (trees) ──
        # We search between 150 and 800 trees.
        "model__n_estimators": randint(150, 800),

        # ── learning_rate: The shrinkage rate for each tree's update ──
        # Lower values (e.g. 0.02) learn slowly and carefully; higher values learn faster.
        "model__learning_rate": loguniform(0.01, 0.2),

        # ── num_leaves: Maximum tree leaves for base learners ──
        # This is the primary knob controlling tree complexity in LightGBM!
        # Because LightGBM splits leaf-wise, num_leaves controls how flexible the tree is.
        # Smaller values (7, 15) prevent overfitting on moderately sized datasets (~7k rows).
        "model__num_leaves": [7, 15, 31, 63],

        # ── max_depth: Limits the maximum tree depth ──
        # -1 means unlimited depth (relying solely on num_leaves to constrain growth).
        # Integers like 4, 6, 8 explicitly limit how deep any branch can reach.
        "model__max_depth": [-1, 4, 6, 8],

        # ── min_child_samples: Minimum data points required in a single leaf ──
        # Also known as min_data_in_leaf. Setting this higher (e.g. 40, 80, 160) prevents
        # leaves from forming around just 1 or 2 outlier shipments, guarding against overfitting.
        "model__min_child_samples": [10, 20, 40, 80, 160],

        # ── subsample (bagging fraction): Fraction of rows sampled per iteration ──
        # Randomly picks between 60% and 100% of shipments for training each tree.
        "model__subsample": uniform(0.6, 0.4),

        # ── colsample_bytree (feature fraction): Fraction of columns sampled per tree ──
        # Selects between 40% and 100% of feature columns for each tree, encouraging variety.
        "model__colsample_bytree": uniform(0.4, 0.6),

        # ── reg_lambda (L2 regularisation) ──
        # Penalises large leaf weights to ensure stable, smooth predictions.
        "model__reg_lambda": loguniform(1e-3, 20),

        # ── reg_alpha (L1 regularisation) ──
        # Encourages sparsity by driving unimportant feature weights directly to zero.
        "model__reg_alpha": loguniform(1e-3, 5),
    }
