"""Model 2 - Isolation Forest: unsupervised anomaly detection on operational state.

Fitted on On-Time training shipments only ("normal"). Inputs are the origin port's operating
state at the cutoff (congestion, dwell, anchor queue, weather, disruptions), standardised within
each origin port so that "unusual for this port" is what gets isolated. No global scaling is
needed: isolation splits are scale-invariant per feature.
Anomaly score = -score_samples (higher = more anomalous). Not a delay-class predictor.
"""

# ──────────────────────────────────────────────────────────────────────────────
# WHAT IS ANOMALY DETECTION?
# ──────────────────────────────────────────────────────────────────────────────
# Anomaly detection is the task of finding data points that look "weird" or
# unusual compared to everything else.  Think of it like a security camera that
# flags a person acting differently from the crowd.  Here we train a model
# ONLY on "normal" shipments (those that arrived on time) so the model learns
# what "normal" looks like.  Anything that deviates from that pattern is
# flagged as a potential anomaly — e.g. a port that is suddenly way more
# congested than usual.
#
# HOW DOES ISOLATION FOREST WORK?
# ──────────────────────────────────────────────────────────────────────────────
# Imagine randomly drawing vertical and horizontal lines through your data to
# separate points from one another.  Normal points are clustered together so
# they take MANY random cuts to isolate.  Anomalies sit far from the crowd, so
# they get isolated in very FEW cuts.  The fewer cuts needed to isolate a
# point, the more anomalous it is.  Isolation Forest repeats this process
# across many random "trees" (each tree = one set of random cuts) and averages
# the results.
# ──────────────────────────────────────────────────────────────────────────────

# IsolationForest is the anomaly-detection algorithm from scikit-learn.
from sklearn.ensemble import IsolationForest

# ParameterGrid generates every combination of hyperparameters we want to try.
# (A "hyperparameter" is a setting WE choose before training — like how many
# trees to build — as opposed to something the model learns on its own.)
from sklearn.model_selection import ParameterGrid

# A Pipeline chains preprocessing steps and the model into one object so that
# we can call .fit() and .predict() without worrying about the order of steps.
from sklearn.pipeline import Pipeline

# Project-wide constants (random seed, number of CPU cores to use, etc.)
from . import config

# make_anomaly_preprocessor builds the data-cleaning / feature-engineering
# steps that run BEFORE the data reaches the model.
from .preprocessing import make_anomaly_preprocessor

# ── Module-level constants ────────────────────────────────────────────────────
# NAME is used as a machine-readable identifier (e.g. in filenames / configs).
NAME = "isolation_forest"
# LABEL is the human-friendly display name shown in reports and dashboards.
LABEL = "Isolation Forest"


def build_pipeline(**params):
    """Create a ready-to-train pipeline: preprocessing ➜ Isolation Forest model.

    Parameters
    ----------
    **params : keyword arguments
        Any Isolation Forest hyperparameters (e.g. n_estimators, max_samples)
        that override the defaults.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A two-step pipeline: first preprocess, then run the Isolation Forest.
    """
    # Create the Isolation Forest model.
    #   • random_state = config.SEED  → makes results reproducible every run
    #     (the random splits will be identical if you use the same seed).
    #   • n_jobs = config.N_JOBS      → how many CPU cores to use in parallel
    #     (-1 means "use all available cores").
    #   • **params                    → extra hyperparameters passed in (see
    #     param_grid() below for the ones we search over).
    model = IsolationForest(random_state=config.SEED, n_jobs=config.N_JOBS, **params)

    # Build a two-step Pipeline:
    #   Step 1 ("prep"): per-port standardisation of the operational features.
    #     scale_globally=False means we do NOT apply an additional global
    #     StandardScaler afterwards — Isolation Forest doesn't need it because
    #     its random splits don't depend on the absolute scale of features.
    #   Step 2 ("model"): the Isolation Forest that learns to separate normal
    #     from anomalous data.
    return Pipeline([("prep", make_anomaly_preprocessor(scale_globally=False)), ("model", model)])


def param_grid():
    """Return every combination of hyperparameters we want to search over.

    This is used during model tuning to find the best settings.

    Returns
    -------
    list[dict]
        Each dict is one hyperparameter combination to try.
    """
    # We search over three hyperparameters:
    #
    # • n_estimators (200 or 500):
    #       The number of random trees the forest builds.  More trees = more
    #       stable scores, but also slower.  Think of it as "how many random
    #       experiments we run before averaging."
    #
    # • max_samples (128, 256, or 1024):
    #       How many data points each tree sees.  Using a small subsample makes
    #       anomalies easier to isolate.  128 is the original paper's default.
    #
    # • max_features (0.5, 0.75, or 1.0):
    #       The fraction of features (columns) each tree is allowed to use.
    #       0.5 means "only look at half the columns per tree", which adds
    #       diversity.  1.0 means "use all columns."
    return list(ParameterGrid({
        "n_estimators": [200, 500],
        "max_samples": [128, 256, 1024],
        "max_features": [0.5, 0.75, 1.0],
    }))


def anomaly_score(pipeline, X):
    """Compute an anomaly score for each row in X (higher = more anomalous).

    Internally, scikit-learn's score_samples() returns a value where MORE
    negative = more anomalous.  We NEGATE it (multiply by -1) so that a HIGHER
    number means "more unusual", which is more intuitive for downstream code
    that sorts or thresholds by score.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A trained (fitted) pipeline containing the preprocessor + model.
    X : pandas.DataFrame
        Raw feature rows to score.

    Returns
    -------
    numpy.ndarray
        One score per row — bigger means "weirder."
    """
    # Negate the raw score so that higher = more anomalous.
    return -pipeline.score_samples(X)


def fit_rows(X_normal):
    """Select which rows to actually train on.

    For Isolation Forest there is no subsampling needed — the algorithm is fast
    enough to handle the full dataset because each tree only draws a small
    random subsample internally (controlled by max_samples above).

    Parameters
    ----------
    X_normal : pandas.DataFrame
        All "normal" (on-time) shipment rows available for training.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame, unchanged.
    """
    # Return all normal rows — Isolation Forest is efficient and can train on
    # the entire dataset without running into speed problems.
    return X_normal
