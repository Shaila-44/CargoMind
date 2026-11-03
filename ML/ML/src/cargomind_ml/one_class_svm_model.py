"""Model 3 - One-Class SVM: boundary-based anomaly detection on operational state.

Same inputs as Isolation Forest, plus global standard scaling after the per-port standardisation
(RBF/sigmoid/poly kernels are distance-based). Kernel, nu and gamma are tuned. Training uses a
seeded random subsample of at most OCSVM_MAX_TRAIN normal (On-Time) rows because kernel SVM
training cost grows between quadratically and cubically with n.
Anomaly score = -decision_function (higher = more anomalous). Not a delay-class predictor.
"""

# ──────────────────────────────────────────────────────────────────────────────
# HOW DOES ONE-CLASS SVM WORK?
# ──────────────────────────────────────────────────────────────────────────────
# Think of it as drawing a tight boundary (like a fence) around ALL the normal
# data points in a high-dimensional space.  Anything that falls OUTSIDE that
# boundary is considered an anomaly.
#
# "SVM" stands for Support Vector Machine.  A regular SVM separates two groups
# (e.g. cats vs. dogs).  A ONE-CLASS SVM only sees one group ("normal") and
# learns a boundary that encloses it.  Points far from that boundary are
# unusual.
#
# The model uses a "kernel trick" to bend and stretch the space so the
# boundary can be non-linear (curvy instead of a flat plane).  Different
# kernels (rbf, sigmoid, poly) produce different boundary shapes.
# ──────────────────────────────────────────────────────────────────────────────

# ParameterGrid generates every combination of hyperparameters we want to try.
from sklearn.model_selection import ParameterGrid

# A Pipeline chains preprocessing steps and the model into a single object.
from sklearn.pipeline import Pipeline

# OneClassSVM is the one-class (anomaly detection) variant of scikit-learn's
# Support Vector Machine.
from sklearn.svm import OneClassSVM

# Project-wide constants (random seed, max training rows, etc.)
from . import config

# make_anomaly_preprocessor builds the data-cleaning / feature-engineering
# steps that run BEFORE the data reaches the model.
from .preprocessing import make_anomaly_preprocessor

# ── Module-level constants ────────────────────────────────────────────────────
# NAME is used as a machine-readable identifier (e.g. in filenames / configs).
NAME = "one_class_svm"
# LABEL is the human-friendly display name shown in reports and dashboards.
LABEL = "One-Class SVM"


def build_pipeline(**params):
    """Create a ready-to-train pipeline: preprocessing ➜ One-Class SVM model.

    Parameters
    ----------
    **params : keyword arguments
        Any One-Class SVM hyperparameters (e.g. kernel, nu, gamma) that
        override the defaults.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A two-step pipeline: first preprocess (with global scaling), then run
        the One-Class SVM.
    """
    # Create the One-Class SVM model.
    #   • cache_size=1000  → allocate 1 000 MB of RAM for caching kernel
    #     computations.  Kernel SVMs repeatedly compute distances between pairs
    #     of points; caching avoids redundant work and speeds up training.
    #   • **params         → extra hyperparameters passed in (kernel, nu,
    #     gamma, degree — see param_grid() below).
    model = OneClassSVM(cache_size=1000, **params)

    # Build a two-step Pipeline:
    #   Step 1 ("prep"): per-port standardisation + global StandardScaler.
    #     scale_globally=True adds an extra scaling step after the per-port
    #     normalisation.  This matters because SVM kernels (rbf, sigmoid, poly)
    #     measure DISTANCES between points — if one feature has values in the
    #     thousands and another in the range 0–1, the large feature would
    #     dominate the distance calculation and drown out the small one.
    #     Scaling ensures every feature contributes equally.
    #   Step 2 ("model"): the One-Class SVM that learns the boundary around
    #     normal data.
    return Pipeline([("prep", make_anomaly_preprocessor(scale_globally=True)), ("model", model)])


def param_grid():
    """Return every combination of hyperparameters we want to search over.

    We try three different kernel functions, each with its own set of tuning
    knobs.

    Returns
    -------
    list[dict]
        Each dict is one hyperparameter combination to try.
    """
    # ── RBF kernel combinations ───────────────────────────────────────────────
    # "RBF" = Radial Basis Function.  This is the most commonly used kernel.
    # It creates smooth, circular-ish boundaries in the transformed space.
    #
    # • nu (0.02 … 0.2):
    #       An upper bound on the fraction of training points that the model is
    #       allowed to mark as anomalies.  nu=0.05 means "at most 5 % of
    #       training data may fall outside the boundary."  Higher nu → looser
    #       boundary (more anomalies flagged).
    #
    # • gamma ("scale", 0.01 … 0.3):
    #       Controls how far the influence of a single training point reaches.
    #       Small gamma → far-reaching influence (smoother boundary).
    #       Large gamma → only nearby points matter (more wiggly boundary).
    #       "scale" = automatic choice based on the number of features.
    grid = list(ParameterGrid({
        "kernel": ["rbf"],
        "nu": [0.02, 0.05, 0.1, 0.2],
        "gamma": ["scale", 0.01, 0.03, 0.1, 0.3],
    }))

    # ── Sigmoid kernel combinations ───────────────────────────────────────────
    # The sigmoid kernel produces a boundary inspired by neural-network
    # activation functions.  It can capture different shapes than RBF.
    grid += list(ParameterGrid({
        "kernel": ["sigmoid"],
        "nu": [0.05, 0.1, 0.2],
        "gamma": ["scale", 0.01, 0.03],
    }))

    # ── Polynomial kernel combinations ────────────────────────────────────────
    # The polynomial kernel fits polynomial-shaped boundaries.
    #
    # • degree (2 or 3):
    #       The degree of the polynomial.  degree=2 → quadratic boundary
    #       (curves), degree=3 → cubic boundary (more complex curves).
    grid += list(ParameterGrid({
        "kernel": ["poly"],
        "degree": [2, 3],
        "nu": [0.05, 0.1, 0.2],
        "gamma": ["scale"],
    }))

    return grid


def anomaly_score(pipeline, X):
    """Compute an anomaly score for each row in X (higher = more anomalous).

    Internally, scikit-learn's decision_function() returns a signed distance
    from the learned boundary: positive means "inside the boundary" (normal),
    negative means "outside" (anomalous).  We NEGATE it (multiply by -1) so
    that a HIGHER number means "more unusual", which is more intuitive for
    downstream code that sorts or thresholds by score.

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
    # Negate the raw decision function so that higher = more anomalous.
    return -pipeline.decision_function(X)


def fit_rows(X_normal):
    """Select which rows to actually train on, subsampling if needed.

    WHY DO WE SUBSAMPLE?
    Kernel SVMs are computationally expensive: training time grows roughly
    proportional to n² or n³ (where n is the number of rows).  That means
    doubling the data can make training 4× to 8× slower!  To keep training
    feasible, we cap the number of rows at config.OCSVM_MAX_TRAIN.  If we
    have fewer rows than the cap, we use everything.

    The subsample is drawn with a fixed random seed (config.SEED) so that
    results are reproducible — you'll get the exact same subset every run.

    Parameters
    ----------
    X_normal : pandas.DataFrame
        All "normal" (on-time) shipment rows available for training.

    Returns
    -------
    pandas.DataFrame
        Either the full DataFrame (if small enough) or a random subsample
        capped at OCSVM_MAX_TRAIN rows.
    """
    # If the dataset is already small enough, use all of it — no need to throw
    # away data.
    if len(X_normal) <= config.OCSVM_MAX_TRAIN:
        return X_normal

    # Otherwise, randomly pick OCSVM_MAX_TRAIN rows.  The random_state
    # parameter ensures we get the same subsample every time for
    # reproducibility.
    return X_normal.sample(config.OCSVM_MAX_TRAIN, random_state=config.SEED)
