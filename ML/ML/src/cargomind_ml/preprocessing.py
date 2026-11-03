"""Chronological splits and per-algorithm preprocessing pipelines.

Every transformer is fitted inside an sklearn Pipeline, so it only ever sees the rows it is
fitted on (the training period, or the training part of a CV fold). Preprocessing differs by
algorithm family on purpose:

| family     | numeric                                  | categorical                 |
|------------|------------------------------------------|-----------------------------| 
| linear     | median impute + missing flags + scaling  | one-hot                     |
| forest     | median impute + missing flags (no scale) | one-hot                     |
| boosting   | passthrough, NaN handled natively        | one-hot                     |
| anomaly    | median impute, per-origin-port z-score   | (none: operational state)   |
| anomaly_svm| as anomaly + global standard scaling     | (none)                      |
"""

# ── Standard library and third-party imports ──────────────────────────────────
# numpy provides fast math on arrays of numbers (like spreadsheets of numbers)
import numpy as np

# pandas gives us DataFrames, which are like Excel spreadsheets in Python
import pandas as pd

# BaseEstimator and TransformerMixin let us create custom data-transformation
# objects that plug into scikit-learn's pipeline system (like building blocks
# that snap together)
from sklearn.base import BaseEstimator, TransformerMixin

# ColumnTransformer lets us apply *different* transformations to different
# columns at the same time (e.g. scale numbers but one-hot encode categories)
from sklearn.compose import ColumnTransformer

# SimpleImputer fills in missing values (NaN / blank cells) with a strategy
# we choose, such as the median (the middle value)
# "Imputation" = filling in blanks with a reasonable guess so models can use the data
from sklearn.impute import SimpleImputer

# Pipeline chains multiple data-processing steps together in order, so data
# flows through step 1, then step 2, etc. — like an assembly line
from sklearn.pipeline import Pipeline

# OneHotEncoder converts text categories (e.g. "NLRTM", "CNSHA") into columns
# of 0s and 1s so math-based models can understand them.
# "One-hot encoding" means: for each category, create a new yes/no column.
# For example, port="NLRTM" becomes: is_NLRTM=1, is_CNSHA=0, is_USNYC=0
#
# StandardScaler rescales numbers so they all have a similar range.
# "Standard scaling" means: subtract the average and divide by the spread
# (standard deviation). This way every feature is centred around 0 with a
# spread of 1 — important for models like logistic regression that are
# sensitive to how big the numbers are.
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# compute_sample_weight creates weights that tell the model to pay MORE
# attention to rare classes and LESS attention to common classes, so it
# doesn't just learn to always predict the majority class.
from sklearn.utils.class_weight import compute_sample_weight

# Project-specific imports: config holds settings (like how much data to
# reserve for testing), and features defines which columns are numeric,
# categorical, or used for anomaly detection.
from . import config
from .features import ANOMALY_FEATURES, ANOMALY_PORT_RELATIVE, CATEGORICAL, NUMERIC


# =============================================================================================
# SPLITS – How we divide data into training, testing, and validation sets
# =============================================================================================
#
# WHY SPLIT BY TIME (chronological split)?
# -----------------------------------------
# In cargo/shipping data, events happen in order over time. If we randomly
# shuffled and split the data, the model could "cheat" — it might see a
# future shipment during training and use that knowledge to predict an
# earlier one. This is called "data leakage" and makes the model look
# great in testing but fail in real life.
#
# A CHRONOLOGICAL SPLIT means: train on OLDER data, test on NEWER data.
# This simulates real-world usage where the model must predict the future
# based only on what it learned from the past.
#
# WHAT IS THE PURGE GAP?
# ----------------------
# Ports have ongoing conditions (congestion, weather events) that last
# hours or days. If a training shipment departs at 11:55 PM and a test
# shipment departs at 12:05 AM, they share the same port conditions —
# the model could "leak" information across the split boundary.
#
# The PURGE GAP removes a buffer of time between train and test (e.g. 48
# hours). Shipments that fall in this gap are discarded entirely, so the
# training set and test set don't share any overlapping port episodes.
# =============================================================================================

def chronological_split(table, test_fraction=config.TEST_FRACTION, gap_hours=config.PURGE_GAP_HOURS):
    """Assign each shipment to train / purge / test by departure time.

    test  : departures at or after T, where T is the departure time of the shipment at the
            (1 - test_fraction) quantile of the time-ordered observed shipments;
    purge : departures in [T - gap, T) - dropped from both sides so port episodes (rolling 24 h
            state, multi-day disruptions) are not shared between train and test;
    train : departures before T - gap.
    """
    # Grab the departure timestamp column — this is what we sort/split by
    t = table["departure_ts"]

    # Find the timestamp that marks the start of the test set.
    # For example, if test_fraction=0.15 and there are 1000 rows, T is the
    # departure time of row 850 (i.e. the last 15% of shipments become "test").
    test_start = t.iloc[int(np.floor(len(t) * (1 - test_fraction)))]

    # The training set ends *before* the purge gap.
    # For example, if test_start is Jan 10 and gap_hours=48, train_end is Jan 8.
    # Everything between Jan 8 and Jan 10 is "purge" (thrown away).
    train_end = test_start - pd.Timedelta(hours=gap_hours)

    # Label every row as "test", "train", or "purge" based on its departure time
    # np.where works like an if/else: if departure >= test_start → "test",
    # else if departure < train_end → "train", otherwise → "purge"
    split = np.where(t >= test_start, "test", np.where(t < train_end, "train", "purge"))

    # Return the labels as a pandas Series (one label per row) plus a dict of
    # the boundary timestamps for reference
    return pd.Series(split, index=table.index, name="split"), {"test_start": test_start, "train_end": train_end}


# =============================================================================================
# CROSS-VALIDATION FOLDS
# =============================================================================================
#
# WHAT IS CROSS-VALIDATION?
# -------------------------
# When training a model, we want to know how well it will perform on unseen
# data BEFORE we use the real test set. Cross-validation (CV) does this by
# splitting the training data into multiple "folds" — each fold takes a
# different chunk as a mini test (called "validation") and trains on the rest.
#
# WHAT ARE "EXPANDING-WINDOW" FOLDS?
# -----------------------------------
# Because our data is time-ordered, we use EXPANDING-WINDOW folds:
#   Fold 1: train on months 1-3,   validate on month 4
#   Fold 2: train on months 1-5,   validate on month 6
#   Fold 3: train on months 1-7,   validate on month 8
#
# Each fold trains on ALL past data up to that point. This respects the
# time ordering and avoids future-data leakage. The validation block always
# comes AFTER the training data in time. A purge gap is also applied between
# training and validation within each fold.
# =============================================================================================

def time_series_folds(train_table, n_folds=config.CV_FOLDS, val_fraction=config.CV_VAL_FRACTION, gap_hours=config.PURGE_GAP_HOURS):
    """Expanding-window folds inside the training period (positional indices into train_table).

    Fold k validates on a consecutive block of `val_fraction` of the training rows; it trains on
    every row that departed more than `gap_hours` before the block starts. The last fold's
    validation block is the most recent part of the training period and is reported as the
    validation set.
    """
    # Reset the index so we have simple 0, 1, 2, ... row numbers
    t = train_table["departure_ts"].reset_index(drop=True)

    # Total number of training rows
    n = len(t)

    # How many rows go into each validation block
    # For example, if n=1000 and val_fraction=0.1, each block is 100 rows
    block = int(n * val_fraction)

    # We'll collect (train_indices, validation_indices) pairs here
    folds = []

    # Build each fold one at a time
    for k in range(n_folds):
        # Calculate where this fold's validation block starts.
        # Later folds validate on more recent data; the last fold validates
        # on the very end of the training period.
        v0 = n - (n_folds - k) * block

        # Get the actual departure timestamp at that position
        val_start = t.iloc[v0]

        # Mark rows that are in the validation window (on or after val_start)
        in_val = t >= val_start

        # For all folds except the last one, cap the validation block so it
        # doesn't extend beyond its intended size
        if k < n_folds - 1:
            in_val &= t < t.iloc[v0 + block]  # blocks are cut on timestamps, so ties never straddle a boundary

        # Convert the True/False mask to actual row numbers (indices)
        val_idx = np.flatnonzero(in_val.values)

        # Training rows are everything that departed BEFORE the validation
        # start minus the purge gap — this ensures no overlap or leakage
        tr_idx = np.flatnonzero((t < val_start - pd.Timedelta(hours=gap_hours)).values)

        # Store this fold's train/validation index pair
        folds.append((tr_idx, val_idx))

    return folds


# =============================================================================================
# BALANCED SAMPLE WEIGHTS
# =============================================================================================
#
# WHY DO WE NEED BALANCED WEIGHTS?
# ---------------------------------
# Imagine 95% of shipments arrive on time and only 5% are delayed. If the
# model just always says "on time", it's 95% accurate — but completely
# useless for catching delays!
#
# Balanced sample weights fix this by giving MORE importance (higher weight)
# to rare events (delays) and LESS importance to common events (on-time).
# Specifically, each class's weight is inversely proportional to how often
# it appears: rare classes get high weights, common classes get low weights.
# This forces the model to learn patterns for ALL classes, not just the
# majority class.
# =============================================================================================

def balanced_sample_weight(y):
    """Inverse class-frequency weights, computed on whatever rows are being fitted."""
    # sklearn's compute_sample_weight("balanced", y) automatically calculates
    # a weight for each row so that rare classes are up-weighted
    return compute_sample_weight("balanced", y)


# =============================================================================================
# SUPERVISED PREPROCESSORS
# =============================================================================================
#
# Different ML algorithm families need the data prepared in different ways:
#
# • LINEAR models (e.g. logistic regression): need numbers on similar scales
#   and can't handle missing values or text categories directly.
#
# • FOREST models (e.g. random forest): handle different scales fine (no
#   scaling needed), but still can't handle missing values or text.
#
# • BOOSTING models (e.g. XGBoost, LightGBM): can handle missing values
#   internally, so we leave numbers as-is. Still need one-hot for categories.
#
# All families get ONE-HOT ENCODING for categorical (text) columns.
# =============================================================================================

def _onehot():
    """Create a one-hot encoder that ignores categories it hasn't seen before.

    'handle_unknown="ignore"' means: if a category appears in new data that
    wasn't in the training data, just set all its columns to 0 instead of
    crashing. 'sparse_output=False' means: return a regular array (not a
    compressed sparse matrix), which is easier to work with.
    """
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def make_preprocessor(kind, numeric=NUMERIC, categorical=CATEGORICAL):
    """`numeric` / `categorical` default to the full feature set; ablations pass subsets."""

    # Choose how to handle numeric columns based on the algorithm family
    if kind == "linear":
        # LINEAR models need:
        # 1. Imputation: fill missing values with the median (middle value)
        #    of each column. add_indicator=True also creates a new column
        #    that flags "this value was originally missing" — the model can
        #    learn whether missingness itself is a useful signal.
        # 2. Standard scaling: subtract the mean and divide by standard
        #    deviation so all features are on a comparable scale.
        num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])

    elif kind == "forest":
        # FOREST models don't need scaling (decision trees split on
        # individual features regardless of scale), but they still need
        # missing values filled in and missing-value indicators added
        num = SimpleImputer(strategy="median", add_indicator=True)

    elif kind == "boosting":
        # BOOSTING models like XGBoost/LightGBM can handle NaN (missing
        # values) natively — they learn which branch to send missing values
        # down. So we just pass the numbers through unchanged ("passthrough").
        num = "passthrough"

    else:
        # If someone passes an unknown kind, raise an error
        raise ValueError(kind)

    # ColumnTransformer applies different transformations to different columns
    # at the same time:
    #   - "num": apply the numeric pipeline (chosen above) to numeric columns
    #   - "cat": apply one-hot encoding to categorical (text) columns
    #   - remainder="drop": throw away any columns not listed above
    return ColumnTransformer(
        [("num", num, list(numeric)), ("cat", _onehot(), list(categorical))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# =============================================================================================
# ANOMALY PREPROCESSORS
# =============================================================================================
#
# WHAT IS THE PortRelativeScaler AND WHY DO WE NEED IT?
# ------------------------------------------------------
# Different ports operate at very different "normal" levels. For example,
# Rotterdam (NLRTM) is one of the world's busiest ports — it routinely has
# high vessel counts, long dwell times, and near-100% berth utilization.
# A smaller port might see a fraction of that traffic.
#
# If we fed raw numbers into an anomaly detector, it would flag Rotterdam as
# "anomalous" every day just because its numbers are big — that's not useful!
#
# The PortRelativeScaler fixes this by computing a Z-SCORE relative to each
# port's own historical average:
#
#   z = (current_value − that_port's_mean) / that_port's_standard_deviation
#
# Now a z-score of +3 means "this port is 3 standard deviations above ITS
# OWN normal" — regardless of whether it's a huge or tiny port. This way
# the anomaly detector finds genuinely unusual conditions, not just "big
# port vs. small port" differences.
#
# For ports the model has never seen before (new ports), we fall back to the
# global average across all ports.
# =============================================================================================

class PortRelativeScaler(BaseEstimator, TransformerMixin):
    """Standardise selected columns within each origin port (mean/sd learned at fit time).

    Normal operating levels differ by port (e.g. NLRTM runs at saturation), so a detector on raw
    values would flag "large port" rather than "unusual state". Unseen ports fall back to the
    global statistics. Other columns pass through unchanged. Output is a numeric ndarray.
    """

    def __init__(self, cols=ANOMALY_PORT_RELATIVE, other=None, group_col="origin_port_id"):
        # cols: the columns that should be scaled relative to each port
        # other: additional columns to keep but NOT port-scale (just impute)
        # group_col: which column identifies the port (used for grouping)
        self.cols = cols
        self.other = other
        self.group_col = group_col

    def fit(self, X, y=None):
        """Learn each port's mean and standard deviation from the training data.

        This method is called once on the training data. It memorises the
        statistics so they can be applied later to new data via transform().
        """
        # Figure out which "other" columns to keep (all anomaly features that
        # are NOT in the port-relative set)
        other = self.other if self.other is not None else [c for c in ANOMALY_FEATURES if c not in self.cols]
        self.other_ = list(other)

        # Compute the median of every column (port-relative + other) for use
        # in filling missing values ("imputation")
        self.medians_ = X[list(self.cols) + self.other_].median()

        # Fill in missing values with the median before computing statistics
        Xc = X[list(self.cols)].fillna(self.medians_)

        # Compute the GLOBAL mean and standard deviation across all ports
        # These are used as a fallback for ports we haven't seen before
        self.global_mean_ = Xc.mean()
        # replace(0, 1) prevents division by zero: if a column has zero
        # spread, pretend the spread is 1 (no scaling effect)
        self.global_sd_ = Xc.std().replace(0, 1)

        # Group the data by port and compute each PORT's own mean and std dev
        g = Xc.groupby(X[self.group_col].values)
        self.mean_ = g.mean()          # per-port averages
        self.sd_ = g.std().fillna(self.global_sd_).replace(0, 1)  # per-port spreads
        return self

    def transform(self, X):
        """Apply the learned per-port scaling to new data.

        For each row, look up which port it belongs to, then subtract that
        port's mean and divide by that port's standard deviation. If the port
        wasn't seen during training, use the global statistics instead.
        """
        # Fill missing values in the port-relative columns using learned medians
        Xc = X[list(self.cols)].fillna(self.medians_[list(self.cols)])

        # Look up which port each row belongs to
        ports = X[self.group_col].values

        # Get the mean for each row's port; if the port is unknown, fall back
        # to the global mean. ".reindex(ports)" aligns the per-port stats to
        # match the order of rows in X.
        mu = self.mean_.reindex(ports).fillna(self.global_mean_).values

        # Same for standard deviation — unknown ports get the global std dev
        sd = self.sd_.reindex(ports).fillna(self.global_sd_).values

        # Compute the Z-score: how many standard deviations away from that
        # port's normal this value is
        z = (Xc.values - mu) / sd

        # For the "other" (non-port-relative) columns, just fill missing
        # values with the median and pass them through unchanged
        rest = X[self.other_].fillna(self.medians_[self.other_]).values

        # Combine the z-scored columns and the pass-through columns side by
        # side into a single numeric array
        return np.hstack([z, rest]).astype(float)

    def get_feature_names_out(self, input_features=None):
        """Return human-readable feature names for the output columns.

        Port-relative columns get a '_portz' suffix to show they've been
        z-scored per port.
        """
        return np.array([f"{c}_portz" for c in self.cols] + self.other_)


# The complete list of columns that the anomaly preprocessor needs as input:
# all anomaly features PLUS the port ID column (needed for grouping)
ANOMALY_INPUT_COLUMNS = ANOMALY_FEATURES + ["origin_port_id"]


def make_anomaly_preprocessor(scale_globally):
    """Build the preprocessing pipeline for anomaly detection models.

    Every anomaly model gets port-relative scaling first. If scale_globally
    is True (needed for algorithms like One-Class SVM that are sensitive to
    overall feature magnitude), an additional StandardScaler is applied on
    top of the port-relative z-scores so that ALL features end up with mean=0
    and std=1 across all ports combined.
    """
    # Start with the PortRelativeScaler (always applied)
    steps = [("port_relative", PortRelativeScaler())]

    # Optionally add global standard scaling on top
    if scale_globally:
        steps.append(("scale", StandardScaler()))

    # Wrap the steps in a Pipeline so they execute in order
    return Pipeline(steps)
