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
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from . import config
from .features import ANOMALY_FEATURES, ANOMALY_PORT_RELATIVE, CATEGORICAL, NUMERIC


# ---------------------------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------------------------
def chronological_split(table, test_fraction=config.TEST_FRACTION, gap_hours=config.PURGE_GAP_HOURS):
    """Assign each shipment to train / purge / test by departure time.

    test  : departures at or after T, where T is the departure time of the shipment at the
            (1 - test_fraction) quantile of the time-ordered observed shipments;
    purge : departures in [T - gap, T) - dropped from both sides so port episodes (rolling 24 h
            state, multi-day disruptions) are not shared between train and test;
    train : departures before T - gap.
    """
    t = table["departure_ts"]
    test_start = t.iloc[int(np.floor(len(t) * (1 - test_fraction)))]
    train_end = test_start - pd.Timedelta(hours=gap_hours)
    split = np.where(t >= test_start, "test", np.where(t < train_end, "train", "purge"))
    return pd.Series(split, index=table.index, name="split"), {"test_start": test_start, "train_end": train_end}


def time_series_folds(train_table, n_folds=config.CV_FOLDS, val_fraction=config.CV_VAL_FRACTION, gap_hours=config.PURGE_GAP_HOURS):
    """Expanding-window folds inside the training period (positional indices into train_table).

    Fold k validates on a consecutive block of `val_fraction` of the training rows; it trains on
    every row that departed more than `gap_hours` before the block starts. The last fold's
    validation block is the most recent part of the training period and is reported as the
    validation set.
    """
    t = train_table["departure_ts"].reset_index(drop=True)
    n = len(t)
    block = int(n * val_fraction)
    folds = []
    for k in range(n_folds):
        v0 = n - (n_folds - k) * block
        val_start = t.iloc[v0]
        in_val = t >= val_start
        if k < n_folds - 1:
            in_val &= t < t.iloc[v0 + block]  # blocks are cut on timestamps, so ties never straddle a boundary
        val_idx = np.flatnonzero(in_val.values)
        tr_idx = np.flatnonzero((t < val_start - pd.Timedelta(hours=gap_hours)).values)
        folds.append((tr_idx, val_idx))
    return folds


def balanced_sample_weight(y):
    """Inverse class-frequency weights, computed on whatever rows are being fitted."""
    return compute_sample_weight("balanced", y)


# ---------------------------------------------------------------------------------------------
# Supervised preprocessors
# ---------------------------------------------------------------------------------------------
def _onehot():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def make_preprocessor(kind, numeric=NUMERIC, categorical=CATEGORICAL):
    """`numeric` / `categorical` default to the full feature set; ablations pass subsets."""
    if kind == "linear":
        num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    elif kind == "forest":
        num = SimpleImputer(strategy="median", add_indicator=True)
    elif kind == "boosting":
        num = "passthrough"
    else:
        raise ValueError(kind)
    return ColumnTransformer(
        [("num", num, list(numeric)), ("cat", _onehot(), list(categorical))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------------------------
# Anomaly preprocessors
# ---------------------------------------------------------------------------------------------
class PortRelativeScaler(BaseEstimator, TransformerMixin):
    """Standardise selected columns within each origin port (mean/sd learned at fit time).

    Normal operating levels differ by port (e.g. NLRTM runs at saturation), so a detector on raw
    values would flag "large port" rather than "unusual state". Unseen ports fall back to the
    global statistics. Other columns pass through unchanged. Output is a numeric ndarray.
    """

    def __init__(self, cols=ANOMALY_PORT_RELATIVE, other=None, group_col="origin_port_id"):
        self.cols = cols
        self.other = other
        self.group_col = group_col

    def fit(self, X, y=None):
        other = self.other if self.other is not None else [c for c in ANOMALY_FEATURES if c not in self.cols]
        self.other_ = list(other)
        self.medians_ = X[list(self.cols) + self.other_].median()
        Xc = X[list(self.cols)].fillna(self.medians_)
        self.global_mean_ = Xc.mean()
        self.global_sd_ = Xc.std().replace(0, 1)
        g = Xc.groupby(X[self.group_col].values)
        self.mean_ = g.mean()
        self.sd_ = g.std().fillna(self.global_sd_).replace(0, 1)
        return self

    def transform(self, X):
        Xc = X[list(self.cols)].fillna(self.medians_[list(self.cols)])
        ports = X[self.group_col].values
        mu = self.mean_.reindex(ports).fillna(self.global_mean_).values
        sd = self.sd_.reindex(ports).fillna(self.global_sd_).values
        z = (Xc.values - mu) / sd
        rest = X[self.other_].fillna(self.medians_[self.other_]).values
        return np.hstack([z, rest]).astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_portz" for c in self.cols] + self.other_)


ANOMALY_INPUT_COLUMNS = ANOMALY_FEATURES + ["origin_port_id"]


def make_anomaly_preprocessor(scale_globally):
    steps = [("port_relative", PortRelativeScaler())]
    if scale_globally:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)
