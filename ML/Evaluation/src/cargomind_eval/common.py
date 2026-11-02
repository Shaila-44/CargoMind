"""Shared paths, loading and plotting helpers for the independent evaluation.

The evaluation reads the saved models in ml/models/ and the stored outputs in ml/results/; it
never writes to either. Every output goes under ml/evaluation/.
"""
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EVAL_ROOT = Path(__file__).resolve().parents[2]  # ML/Evaluation
ML_ROOT = EVAL_ROOT.parent / "ML"  # ML/ML (modelling stage); cargomind_ml is an installed package

from cargomind_ml import config as mlc  # noqa: E402
from cargomind_ml import train_models  # noqa: E402
from cargomind_ml.features import FEATURES  # noqa: E402
from cargomind_ml.preprocessing import ANOMALY_INPUT_COLUMNS  # noqa: E402

SEED = mlc.SEED
CLASSES = mlc.CLASSES
LABELS = [0, 1, 2]
SUPERVISED = train_models.SUPERVISED
ANOMALY = train_models.ANOMALY
SUP_LABEL = {m.NAME: m.LABEL for m in SUPERVISED}
ANO_LABEL = {m.NAME: m.LABEL for m in ANOMALY}
BOOT_REPS = mlc.BOOTSTRAP_REPS

CLASS_COLORS = {"On-Time": "#2a78d6", "At-Risk": "#eda100", "Critical": "#e34948"}  # same as EDA / ML stage
MODEL_COLORS = {"logistic_regression": "#8a8985", "random_forest": "#2a78d6", "xgboost": "#1b9e77", "lightgbm": "#7b3fb5",
                "isolation_forest": "#d95f02", "one_class_svm": "#1f78b4"}

OUT = {k: EVAL_ROOT / k for k in ("reports", "metrics", "figures", "validation", "data")}
FIG = {k: OUT["figures"] / k for k in ("confusion_matrices", "calibration", "roc_curves", "precision_recall",
                                       "anomaly_distributions", "temporal_performance", "operational", "robustness")}


def ensure_dirs():
    for p in [*OUT.values(), *FIG.values()]:
        p.mkdir(parents=True, exist_ok=True)


def setup_plots():
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.titlesize": 11, "axes.titleweight": "semibold",
                         "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e6e5e1", "legend.frameon": False})


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_context():
    """Rebuild the modelling table and split exactly as the ML stage did (read-only, nothing saved)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return train_models.prepare_data(save=False)


def load_supervised():
    return {m.NAME: joblib.load(mlc.MODELS / f"{m.NAME}.joblib") for m in SUPERVISED}


def load_anomaly():
    return {m.NAME: joblib.load(mlc.MODELS / f"{m.NAME}.joblib") for m in ANOMALY}


def predict_proba(models, df):
    return {k: m.predict_proba(df[FEATURES]) for k, m in models.items()}


def anomaly_scores(bundles, df):
    mods = {m.NAME: m for m in ANOMALY}
    return {k: mods[k].anomaly_score(b["pipeline"], df[ANOMALY_INPUT_COLUMNS]) for k, b in bundles.items()}


def day_index(df):
    return df["departure_ts"].dt.floor("D").values


def block_resamples(blocks, reps=BOOT_REPS, seed=SEED):
    """Yield row-index arrays for a cluster bootstrap that resamples whole blocks (e.g. departure days)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    idx_by = [np.flatnonzero(blocks == b) for b in uniq]
    for _ in range(reps):
        yield np.concatenate([idx_by[i] for i in rng.integers(0, len(uniq), len(uniq))])


def md_table(df, digits=3):
    return df.to_markdown(index=False, floatfmt=f".{digits}f")


__all__ = ["pd", "np", "plt", "mlc", "FEATURES", "ANOMALY_INPUT_COLUMNS"]
