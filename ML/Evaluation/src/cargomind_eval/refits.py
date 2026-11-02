"""Evaluation-only re-fits of the saved model configurations.

None of these replace the saved models. They exist because some evaluations need predictions
that the ML stage did not store:
* out-of-fold predictions on the three time-series CV validation blocks (calibration and
  decision-threshold analysis must use validation data, never the test period);
* predictions of the ablation / sensitivity variants (the ML stage stored only summary metrics,
  so paired confidence intervals for the differences could not be computed from the artifacts);
* a refit of each final model on the full training period, to check that training is reproducible.
Each re-fit clones the saved pipeline, i.e. it reuses the tuned hyperparameters unchanged.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from .common import FEATURES, OUT, SUPERVISED, mlc, train_models


def _fit(pipe, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return pipe.fit(X, y)


def oof_validation(ctx, models):
    """Probabilities for every CV validation block, from a model fitted on that fold's training rows."""
    train, folds = ctx["train"], ctx["folds"]
    parts = []
    for k, (tr_idx, val_idx) in enumerate(folds):
        val = train.iloc[val_idx]
        part = val[["shipment_id", "departure_ts", "origin_port_id", "shipping_mode", "delay_status", "y",
                    mlc.OUTCOME, "actual_arrival_ts"]].copy()
        part["fold"] = k
        for name, model in models.items():
            m = _fit(clone(model), train.iloc[tr_idx][FEATURES], train.iloc[tr_idx]["y"])
            p = m.predict_proba(val[FEATURES])
            for i, c in enumerate(mlc.CLASSES):
                part[f"{name}_p_{c}"] = p[:, i]
        parts.append(part)
    oof = pd.concat(parts, ignore_index=True)
    oof.to_csv(OUT["data"] / "oof_validation_predictions.csv", index=False)
    return oof


def proba_from(df, name):
    return df[[f"{name}_p_{c}" for c in mlc.CLASSES]].to_numpy()


def final_refit(ctx, models):
    """Refit each saved configuration on the whole training period; return test probabilities."""
    train, test = ctx["train"], ctx["test"]
    return {name: _fit(clone(m), train[FEATURES], train["y"]).predict_proba(test[FEATURES]) for name, m in models.items()}


def ablation_predictions(ctx, models):
    """Re-run the ML stage's ablation / sensitivity variants (same definitions) and keep row-level predictions."""
    from cargomind_ml.features import CATEGORICAL, FEATURE_GROUPS, IDENTITY_FEATURES, NUMERIC

    train, test, info = ctx["train"], ctx["test"], ctx["split_info"]
    state_groups = ["origin_congestion", "container_dwell", "anchor_queue", "weather", "disruptions", "destination_state", "hub_state"]
    state_cols = {c for g in state_groups for c in FEATURE_GROUPS[g]}
    filled = ["filled_route_id", "filled_carrier", "filled_shipping_mode", "filled_cargo_type",
              "filled_container_teu", "filled_shipment_priority", "filled_expected_arrival_ts"]
    variants = {
        "full model (reference)": (train, test, NUMERIC, CATEGORICAL),
        "ablation: no port/carrier identity": (train, test, NUMERIC, [c for c in CATEGORICAL if c not in IDENTITY_FEATURES]),
        "ablation: no operational state (booking-time info only)": (train, test, [c for c in NUMERIC if c not in state_cols], CATEGORICAL),
        "ablation: no historical delay rates": (train, test, [c for c in NUMERIC if c not in FEATURE_GROUPS["historical_delay_rates"]], CATEGORICAL),
        "sensitivity: train only on labels available at test start": (
            train[train["actual_arrival_ts"] < info["test_start"]], test, NUMERIC, CATEGORICAL),
        "sensitivity: exclude shipments with filled inputs (train+test)": (
            train[~train[filled].any(axis=1)], test[~test[filled].any(axis=1)], NUMERIC, CATEGORICAL),
        "sensitivity: sea shipments only (train+test)": (
            train[train["shipping_mode"] == "sea"], test[test["shipping_mode"] == "sea"], NUMERIC, CATEGORICAL),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx24 = train_models.prepare_data(lag_hours=24, save=False)
    t24 = ctx24["table"]
    tr24 = t24[t24["shipment_id"].isin(set(train["shipment_id"]))]
    te24 = t24[t24["shipment_id"].isin(set(test["shipment_id"]))]
    variants["sensitivity: prediction 24 h before departure"] = (tr24, te24, NUMERIC, CATEGORICAL)

    rows = []
    for module in SUPERVISED:
        for variant, (tr, te, num, cat) in variants.items():
            pipe = train_models._refit(models[module.NAME], module, tr, num, cat)
            p = pipe.predict_proba(te[num + cat])
            rows.append(pd.DataFrame({"model": module.NAME, "variant": variant, "shipment_id": te["shipment_id"].values,
                                      "y": te["y"].values, "pred": p.argmax(1), "p_critical": p[:, 2]}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT["data"] / "ablation_predictions.csv", index=False)
    return out


def oof_check(oof, models):
    """Per-fold macro-F1 / Critical recall of the re-fits vs. the scores stored at tuning time."""
    import json

    from sklearn.metrics import f1_score, recall_score

    rows = []
    for name in models:
        cfg = json.loads((mlc.MODELS / f"{name}_config.json").read_text())["cv_scores_best"]
        for k, g in oof.groupby("fold"):
            pred = proba_from(g, name).argmax(1)
            rows.append({"model": name, "fold": k,
                         "macro_f1_refit": f1_score(g["y"], pred, average="macro"),
                         "macro_f1_stored": cfg["f1_macro"]["per_fold"][k],
                         "critical_recall_refit": recall_score(g["y"], pred, labels=[2], average="macro"),
                         "critical_recall_stored": cfg["recall_critical"]["per_fold"][k]})
    df = pd.DataFrame(rows)
    df["max_abs_diff"] = np.maximum((df["macro_f1_refit"] - df["macro_f1_stored"]).abs(),
                                    (df["critical_recall_refit"] - df["critical_recall_stored"]).abs())
    return df
