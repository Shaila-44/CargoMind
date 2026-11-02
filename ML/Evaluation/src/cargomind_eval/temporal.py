"""Temporal validation: split integrity, label availability, performance by period and drift."""
import json

import numpy as np
import pandas as pd

from . import metrics as M
from .common import CLASS_COLORS, CLASSES, FIG, MODEL_COLORS, OUT, SUP_LABEL, mlc, plt, savefig
from .refits import proba_from


def verify_split(ctx):
    table, train, test, folds, info = ctx["table"], ctx["train"], ctx["test"], ctx["folds"], ctx["split_info"]
    purge = table[table["split"] == "purge"]
    stored = pd.read_csv(mlc.RESULT_DIRS["data"] / "splits.csv")
    rebuilt = pd.concat([train.assign(split="train"), purge, test.assign(split="test")])[["shipment_id", "split"]]
    same = stored.set_index("shipment_id")["split"].reindex(rebuilt["shipment_id"]).values == rebuilt["split"].values
    gap_h = (test["departure_ts"].min() - train["departure_ts"].max()).total_seconds() / 3600
    checks = {
        "n_train": len(train), "n_purge": len(purge), "n_test": len(test),
        "train_departure_range": [str(train["departure_ts"].min()), str(train["departure_ts"].max())],
        "purge_departure_range": [str(purge["departure_ts"].min()), str(purge["departure_ts"].max())],
        "test_departure_range": [str(test["departure_ts"].min()), str(test["departure_ts"].max())],
        "test_start": str(info["test_start"]), "train_end_exclusive": str(info["train_end"]),
        "train_strictly_before_purge_before_test": bool(train["departure_ts"].max() < info["train_end"] <= purge["departure_ts"].min()
                                                        and purge["departure_ts"].max() < info["test_start"] <= test["departure_ts"].min()),
        "observed_gap_last_train_to_first_test_departure_h": gap_h,
        "purge_gap_at_least_72h": bool(gap_h >= 72),
        "shipment_id_overlap_train_test": int(len(set(train["shipment_id"]) & set(test["shipment_id"]))),
        "shipment_id_overlap_purge_any": int(len(set(purge["shipment_id"]) & (set(train["shipment_id"]) | set(test["shipment_id"])))),
        "test_fraction_of_observed": len(test) / len(table),
        "rebuilt_split_matches_stored_splits_csv": bool(same.all()),
        "train_arrivals_after_test_start": int((train["actual_arrival_ts"] >= info["test_start"]).sum()),
        "folds": [],
    }
    for k, (tr, va) in enumerate(folds):
        t_tr, t_va = train.iloc[tr], train.iloc[va]
        checks["folds"].append({
            "fold": k, "n_train": len(tr), "n_val": len(va),
            "train_to": str(t_tr["departure_ts"].max()), "val_from": str(t_va["departure_ts"].min()), "val_to": str(t_va["departure_ts"].max()),
            "gap_h": (t_va["departure_ts"].min() - t_tr["departure_ts"].max()).total_seconds() / 3600,
            "indices_inside_training_period": bool(max(tr.max(), va.max()) < len(train)),
            "val_critical_share": float((t_va["y"] == 2).mean()),
            "fold_train_labels_not_arrived_at_val_start": int((t_tr["actual_arrival_ts"] >= t_va["departure_ts"].min()).sum()),
            "fold_train_labels_not_arrived_share": float((t_tr["actual_arrival_ts"] >= t_va["departure_ts"].min()).mean()),
        })
    for name in SUP_LABEL:
        cvr = pd.read_csv(mlc.RESULT_DIRS["metrics"] / f"{name}_cv_results.csv")
        checks.setdefault("cv_results_n_candidates", {})[name] = len(cvr)
    # test labels: when do they become known?
    arr = test["actual_arrival_ts"]
    checks["test_labels_known_by_test_end_share"] = float((arr <= test["departure_ts"].max()).mean())
    checks["test_label_arrival_range"] = [str(arr.min()), str(arr.max())]
    (OUT["validation"] / "split_verification.json").write_text(json.dumps(checks, indent=2, default=str))
    return checks


def _period_rows(df, probas, period_col, set_name):
    rows = []
    for per, g in df.groupby(period_col, sort=True):
        idx = df.index.get_indexer(g.index)
        y = g["y"].values
        for name, p in probas.items():
            m = M.classification(y, p[idx])
            rows.append({"set": set_name, "period": str(per), "model": SUP_LABEL[name], "n": len(g),
                         "critical_share": (y == 2).mean(), "at_risk_share": (y == 1).mean(),
                         **{k: m[k] for k in ("accuracy", "macro_f1", "critical_recall", "critical_precision",
                                              "at_risk_recall", "critical_roc_auc_ovr", "roc_auc_ovr_macro", "log_loss")},
                         "mean_p_critical": p[idx, 2].mean()})
    return rows


def performance_by_period(ctx, oof, test_proba):
    test = ctx["test"].copy()
    rows = []
    for k, g in oof.groupby("fold"):
        g = g.reset_index(drop=True).assign(period=f"cv fold {k}")
        rows += _period_rows(g, {n: proba_from(g, n) for n in SUP_LABEL}, "period", "cv")
    test["period"] = test["departure_ts"].dt.floor("D").dt.strftime("%m-%d")
    rows += _period_rows(test, test_proba, "period", "test day")
    # 2-3 day blocks so each block holds enough Critical shipments
    start = test["departure_ts"].min().floor("D")
    test["block"] = ((test["departure_ts"] - start).dt.days // 3).map(lambda b: f"test days {3 * b + 1}-{3 * b + 3}")
    rows += _period_rows(test, test_proba, "block", "test 3-day block")
    out = pd.DataFrame(rows)
    out.to_csv(OUT["metrics"] / "temporal_performance.csv", index=False)
    return out


def prevalence_over_time(ctx):
    t = ctx["table"].copy()
    t["day"] = t["departure_ts"].dt.floor("D")
    share = pd.crosstab(t["day"], t["delay_status"], normalize="index")[CLASSES]
    n = t.groupby("day").size()
    split = t.groupby("day")["split"].agg(lambda s: s.mode().iloc[0])
    out = share.assign(n=n, split=split)
    out.to_csv(OUT["metrics"] / "class_prevalence_by_day.csv")
    return out


def plot_temporal(perf, prev, ctx):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    ax = axes[0]
    bottom = np.zeros(len(prev))
    x = prev.index
    for c in CLASSES:
        ax.bar(x, prev[c], bottom=bottom, color=CLASS_COLORS[c], width=0.9, label=c)
        bottom += prev[c].values
    for t, lab in [(ctx["split_info"]["train_end"], "purge starts"), (ctx["split_info"]["test_start"], "test starts")]:
        ax.axvline(t, color="#0b0b0b", ls="--", lw=1)
        ax.text(t, 1.02, lab, fontsize=8)
    for k, f in enumerate(ctx["folds"]):
        v = ctx["train"].iloc[f[1]]["departure_ts"]
        ax.axvspan(v.min(), v.max(), ymin=0.97, ymax=1.0, color="#52514e", alpha=0.6)
    ax.set_ylabel("class share of departures")
    ax.set_title("Class mix by departure day (grey strips at the top: CV validation blocks)")
    ax.legend(ncol=3, fontsize=8, loc="lower left")
    test_day = perf[perf["set"] == "test day"]
    for ax, metric, title in [(axes[1], "critical_recall", "Critical recall by test day"),
                              (axes[2], "critical_roc_auc_ovr", "Critical ROC-AUC (prevalence-independent) by test day")]:
        for name, label in SUP_LABEL.items():
            g = test_day[test_day["model"] == label]
            ax.plot(g["period"], g[metric], "o-", color=MODEL_COLORS[name], lw=1.4, ms=4, label=label)
        ax2 = ax.twinx()
        g = test_day[test_day["model"] == SUP_LABEL["logistic_regression"]]
        ax2.bar(g["period"], g["critical_share"], color=CLASS_COLORS["Critical"], alpha=0.15, width=0.6)
        ax2.set_ylabel("Critical share (bars)", fontsize=8)
        ax2.grid(False)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, ncol=4, loc="lower left")
    axes[2].set_xlabel("test departure day (2025, UTC)")
    savefig(fig, FIG["temporal_performance"] / "prevalence_and_performance_over_time.png")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    cvv = perf[perf["set"].isin(["cv", "test 3-day block"])]
    order = list(dict.fromkeys(cvv["period"]))
    for name, label in SUP_LABEL.items():
        g = cvv[cvv["model"] == label].set_index("period").reindex(order)
        ax.plot(order, g["macro_f1"], "o-", color=MODEL_COLORS[name], label=label)
    ax.set_ylabel("macro-F1")
    ax.set_title("Macro-F1 across time: CV validation blocks (January) and 3-day test blocks (February)")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20)
    savefig(fig, FIG["temporal_performance"] / "macro_f1_cv_vs_test_blocks.png")


def psi(a, b, bins=10):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        return np.nan
    ea = np.histogram(np.clip(a, q[0], q[-1]), q)[0] / len(a)
    eb = np.histogram(np.clip(b, q[0], q[-1]), q)[0] / len(b)
    ea, eb = np.clip(ea, 1e-4, None), np.clip(eb, 1e-4, None)
    return float(((eb - ea) * np.log(eb / ea)).sum())


def feature_drift(ctx):
    from cargomind_ml.features import FEATURE_TO_GROUP, NUMERIC

    train, test = ctx["train"], ctx["test"]
    rows = []
    for c in NUMERIC:
        a, b = train[c].astype(float).values, test[c].astype(float).values
        rows.append({"feature": c, "group": FEATURE_TO_GROUP[c], "psi_train_vs_test": psi(a, b),
                     "train_mean": np.nanmean(a), "test_mean": np.nanmean(b),
                     "train_missing": np.isnan(a).mean(), "test_missing": np.isnan(b).mean(),
                     # relationship drift: mean of the feature among Critical shipments relative to all shipments
                     "train_mean_critical_minus_all": np.nanmean(a[train["y"].values == 2]) - np.nanmean(a),
                     "test_mean_critical_minus_all": np.nanmean(b[test["y"].values == 2]) - np.nanmean(b)})
    out = pd.DataFrame(rows).sort_values("psi_train_vs_test", ascending=False)
    out.to_csv(OUT["metrics"] / "feature_drift_train_vs_test.csv", index=False)
    return out


def prevalence_adjusted_precision(table, priors):
    """Critical precision implied by each model's test recall / false-positive rate at other prevalences."""
    rows = []
    t = table[table["set"] == "test"]
    for _, r in t[t["key"] != "dummy_prior"].iterrows():
        n_neg = r["n"] - r["critical_support"]
        fpr = r["false_critical_alerts"] / n_neg
        rec = r["critical_recall"]
        row = {"model": r["model"], "test_critical_recall": rec, "test_fpr_critical": fpr, "test_critical_precision": r["critical_precision"]}
        for lab, pi in priors.items():
            row[f"critical_precision_at_{lab}"] = rec * pi / (rec * pi + fpr * (1 - pi))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT["metrics"] / "critical_precision_vs_prevalence.csv", index=False)
    return out
