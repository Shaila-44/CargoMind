"""Robustness: ablation / sensitivity variants with paired uncertainty, and performance slices."""
import numpy as np
import pandas as pd

from . import metrics as M
from .common import FIG, MODEL_COLORS, OUT, SUP_LABEL, block_resamples, mlc, plt, savefig

REF = "full model (reference)"


def _scores(y, pred):
    cm = M.confusion(y, pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.nan_to_num(np.diag(cm) / cm.sum(0))
        rec = np.nan_to_num(np.diag(cm) / cm.sum(1))
        f1 = np.nan_to_num(2 * prec * rec / (prec + rec))
    return {"macro_f1": f1.mean(), "critical_recall": rec[2], "critical_precision": prec[2], "at_risk_recall": rec[1],
            "accuracy": np.trace(cm) / cm.sum()}


def ablation_analysis(abl, test):
    """Each variant vs. the full model on the same test shipments, day-block paired bootstrap."""
    day = test.set_index("shipment_id")["departure_ts"].dt.floor("D").astype(str)
    stored = pd.read_csv(mlc.RESULT_DIRS["comparison"] / "ablation_and_sensitivity.csv")
    rows = []
    for name in SUP_LABEL:
        a = abl[abl["model"] == name]
        ref = a[a["variant"] == REF].set_index("shipment_id")
        for variant, g in a.groupby("variant", sort=False):
            g = g.set_index("shipment_id")
            r = ref.loc[g.index]
            y = g["y"].values
            sv, sr = _scores(y, g["pred"].values), _scores(y, r["pred"].values)
            blocks = day.loc[g.index].values
            draws = {k: [] for k in sv}
            for idx in block_resamples(blocks):
                a1, a0 = _scores(y[idx], g["pred"].values[idx]), _scores(y[idx], r["pred"].values[idx])
                for k in draws:
                    draws[k].append(a1[k] - a0[k])
            st = stored[(stored["model"] == SUP_LABEL[name]) & (stored["variant"] == variant)].iloc[0]
            row = {"model": SUP_LABEL[name], "variant": variant, "n_test": len(g),
                   "stored_macro_f1": st["macro_f1"], "rerun_matches_stored": bool(abs(st["macro_f1"] - sv["macro_f1"]) < 1e-9
                                                                                  and abs(st["critical_recall"] - sv["critical_recall"]) < 1e-9)}
            for k in sv:
                lo, hi = np.percentile(draws[k], [2.5, 97.5])
                row.update({k: sv[k], f"ref_{k}_same_rows": sr[k], f"delta_{k}": sv[k] - sr[k],
                            f"delta_{k}_ci_low": lo, f"delta_{k}_ci_high": hi})
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT["validation"] / "robustness_analysis.csv", index=False)
    return out


def plot_ablation(rob):
    variants = [v for v in dict.fromkeys(rob["variant"]) if v != REF]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    h = 0.2
    for ax, k, title in [(axes[0], "macro_f1", "Change in macro-F1 vs full model"), (axes[1], "critical_recall", "Change in Critical recall vs full model")]:
        for j, (name, label) in enumerate(SUP_LABEL.items()):
            g = rob[rob["model"] == label].set_index("variant").loc[variants]
            ypos = np.arange(len(variants)) + (j - 1.5) * h
            ax.errorbar(g[f"delta_{k}"], ypos, xerr=[g[f"delta_{k}"] - g[f"delta_{k}_ci_low"], g[f"delta_{k}_ci_high"] - g[f"delta_{k}"]],
                        fmt="o", ms=4, color=MODEL_COLORS[name], label=label, capsize=2, lw=1)
        ax.axvline(0, color="#0b0b0b", lw=0.8)
        ax.set_yticks(range(len(variants)), [v.replace("sensitivity: ", "S: ").replace("ablation: ", "A: ") for v in variants], fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("difference (95% day-block bootstrap interval)")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].invert_yaxis()
    savefig(fig, FIG["robustness"] / "ablation_deltas.png")


def slices(test, test_proba):
    """Performance of each saved model on subgroups of the test period."""
    t = test.copy()
    t["origin_disruption_active"] = np.where(t["dis_o_active_any"] > 0, "yes", "no")
    t["destination_disruption_active"] = np.where(t["dis_d_active_any"] > 0, "yes", "no")
    q = t["d_util_mean24"].quantile([1 / 3, 2 / 3]).values
    t["destination_utilisation_tercile"] = pd.cut(t["d_util_mean24"], [-np.inf, *q, np.inf], labels=["low", "mid", "high"]).astype(str)
    q = t["o_util_max24"].quantile([1 / 3, 2 / 3]).values
    t["origin_utilisation_tercile"] = pd.cut(t["o_util_max24"], [-np.inf, *q, np.inf], labels=["low", "mid", "high"]).astype(str)
    dims = ["shipping_mode", "route_type", "origin_port_id", "destination_port_id", "carrier", "origin_disruption_active",
            "destination_disruption_active", "destination_utilisation_tercile", "origin_utilisation_tercile"]
    rows = []
    for d in dims:
        for val, g in t.groupby(d):
            idx = t.index.get_indexer(g.index)
            y = g["y"].values
            for name, p in test_proba.items():
                s = _scores(y, p[idx].argmax(1))
                rows.append({"dimension": d, "value": val, "model": SUP_LABEL[name], "n": len(g), "n_critical": int((y == 2).sum()),
                             "critical_share": (y == 2).mean(), **s,
                             "critical_roc_auc": M.auc_rank(y == 2, p[idx, 2]) if 0 < (y == 2).sum() < len(y) else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(OUT["validation"] / "slice_performance.csv", index=False)
    return out
