"""Validation-only comparison of the GCN + LSTM experiment with EWMA, GRU, C_w and Transformer T_w.

    .venv/bin/python -m dl.evaluation.compare_gnn

GNN predictions are taken for exactly the kept validation windows (graph mask) and aligned to the other models
by window_id, so every model is scored on the same 1,306 windows. Reuses frames.context and the metrics module.
The test split is not loaded. Outputs: DL/Evaluation/results/gnn/, DL/Evaluation/figures/gnn/.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dl.models import load_checkpoint  # noqa: E402
from dl.training.config import ARTIFACTS, EVAL_FIGURES, EVAL_RESULTS, TRANSFORMER_ARTIFACTS  # noqa: E402
from dl.training.gnn_trainer import GNN, GNN_ARTIFACTS, GNN_CONTROL, predict_hours  # noqa: E402
from dl.training.graph_data import load_graph_split  # noqa: E402

from . import frames, metrics  # noqa: E402
from .evaluate import model_predictions  # noqa: E402

OUT = EVAL_RESULTS / "gnn"
FIG = EVAL_FIGURES / "gnn"
EWMA = "EWMA α=0.2"
WINDOW_MODELS = {"GRU": ARTIFACTS / "runs" / "GRU__B_w_lstm_port_embedding_hour_weighted",
                 "C_w LSTM": ARTIFACTS / "runs" / "C_w_lstm_ewma_residual_hour_weighted",
                 "Transformer T_w": TRANSFORMER_ARTIFACTS / "runs" / "T_w_transformer_hour_weighted"}
GRAPH_MODELS = {"GNN (GCN + LSTM)": GNN_ARTIFACTS / "runs" / GNN.name,
                "GNN control: no edges (diagnostic)": GNN_ARTIFACTS / "runs" / GNN_CONTROL.name}


def score(y, p) -> dict:
    return {"rmse_h": metrics.rmse(y, p), "mae_h": metrics.mae(y, p), "bias_h": float(np.mean(p - y)), "r2": metrics.r2(y, p)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    preds, frame = {}, None
    for label, d in WINDOW_MODELS.items():
        for s in range(5):
            p, sd = model_predictions(d / f"seed_{s}.pt", "val")
            preds[(label, s)] = p
            if frame is None:
                frame = frames.context(sd.window_id, sd.y_hours)
    va = load_graph_split("val")
    pos = {w: i for i, w in enumerate(frame["window_id"])}
    si, ni = np.nonzero(va.mask)
    order = np.argsort([pos[w] for w in va.window_id[si, ni]])  # graph entries -> frame row order
    assert len(order) == len(frame)
    for label, d in GRAPH_MODELS.items():
        for s in range(5):
            model, _ = load_checkpoint(d / f"seed_{s}.pt")
            p = predict_hours(model, va)[si, ni][order]
            assert np.allclose(va.y_hours[si, ni][order], frame["y_hours"], atol=1e-3)
            preds[(label, s)] = p
    y, port = frame["y_hours"].to_numpy(), frame["port_id"].to_numpy()
    models = {EWMA: [frames.baseline_predictions(frame)[EWMA]]}
    for label in list(WINDOW_MODELS) + list(GRAPH_MODELS):
        models[label] = [preds[(label, s)] for s in range(5)]

    rows, port_rows, sub_rows = [], [], []
    for name, plist in models.items():
        for s, p in enumerate(plist):
            rows.append({"model": name, "seed": s if len(plist) > 1 else None, **score(y, p),
                         "port_balanced_rmse_h": metrics.port_balanced_rmse(y, p, port)})
        for q in sorted(np.unique(port)):
            m = port == q
            port_rows.append({"model": name, "port": q, "n": int(m.sum()), "rmse_mean_h": float(np.mean([metrics.rmse(y[m], p[m]) for p in plist])),
                              "bias_mean_h": float(np.mean([np.mean(p[m] - y[m]) for p in plist]))})
        for g, m in {"all": np.ones(len(y), bool), "disruption active at t": frame["disruption_active_t"].to_numpy() == 1,
                     "no disruption at t": frame["disruption_active_t"].to_numpy() == 0}.items():
            sub_rows.append({"model": name, "subgroup": g, "n": int(m.sum()), "rmse_mean_h": float(np.mean([metrics.rmse(y[m], p[m]) for p in plist]))})
    seedwise = pd.DataFrame(rows)
    seedwise.to_csv(OUT / "validation_seedwise.csv", index=False)
    summary = seedwise.groupby("model", sort=False).agg(seeds=("rmse_h", "count"), rmse_mean=("rmse_h", "mean"), rmse_sd=("rmse_h", "std"),
                                                         mae_mean=("mae_h", "mean"), bias_mean=("bias_h", "mean"), r2_mean=("r2", "mean"),
                                                         port_balanced_rmse=("port_balanced_rmse_h", "mean")).reset_index()
    ewma = summary.loc[summary["model"] == EWMA, "rmse_mean"].iloc[0]
    summary["minus_ewma_h"] = summary["rmse_mean"] - ewma
    summary["n_parameters"] = [None] + [sum(p.numel() for p in load_checkpoint(d / "seed_0.pt")[0].parameters())
                                        for d in list(WINDOW_MODELS.values()) + list(GRAPH_MODELS.values())]
    summary.to_csv(OUT / "validation_comparison.csv", index=False)
    pd.DataFrame(port_rows).to_csv(OUT / "validation_per_port.csv", index=False)
    pd.DataFrame(sub_rows).to_csv(OUT / "validation_subgroups.csv", index=False)
    (OUT / "comparison_info.json").write_text(json.dumps({"split": "validation only", "n_windows": int(len(y)),
                                                          "note": "test split not loaded; GNN aligned to the same windows by window_id"}, indent=2))
    plot(pd.DataFrame(port_rows), summary)
    print(summary.round(4).to_string())


def plot(per_port: pd.DataFrame, summary: pd.DataFrame) -> None:
    colors = {EWMA: "#1b9e77", "GRU": "#eda100", "C_w LSTM": "#2a78d6", "Transformer T_w": "#e34948",
              "GNN (GCN + LSTM)": "#7b3fb5", "GNN control: no edges (diagnostic)": "#8a8985"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 2]})
    s = summary.set_index("model")
    axes[0].barh(s.index, s["rmse_mean"], xerr=s["rmse_sd"].fillna(0), color=[colors[m] for m in s.index], capsize=3)
    axes[0].invert_yaxis()
    axes[0].set(title="Validation RMSE (5-seed mean ± SD)", xlabel="RMSE (h)")
    ports = sorted(per_port["port"].unique())
    x = np.arange(len(ports))
    for i, m in enumerate(colors):
        v = per_port[per_port["model"] == m].set_index("port").loc[ports, "rmse_mean_h"]
        axes[1].bar(x + (i - 2.5) * 0.14, v, width=0.14, color=colors[m], label=m)
    axes[1].set_xticks(x, ports, rotation=45)
    axes[1].set(title="Validation RMSE by port (5-seed mean)", ylabel="RMSE (h)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "gnn_validation_comparison.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
