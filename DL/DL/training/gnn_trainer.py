"""Training of the GCN + LSTM graph experiment (train + validation only; the test split is never loaded).

    .venv/bin/python -m dl.training.gnn_trainer            # 5 seeds, fixed configuration
    .venv/bin/python -m dl.training.gnn_trainer --control  # diagnostic control: same model with no edges (not a candidate)

Same protocol as the hour-weighted models (A_w / C_w / T_w): Adam lr 1e-3, gradient clipping 1.0,
ReduceLROnPlateau on validation RMSE, early stopping (patience 15) and checkpoint selection on validation
RMSE in hours, MSE on the per-port z-scored target weighted by each port's dwell variance (= MSE in hours).
Graph-specific: each sample is a snapshot of all 10 ports, and the loss and metrics use only targets whose
(port, t) window was kept by the dataset build (mask). Batch = 8 snapshots (~76 windows, close to the 64
windows per batch used by the other models). Outputs: DL/DL/artifacts/gnn/, DL/DL/training/logs/gnn/.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl.models import GCNLSTMForecaster, GNNConfig, save_checkpoint

from . import config as C
from . import data
from .graph_data import CHANNELS, GraphSplit, direct_route_adjacency, load_graph_split
from .trainer import set_seed

GNN = replace(C.A_W, name="G_w_gcn_lstm_hour_weighted", cell="gcn_lstm", batch_size=8,
              notes="GCN (direct-route graph, hidden 32) + per-node LSTM (hidden 32), hour-weighted masked MSE")
# Diagnostic control (not a candidate): identical model and protocol with the edges removed (self-loops only), so
# each port sees only its own features. Isolates the effect of neighbour aggregation from the rest of the pipeline.
GNN_CONTROL = replace(GNN, name="G_w_control_no_edges", notes="diagnostic control: same GCN+LSTM with an empty graph (self-loops only)")
GNN_ARTIFACTS = C.ARTIFACTS / "gnn"
GNN_LOGS = C.LOGS / "gnn"


def build_model(split: GraphSplit, cfg=GNN) -> GCNLSTMForecaster:
    adjacency = direct_route_adjacency(split.ports)
    if cfg.name == GNN_CONTROL.name:
        adjacency = adjacency * 0
    return GCNLSTMForecaster(GNNConfig(adjacency=adjacency.tolist(), ports=list(split.ports),
                                       input_size=len(CHANNELS), seq_len=split.X.shape[1], gcn_hidden=32, lstm_hidden=32,
                                       dropout=cfg.dropout))


def node_weights(train: GraphSplit) -> np.ndarray:
    """Per-port loss weight = dwell variance (saved scaler), normalised to mean 1 over the training targets used."""
    var = data.dwell_std_by_port_index() ** 2
    w = var[None, :].repeat(len(train.X), 0)[train.mask]
    return (var / w.mean()).astype(np.float32)


def masked_weighted_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if not (pred.shape == y.shape == mask.shape):
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)}, y {tuple(y.shape)}, mask {tuple(mask.shape)}")
    m = mask.to(pred.dtype)
    return (m * w * (pred - y) ** 2).sum() / m.sum().clamp(min=1.0)


@torch.no_grad()
def predict_hours(model: GCNLSTMForecaster, split: GraphSplit, batch: int = 64) -> np.ndarray:
    """[S, N] predictions in hours (all nodes; callers apply split.mask)."""
    model.eval()
    z = np.concatenate([model(torch.from_numpy(split.X[i:i + batch])).numpy() for i in range(0, len(split.X), batch)])
    port_index = np.broadcast_to(np.arange(z.shape[1]), z.shape)
    return data.to_hours(z.ravel(), port_index.ravel()).reshape(z.shape)


def rmse_hours(model, split: GraphSplit) -> float:
    e = predict_hours(model, split)[split.mask] - split.y_hours[split.mask]
    return float(np.sqrt(np.mean(e ** 2)))


def train(cfg=GNN, train_split: GraphSplit | None = None, val_split: GraphSplit | None = None, max_epochs: int | None = None):
    set_seed(cfg.seed, cfg.num_threads)
    tr = train_split or load_graph_split("train")
    va = val_split or load_graph_split("val")
    model = build_model(tr, cfg)
    w = torch.from_numpy(node_weights(tr))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)
    g = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(tr.X), torch.from_numpy(tr.y), torch.from_numpy(tr.mask)),
                        batch_size=cfg.batch_size, shuffle=True, generator=g, num_workers=0)
    history, best, best_epoch, best_state, stale = [], float("inf"), -1, None, 0
    t0 = time.time()
    for epoch in range(1, (max_epochs or cfg.max_epochs) + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb, mb in loader:
            opt.zero_grad()
            loss = masked_weighted_mse(model(xb), yb, mb, w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item() * int(mb.sum())
            n += int(mb.sum())
        val_rmse, train_rmse = rmse_hours(model, va), rmse_hours(model, tr)
        history.append({"epoch": epoch, "train_objective": total / n, "train_rmse_h": train_rmse, "val_rmse_h": val_rmse,
                        "lr": opt.param_groups[0]["lr"]})
        sched.step(val_rmse)
        if val_rmse < best - 1e-6:
            best, best_epoch, stale = val_rmse, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_epoch": best_epoch, "best_val_rmse_h": best, "seconds": time.time() - t0}


def main(seeds=C.SEEDS, base=GNN) -> pd.DataFrame:
    tr, va = load_graph_split("train"), load_graph_split("val")
    rows = []
    for seed in seeds:
        cfg = base.with_seed(seed)
        model, res = train(cfg, tr, va)
        hist = pd.DataFrame(res["history"])
        (GNN_LOGS / base.name).mkdir(parents=True, exist_ok=True)
        hist.to_csv(GNN_LOGS / base.name / f"seed_{seed}.csv", index=False)
        assert abs(rmse_hours(model, va) - res["best_val_rmse_h"]) < 1e-4, "restored model must reproduce its best validation RMSE"
        save_checkpoint(GNN_ARTIFACTS / "runs" / base.name / f"seed_{seed}.pt", model, CHANNELS,
                        {"train_config": cfg.to_dict(), "best_epoch": res["best_epoch"], "best_val_rmse_h": res["best_val_rmse_h"],
                         "input": "graph snapshots [batch, 12, 10, 8] from dl.training.graph_data.load_graph_split",
                         "graph": "undirected direct-route adjacency from port_network.csv, self-loops, symmetric normalisation"
                         if base.name == GNN.name else "control: empty graph (self-loops only)"})
        best_row = hist.loc[hist["epoch"] == res["best_epoch"]].iloc[0]
        rows.append({"experiment": base.name, "seed": seed, "n_parameters": model.n_parameters(), "best_epoch": res["best_epoch"],
                     "epochs_run": len(hist), "train_rmse_h_at_best": best_row["train_rmse_h"], "val_rmse_h": res["best_val_rmse_h"],
                     "seconds": round(res["seconds"], 1), "notes": base.notes})
        print(f"  {base.name} seed {seed}: val RMSE {res['best_val_rmse_h']:.4f} h at epoch {res['best_epoch']} ({res['seconds']:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    GNN_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = GNN_ARTIFACTS / "experiment_results.csv"
    if path.exists():  # keep other experiments' rows
        old = pd.read_csv(path)
        out = pd.concat([old[old["experiment"] != base.name], out], ignore_index=True)
    out.to_csv(path, index=False)
    out = out[out["experiment"] == base.name]
    print(out[["seed", "val_rmse_h", "best_epoch"]].to_string(index=False), f"\nmean {out.val_rmse_h.mean():.4f} ± {out.val_rmse_h.std():.4f} h")
    return out


if __name__ == "__main__":
    import sys

    main(base=GNN_CONTROL if "--control" in sys.argv[1:] else GNN)
