"""Seeded, deterministic training of one configuration on one seed.

* Loss: MSE on the scaled target (shapes asserted equal: [batch] vs [batch]).
* Model selection / early stopping: validation RMSE in hours (via the dataset's inverse transform).
* Training windows are shuffled within the training split only (seeded generator); validation is
  evaluated in stored order. The test split is never loaded here.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl.models import ModelConfig, RecurrentForecaster, TransformerConfig, TransformerForecaster

from . import data
from .config import TrainConfig


def set_seed(seed: int, num_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)


def build_model(cfg: TrainConfig, input_size: int, n_ports: int = 10, dwell_channel: int = 0) -> RecurrentForecaster | TransformerForecaster:
    if cfg.cell == "transformer":
        if cfg.port_embedding_dim or cfg.residual_ewma_alpha is not None:
            raise ValueError("the Transformer has no port embedding or residual head")
        return TransformerForecaster(TransformerConfig(input_size=input_size, d_model=cfg.d_model, nhead=cfg.nhead,
                                                       dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
                                                       num_layers=cfg.num_layers))
    return RecurrentForecaster(ModelConfig(input_size=input_size, cell=cfg.cell, hidden_size=cfg.hidden_size,
                                           num_layers=cfg.num_layers, dropout=cfg.dropout, n_ports=n_ports,
                                           port_embedding_dim=cfg.port_embedding_dim,
                                           residual_ewma_alpha=cfg.residual_ewma_alpha, dwell_channel=dwell_channel))


def sample_weights(cfg: TrainConfig, split: data.SplitData) -> np.ndarray:
    """Per-sample loss weights. 'hours': port dwell variance from the saved scaler, normalised to mean 1 over
    the training samples, so weighted scaled-MSE = MSE in hours / mean variance."""
    if cfg.loss_weighting == "none":
        return np.ones(len(split.y), np.float32)
    if cfg.loss_weighting == "hours":
        var = data.dwell_std_by_port_index() ** 2
        w = var[split.port_index]
        return (w / w.mean()).astype(np.float32)
    raise ValueError(f"unknown loss_weighting {cfg.loss_weighting!r}")


def mse_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if pred.shape != target.shape or (weight is not None and weight.shape != target.shape):
        raise ValueError(f"prediction {tuple(pred.shape)} and target {tuple(target.shape)} shapes differ (would broadcast)")
    sq = (pred - target) ** 2
    return torch.mean(sq if weight is None else weight * sq)


@torch.no_grad()
def predict(model: RecurrentForecaster, X: np.ndarray, port_index: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Scaled predictions in stored order."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size])
        pb = torch.from_numpy(port_index[i:i + batch_size])
        out.append(model(xb, pb).numpy())
    return np.concatenate(out)


def rmse_hours(model: RecurrentForecaster, split: data.SplitData) -> float:
    yh = data.to_hours(predict(model, split.X, split.port_index), split.port_index)
    return float(np.sqrt(np.mean((yh - split.y_hours) ** 2)))


@dataclass
class RunResult:
    config: TrainConfig
    history: list[dict]
    best_epoch: int
    best_val_rmse_h: float
    best_state: dict
    channels: list[str]
    n_parameters: int
    seconds: float


def train(cfg: TrainConfig, train_split: data.SplitData | None = None, val_split: data.SplitData | None = None,
          max_epochs: int | None = None) -> RunResult:
    set_seed(cfg.seed, cfg.num_threads)
    tr = train_split or data.load("train", cfg.mask_channels, cfg.drop_features)
    va = val_split or data.load("val", cfg.mask_channels, cfg.drop_features)
    if tr.channels != va.channels:
        raise ValueError("train and validation channels differ")
    if "dwell" not in tr.channels:
        raise ValueError("the dwell channel is required")
    model = build_model(cfg, input_size=tr.X.shape[-1], dwell_channel=tr.channels.index("dwell"))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)
    g = torch.Generator().manual_seed(cfg.seed)
    w = torch.from_numpy(sample_weights(cfg, tr))
    loader = DataLoader(TensorDataset(*tr.tensors(), w), batch_size=cfg.batch_size, shuffle=True, generator=g, num_workers=0)

    history, best, best_epoch, best_state, stale = [], float("inf"), -1, None, 0
    t0 = time.time()
    for epoch in range(1, (max_epochs or cfg.max_epochs) + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb, pb, wb in loader:
            opt.zero_grad()
            loss = mse_loss(model(xb, pb), yb, wb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        val_pred = predict(model, va.X, va.port_index)
        val_loss = float(np.mean((val_pred - va.y) ** 2))
        val_rmse = float(np.sqrt(np.mean((data.to_hours(val_pred, va.port_index) - va.y_hours) ** 2)))
        train_rmse = rmse_hours(model, tr)
        lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_objective": total / n, "train_rmse_h": train_rmse,
                        "val_loss_scaled_mse": val_loss, "val_rmse_h": val_rmse, "lr": lr})
        sched.step(val_rmse)
        if val_rmse < best - 1e-6:
            best, best_epoch, stale = val_rmse, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    return RunResult(config=cfg, history=history, best_epoch=best_epoch, best_val_rmse_h=best, best_state=best_state,
                     channels=tr.channels, n_parameters=model.n_parameters(), seconds=time.time() - t0)


def restore(result: RunResult) -> RecurrentForecaster:
    model = build_model(result.config, input_size=len(result.channels), dwell_channel=result.channels.index("dwell"))
    model.load_state_dict(result.best_state)
    model.eval()
    return model
