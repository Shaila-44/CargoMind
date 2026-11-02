"""Small recurrent regressors for next-hour dwell forecasting.

Input  x: [batch, 12, channels] (scaled, as produced by cargomind_dl.dataset.load_split)
Output    [batch]                (next-hour dwell, per-port z-score)

Sequence representation: the last layer's final hidden state h_T. With 12 steps and a
one-step-ahead target, the state after the most recent hour is the natural summary; no
attention or pooling is added without evidence that it helps.
An optional port embedding is concatenated to h_T before the linear head.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_size: int = 8
    cell: str = "lstm"  # "lstm" | "gru"
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.1  # applied to the sequence representation before the head
    n_ports: int = 10
    port_embedding_dim: int = 0  # 0 = no port embedding
    # If set, the output is EWMA(dwell channel over the window, alpha) + learned correction. The EWMA is a
    # fixed function of the input (the dataset's EWMA baseline definition, initialised at the first hour);
    # the head starts at zero, so an untrained model reproduces the EWMA baseline exactly.
    residual_ewma_alpha: float | None = None
    dwell_channel: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def ewma_last(seq: torch.Tensor, alpha: float) -> torch.Tensor:
    """EWMA over the time axis of [batch, time], initialised at the first step; returns [batch]."""
    s = seq[:, 0]
    for j in range(1, seq.shape[1]):
        s = alpha * seq[:, j] + (1 - alpha) * s
    return s


class RecurrentForecaster(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.cell not in ("lstm", "gru"):
            raise ValueError(f"unknown cell {cfg.cell!r}")
        self.cfg = cfg
        rnn = nn.LSTM if cfg.cell == "lstm" else nn.GRU
        self.rnn = rnn(cfg.input_size, cfg.hidden_size, num_layers=cfg.num_layers, batch_first=True,
                       dropout=cfg.dropout if cfg.num_layers > 1 else 0.0)
        self.embedding = nn.Embedding(cfg.n_ports, cfg.port_embedding_dim) if cfg.port_embedding_dim > 0 else None
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size + cfg.port_embedding_dim, 1)
        if cfg.residual_ewma_alpha is not None:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, port_index: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.cfg.input_size:
            raise ValueError(f"expected x of shape [batch, time, {self.cfg.input_size}], got {tuple(x.shape)}")
        _, state = self.rnn(x)
        h = state[0] if self.cfg.cell == "lstm" else state  # [layers, batch, hidden]
        z = h[-1]
        if self.embedding is not None:
            if port_index is None:
                raise ValueError("this model has a port embedding; port_index is required")
            z = torch.cat([z, self.embedding(port_index.long())], dim=-1)
        out = self.head(self.dropout(z)).squeeze(-1)
        if self.cfg.residual_ewma_alpha is not None:
            out = out + ewma_last(x[..., self.cfg.dwell_channel], self.cfg.residual_ewma_alpha)
        return out

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
