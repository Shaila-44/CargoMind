"""Small Transformer encoder for next-hour dwell regression.

Input  x: [batch, 12, channels] (scaled, as produced by cargomind_dl.dataset.load_split)
Output    [batch]                (next-hour dwell, per-port z-score)

Linear(channels -> d_model) + fixed sinusoidal positional encoding (positions 0..11, oldest -> newest)
-> one TransformerEncoderLayer -> representation at the last position (hour t) -> Linear(d_model, 1).

Attention runs over the 12 input hours only. Every input hour is at or before t and the target hour
t+1 is never part of the window, so no mask is needed to keep future information out.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class TransformerConfig:
    input_size: int = 8
    seq_len: int = 12
    d_model: int = 32
    nhead: int = 4
    dim_feedforward: int = 64
    dropout: float = 0.1
    num_layers: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """[seq_len, d_model] fixed encoding (Vaswani et al. 2017): sin on even, cos on odd dimensions."""
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TransformerForecaster(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        if cfg.d_model % cfg.nhead:
            raise ValueError(f"d_model {cfg.d_model} must be divisible by nhead {cfg.nhead}")
        self.cfg = cfg
        self.input_projection = nn.Linear(cfg.input_size, cfg.d_model)
        self.register_buffer("positional_encoding", sinusoidal_encoding(cfg.seq_len, cfg.d_model))
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.nhead, dim_feedforward=cfg.dim_feedforward,
                                           dropout=cfg.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor, port_index: torch.Tensor | None = None) -> torch.Tensor:
        """`port_index` is accepted for interface compatibility with the trainer and ignored (no port embedding)."""
        if x.dim() != 3 or x.shape[1] != self.cfg.seq_len or x.shape[2] != self.cfg.input_size:
            raise ValueError(f"expected x of shape [batch, {self.cfg.seq_len}, {self.cfg.input_size}], got {tuple(x.shape)}")
        h = self.input_projection(x) + self.positional_encoding
        h = self.encoder(h)
        return self.head(h[:, -1]).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
