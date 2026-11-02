"""GCN + LSTM forecaster over the port graph (supplementary experiment; not the selected model).

Input  x: [batch, 12, nodes, channels]  (graph snapshots; see dl.training.graph_data)
Output    [batch, nodes]                (next-hour dwell per port, per-port z-score)

1. Graph convolution, applied at every hour with shared weights (Kipf & Welling 2017):
       H_t = ReLU( D^-1/2 (A + I) D^-1/2  X_t W )        X_t: [nodes, channels], W: [channels, gcn_hidden]
   Each port's representation becomes a degree-normalised sum of its own and its direct-route neighbours'
   transformed features: genuine one-hop message passing over a fixed adjacency (a registered buffer).
2. Temporal LSTM over the 12 hourly graph representations, run per node with shared weights.
3. Linear head on each node's final hidden state -> one prediction per port.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn


def normalized_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation with self-loops: D^-1/2 (A + I) D^-1/2, D = degree matrix of A + I."""
    a = adjacency.to(torch.float32)
    if a.dim() != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adjacency must be square, got {tuple(a.shape)}")
    if not torch.equal(a, a.T):
        raise ValueError("adjacency must be symmetric for this undirected GCN")
    a_hat = a.clone()
    a_hat.fill_diagonal_(1.0)  # self-loops (any given diagonal is replaced by 1)
    d_inv_sqrt = a_hat.sum(1).pow(-0.5)
    return d_inv_sqrt[:, None] * a_hat * d_inv_sqrt[None, :]


class GraphConv(nn.Module):
    """One GCN layer: A_norm @ (X W + b). Works on any leading batch/time dimensions: [..., nodes, in] -> [..., nodes, out]."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a_norm, self.linear(x))  # message passing: each node aggregates its neighbours


@dataclass
class GNNConfig:
    adjacency: list = field(default_factory=list)  # 0/1 symmetric [nodes x nodes], no self-loops
    ports: list = field(default_factory=list)      # node order (= dataset port_index order)
    input_size: int = 8
    seq_len: int = 12
    gcn_hidden: int = 32
    lstm_hidden: int = 32
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class GCNLSTMForecaster(nn.Module):
    def __init__(self, cfg: GNNConfig):
        super().__init__()
        adjacency = torch.tensor(cfg.adjacency, dtype=torch.float32)
        if len(cfg.ports) != adjacency.shape[0]:
            raise ValueError("ports and adjacency size differ")
        self.cfg = cfg
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("a_norm", normalized_adjacency(adjacency))
        self.gcn = GraphConv(cfg.input_size, cfg.gcn_hidden)
        self.lstm = nn.LSTM(cfg.gcn_hidden, cfg.lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.lstm_hidden, 1)

    @property
    def n_nodes(self) -> int:
        return len(self.cfg.ports)

    def forward(self, x: torch.Tensor, port_index: torch.Tensor | None = None) -> torch.Tensor:
        """`port_index` is ignored (node identity is the position in the graph); accepted for interface symmetry."""
        expected = (self.cfg.seq_len, self.n_nodes, self.cfg.input_size)
        if x.dim() != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"expected x of shape [batch, {expected[0]}, {expected[1]}, {expected[2]}], got {tuple(x.shape)}")
        b, t, n, _ = x.shape
        h = torch.relu(self.gcn(x, self.a_norm))                            # [B, T, N, gcn_hidden]
        h = h.permute(0, 2, 1, 3).reshape(b * n, t, self.cfg.gcn_hidden)   # one sequence per (sample, node)
        _, (h_n, _) = self.lstm(h)                                          # h_n: [1, B*N, lstm_hidden]
        return self.head(self.dropout(h_n[-1])).view(b, n)                  # [B, N]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
