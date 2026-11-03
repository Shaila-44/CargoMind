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
# =============================================================================
# WHAT IS A GRAPH NEURAL NETWORK (GNN) FOR PORTS?
# =============================================================================
# Imagine a global network of maritime ports (Singapore, Rotterdam, Shanghai, LA, etc.):
# Ports do not exist in isolation. They are connected by active shipping lanes.
# If a massive typhoon or union strike hits Singapore, ships cannot dock or unload.
# Days later, those delayed container ships cause a sudden ripple effect of congestion
# at their destination ports (like Shanghai or Los Angeles).
#
# A Graph Neural Network (GNN) models this interconnected reality:
#   - NODES (dots): The individual ports.
#   - EDGES (lines): Direct maritime shipping routes connecting two ports.
#
# How "Message Passing" Works:
# In each hour, every port shares a "status update" (congestion, weather, dwell time)
# with all its connected neighbors. Each port updates its own status by combining
# its own information with the messages received from its neighbors.
#
# COMBINING GNN + LSTM:
#   1. Spatial Stage (GCN): Understands how congestion spreads ACROSS ports.
#   2. Temporal Stage (LSTM): Understands how conditions evolve OVER TIME (across 12 hours).
#   3. Prediction: Forecasts container dwell time for every port in the next hour.
# =============================================================================

from __future__ import annotations

# --- Standard library imports ---
# dataclass allows us to create clean configuration containers with default values
from dataclasses import asdict, dataclass, field

# --- PyTorch imports ---
import torch
from torch import nn  # Neural network building blocks (Linear, LSTM, Dropout, etc.)


def normalized_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Compute the symmetrically normalized graph adjacency matrix with self-loops.

    Mathematical Formula:
        A_norm = D^(-1/2) * (A + I) * D^(-1/2)

    WHY DO WE NEED THIS NORMALIZATION?
    Imagine Port A has only 1 connected route, while Port B is a massive global hub
    connected to 20 routes.
    If we just sum up neighbor messages:
      - Port B would receive 20 times more signal than Port A, causing numbers to explode!
      - A port might completely forget its own local conditions.

    The math solves both problems:
      1. (A + I): Adds an "identity matrix" (self-loop), meaning every port sends a
         message to ITSELF, ensuring it remembers its own local weather and queues.
      2. D^(-1/2) ... D^(-1/2): Scales neighbor messages based on how connected the
         ports are. Highly connected hubs divide incoming signals so values stay stable.
    """
    a = adjacency.to(torch.float32)

    # Sanity checks: adjacency matrix must be square (e.g. 10x10 for 10 ports) and symmetric
    if a.dim() != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adjacency must be square, got {tuple(a.shape)}")
    if not torch.equal(a, a.T):
        raise ValueError("adjacency must be symmetric for this undirected GCN")

    a_hat = a.clone()
    # Step 1: Add self-loops by putting 1.0 along the diagonal (each port connects to itself)
    a_hat.fill_diagonal_(1.0)

    # Step 2: Compute D^(-1/2), the inverse square root of node degrees (row sums)
    d_inv_sqrt = a_hat.sum(1).pow(-0.5)

    # Step 3: Symmetrically scale rows and columns: D^(-1/2) * A_hat * D^(-1/2)
    return d_inv_sqrt[:, None] * a_hat * d_inv_sqrt[None, :]


class GraphConv(nn.Module):
    """A single Graph Convolutional Network (GCN) layer.

    Formula: Output = A_norm @ (X * W + b)

    How it works:
      1. Transforms each port's raw feature channels through a Linear layer (X * W).
      2. Multiplies by the normalized adjacency matrix (A_norm), which passes messages
         between neighboring ports and blends them together.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Linear layer that transforms raw channel numbers into hidden feature representations
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        # Message passing step: matrix multiplication aggregates neighbor representations
        return torch.matmul(a_norm, self.linear(x))


@dataclass
class GNNConfig:
    """Configuration container holding all hyperparameters for the GCN+LSTM model."""
    adjacency: list = field(default_factory=list)  # 2D grid [N x N] of 1s (connected) and 0s (not connected)
    ports: list = field(default_factory=list)      # Ordered list of port IDs (e.g., ['SGSIN', 'NLRTM', ...])
    input_size: int = 8                            # Number of sensor/telemetry channels per port per hour
    seq_len: int = 12                              # Lookback window length in hours (past 12 hours)
    gcn_hidden: int = 32                           # Size of the hidden representation after graph convolution
    lstm_hidden: int = 32                          # Number of memory units in the temporal LSTM
    dropout: float = 0.1                           # Dropout rate (randomly turns off 10% of neurons to prevent overfitting)

    def to_dict(self) -> dict:
        return asdict(self)


class GCNLSTMForecaster(nn.Module):
    """The complete hybrid Spatio-Temporal Graph Forecaster.

    Architecture pipeline:
      [Batch, 12 Hours, 10 Ports, 8 Features]
         │
         ▼ (Step 1: Spatial Message Passing)
      GraphConv + ReLU: Ports share information with their shipping route neighbors
         │
         ▼ (Step 2: Reshape for Time Series)
      Group data per port across the 12-hour window
         │
         ▼ (Step 3: Temporal Sequence Modeling)
      LSTM: Models how each port's conditions evolved hour-by-hour
         │
         ▼ (Step 4: Output Prediction Head)
      Linear Head: Predicts the next-hour container dwell time for every port simultaneously!
         │
         ▼
      [Batch, 10 Ports] (predicted dwell times)
    """

    def __init__(self, cfg: GNNConfig):
        super().__init__()
        adjacency = torch.tensor(cfg.adjacency, dtype=torch.float32)
        if len(cfg.ports) != adjacency.shape[0]:
            raise ValueError("ports and adjacency size differ")
        self.cfg = cfg

        # register_buffer saves tensors as part of the model state without treating them as trainable weights
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("a_norm", normalized_adjacency(adjacency))

        # Spatial component: GCN message passing layer
        self.gcn = GraphConv(cfg.input_size, cfg.gcn_hidden)

        # Temporal component: Long Short-Term Memory (LSTM) network
        self.lstm = nn.LSTM(cfg.gcn_hidden, cfg.lstm_hidden, batch_first=True)

        # Regularisation: Dropout randomly deactivates neurons during training
        self.dropout = nn.Dropout(cfg.dropout)

        # Prediction head: Maps LSTM hidden state (32 numbers) to 1 final dwell forecast
        self.head = nn.Linear(cfg.lstm_hidden, 1)

    @property
    def n_nodes(self) -> int:
        """Returns the number of ports (nodes) in the graph."""
        return len(self.cfg.ports)

    def forward(self, x: torch.Tensor, port_index: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass making predictions for all ports.

        Input shape:
          x: [Batch, Time (12), Nodes (10), Channels (8)]

        Output shape:
          [Batch, Nodes (10)] -> Next-hour predicted dwell for each port.
        """
        # Validate that the input tensor matches our expected dimensions
        expected = (self.cfg.seq_len, self.n_nodes, self.cfg.input_size)
        if x.dim() != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"expected x of shape [batch, {expected[0]}, {expected[1]}, {expected[2]}], got {tuple(x.shape)}")

        b, t, n, _ = x.shape

        # Step 1: Spatial convolution across the graph at every hour
        # h shape: [Batch, 12 Hours, 10 Nodes, 32 Hidden]
        h = torch.relu(self.gcn(x, self.a_norm))

        # Step 2: Rearrange dimensions so each port's 12-hour history is a contiguous sequence
        # Permute to [Batch, Nodes, Time, Hidden] -> Reshape to [Batch * Nodes, 12 Hours, 32 Hidden]
        h = h.permute(0, 2, 1, 3).reshape(b * n, t, self.cfg.gcn_hidden)

        # Step 3: Run the temporal LSTM over the 12 hours
        # h_n contains the final summary state after seeing all 12 hours
        _, (h_n, _) = self.lstm(h)

        # Step 4: Pass through dropout and linear head to get 1 predicted number per port
        # Reshape back to [Batch, Nodes]
        return self.head(self.dropout(h_n[-1])).view(b, n)

    def n_parameters(self) -> int:
        """Counts the total number of trainable weights in the model."""
        return sum(p.numel() for p in self.parameters())
