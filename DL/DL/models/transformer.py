"""Small Transformer encoder for next-hour dwell regression.

Input  x: [batch, 12, channels] (scaled, as produced by cargomind_dl.dataset.load_split)
Output    [batch]                (next-hour dwell, per-port z-score)

Linear(channels -> d_model) + fixed sinusoidal positional encoding (positions 0..11, oldest -> newest)
-> one TransformerEncoderLayer -> representation at the last position (hour t) -> Linear(d_model, 1).

Attention runs over the 12 input hours only. Every input hour is at or before t and the target hour
t+1 is never part of the window, so no mask is needed to keep future information out.
"""
# =============================================================================
# WHAT IS A TRANSFORMER FOR TIME-SERIES FORECASTING?
# =============================================================================
# The Transformer architecture is the groundbreaking neural network design behind
# modern AI (like ChatGPT, Gemini, and BERT). While famous for language, it is also
# exceptionally powerful for time-series forecasting (like port dwell times).
#
# HOW DOES A TRANSFORMER DIFFER FROM AN LSTM?
# - An LSTM reads time steps one-by-one in order (Hour 1 -> Hour 2 -> Hour 3...).
#   If something important happened 12 hours ago, the LSTM has to carry that memory
#   through 11 intermediate steps without forgetting it.
# - A Transformer uses "SELF-ATTENTION": It looks at ALL 12 past hours at the exact
#   same time! It can directly connect Hour 12 with Hour 3 in a single leap, calculating
#   how much "attention" the current moment should pay to past events (e.g. a rainstorm
#   or shift change that happened 8 hours ago).
#
# WHAT IS POSITIONAL ENCODING?
# Because a Transformer processes all 12 hours in parallel, it doesn't naturally know
# which hour came first or last!
# To give the model a sense of time, we inject "Positional Encodings" — mathematically
# unique sine and cosine waves that act like timestamp stamps for each hour.
# =============================================================================

from __future__ import annotations

# --- Standard library imports ---
import math  # For mathematical constants (pi, logarithms, sine, cosine)
from dataclasses import asdict, dataclass

# --- PyTorch imports ---
import torch
from torch import nn  # Neural network modules (Linear, TransformerEncoder, etc.)


@dataclass
class TransformerConfig:
    """Hyperparameter configuration for the Transformer Forecaster."""
    input_size: int = 8          # Number of telemetry features per hour (dwell, queue, weather, etc.)
    seq_len: int = 12            # Number of historical hours in the input window
    d_model: int = 32            # Embedding dimension (width of the internal neural representation)
    nhead: int = 4               # Number of parallel attention heads (each head learns different patterns)
    dim_feedforward: int = 64    # Hidden dimension inside the feed-forward sub-layer
    dropout: float = 0.1         # Regularisation rate (randomly drops 10% of connections to prevent overfitting)
    num_layers: int = 1          # Number of stacked Transformer encoder layers

    def to_dict(self) -> dict:
        return asdict(self)


def sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Generate fixed sinusoidal positional encodings (Vaswani et al. 2017).

    Returns a tensor of shape [seq_len, d_model].

    WHY SINE AND COSINE WAVES?
    Every position from 0 to 11 receives a unique combination of frequencies:
      - Even feature dimensions use Sine waves: sin(pos / 10000^(2i / d_model))
      - Odd feature dimensions use Cosine waves: cos(pos / 10000^(2i / d_model))

    This creates a smooth, continuous geometric clock: the model can easily learn
    relative time distances ("this event happened 3 hours before that event")
    using simple linear dot products.
    """
    # pos: column vector [0, 1, 2, ..., 11] representing the hourly time index
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)

    # div: frequency scaling factors for each pair of dimensions
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

    # Initialise empty tensor for positional encodings
    pe = torch.zeros(seq_len, d_model)

    # Even columns get sine waves
    pe[:, 0::2] = torch.sin(pos * div)

    # Odd columns get cosine waves
    pe[:, 1::2] = torch.cos(pos * div)

    return pe


class TransformerForecaster(nn.Module):
    """The Transformer Encoder Forecaster for container dwell time.

    Flow of data through the model:
      Input: [Batch, 12 Hours, 8 Features]
         │
         ▼ (Step 1: Input Projection)
      Linear layer projects 8 raw features to d_model (32 numbers)
         │
         ▼ (Step 2: Add Positional Clocks)
      Adds fixed sinusoidal positional encoding so the model knows the time order
         │
         ▼ (Step 3: Self-Attention)
      TransformerEncoderLayer: Each hour computes attention weights across all other hours
         │
         ▼ (Step 4: Take the Latest Hour Representation)
      Selects the final time step (hour t, representing the current state)
         │
         ▼ (Step 5: Output Projection Head)
      Linear layer maps 32 features to 1 final number (next-hour predicted dwell z-score)
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        # Ensure d_model can be evenly divided among the attention heads
        if cfg.d_model % cfg.nhead:
            raise ValueError(f"d_model {cfg.d_model} must be divisible by nhead {cfg.nhead}")

        self.cfg = cfg

        # Step 1: Project the 8 raw telemetry features up to the internal model dimension (d_model=32)
        self.input_projection = nn.Linear(cfg.input_size, cfg.d_model)

        # Step 2: Store positional encoding table as a registered buffer (saved with weights, but not trained)
        self.register_buffer("positional_encoding", sinusoidal_encoding(cfg.seq_len, cfg.d_model))

        # Step 3: Self-attention encoder layer (Multi-Head Self-Attention + Feed-Forward Network + LayerNorm)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,  # Expects tensors formatted as [Batch, Sequence Length, Features]
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers, enable_nested_tensor=False)

        # Step 4: Final output head mapping the 32-dimensional summary to 1 predicted dwell value
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor, port_index: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass for dwell forecasting.

        Parameters:
          x: Input tensor of shape [Batch, 12, 8]
          port_index: Optional port identifier (accepted for compatibility with other models)

        Returns:
          Tensor of shape [Batch] with predicted next-hour dwell times (per-port z-score).
        """
        # Validate dimensions
        if x.dim() != 3 or x.shape[1] != self.cfg.seq_len or x.shape[2] != self.cfg.input_size:
            raise ValueError(f"expected x of shape [batch, {self.cfg.seq_len}, {self.cfg.input_size}], got {tuple(x.shape)}")

        # Step 1 & 2: Project inputs and add the positional clock
        # h shape: [Batch, 12, 32]
        h = self.input_projection(x) + self.positional_encoding

        # Step 3: Run self-attention through the Transformer encoder
        h = self.encoder(h)

        # Step 4 & 5: Take the representation at the very last position (hour t, index -1),
        # pass it through the head, and squeeze away the last dimension to get [Batch]
        return self.head(h[:, -1]).squeeze(-1)

    def n_parameters(self) -> int:
        """Count total trainable parameters in the network."""
        return sum(p.numel() for p in self.parameters())
