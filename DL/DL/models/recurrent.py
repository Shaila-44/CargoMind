"""Small recurrent regressors for next-hour dwell forecasting.

Input  x: [batch, 12, channels] (scaled, as produced by cargomind_dl.dataset.load_split)
Output    [batch]                (next-hour dwell, per-port z-score)

Sequence representation: the last layer's final hidden state h_T. With 12 steps and a
one-step-ahead target, the state after the most recent hour is the natural summary; no
attention or pooling is added without evidence that it helps.
An optional port embedding is concatenated to h_T before the linear head.
"""
# =============================================================================
# WHAT ARE RECURRENT NEURAL NETWORKS (LSTM & GRU)?
# =============================================================================
# Recurrent Neural Networks (RNNs) are designed specifically for sequential data
# like time series (hourly port measurements) or sentences (words in order).
#
# Unlike standard feed-forward networks (which treat every input as independent),
# an RNN processes inputs step-by-step:
#   At Hour 1: Reads Hour 1 data, updates its internal "memory" (hidden state).
#   At Hour 2: Reads Hour 2 data, combines it with the memory from Hour 1.
#   ...
#   At Hour 12: Holds a rich accumulated memory representing the whole 12-hour period!
#
# TWO ARCHITECTURES SUPPORTED HERE:
# 1. LSTM (Long Short-Term Memory): Uses 3 specialised "gates" (Forget gate, Input gate,
#    Output gate) to selectively keep or erase long-term memories without vanishing gradients.
# 2. GRU (Gated Recurrent Unit): A slightly simpler, faster alternative to LSTM with
#    2 gates (Reset and Update).
#
# WHAT IS THE EWMA RESIDUAL SHORTCUT?
# In port operations, container dwell time doesn't change drastically out of nowhere.
# A simple Exponential Weighted Moving Average (EWMA) of recent hours is already a strong guess!
# If `residual_ewma_alpha` is enabled:
#   - We compute the EWMA mathematically.
#   - We initialise the neural head to zero.
#   - The neural network only has to learn the *adjustments* (residuals) needed to improve
#     upon the simple moving average!
# =============================================================================

from __future__ import annotations

# --- Standard library imports ---
from dataclasses import asdict, dataclass

# --- PyTorch imports ---
import torch
from torch import nn


@dataclass
class ModelConfig:
    """Configuration container holding all hyperparameters for the Recurrent Forecaster."""
    input_size: int = 8          # Number of telemetry features per hour
    cell: str = "lstm"           # Which recurrent cell to use: 'lstm' or 'gru'
    hidden_size: int = 32        # Number of memory units in the recurrent hidden state
    num_layers: int = 1          # Number of stacked recurrent layers (1 = single layer)
    dropout: float = 0.1         # Dropout probability to prevent overfitting
    n_ports: int = 10            # Total number of unique ports in the dataset
    port_embedding_dim: int = 0  # Size of port embedding vector (0 = no port embedding)

    # Optional EWMA residual connection settings:
    # If provided, the model predicts: EWMA baseline + learned neural correction
    residual_ewma_alpha: float | None = None
    dwell_channel: int = 0       # Which input channel index corresponds to container dwell time

    def to_dict(self) -> dict:
        return asdict(self)


def ewma_last(seq: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute the Exponentially Weighted Moving Average (EWMA) over a time sequence.

    Formula:
        State_t = alpha * Input_t + (1 - alpha) * State_(t-1)

    Parameters:
      seq: Tensor of shape [Batch, Time] containing past dwell values.
      alpha: Smoothing factor (e.g. 0.2). Higher alpha gives more weight to recent hours;
             lower alpha gives more weight to older history.

    Returns:
      Tensor of shape [Batch] with the smoothed moving average at the final hour.
    """
    s = seq[:, 0]  # Initialise state with the very first hour's value
    for j in range(1, seq.shape[1]):
        s = alpha * seq[:, j] + (1 - alpha) * s
    return s


class RecurrentForecaster(nn.Module):
    """The Recurrent Neural Network Forecaster (LSTM or GRU) for container dwell time.

    Flow of data:
      Input Sequence: [Batch, 12 Hours, 8 Features]
         │
         ▼ (Step 1: Recurrent Processing)
      LSTM or GRU processes the 12 hours sequentially, building a 32-dimensional memory
         │
         ▼ (Step 2: Port Embedding - Optional)
      Optionally attaches a learned vector representing the specific port's personality
         │
         ▼ (Step 3: Dropout Regularisation)
      Randomly zeros out connections to prevent memorisation
         │
         ▼ (Step 4: Output Linear Head)
      Calculates the raw prediction
         │
         ▼ (Step 5: Residual EWMA Baseline - Optional)
      Adds the moving average baseline if residual learning is turned on
         │
         ▼
      Final Forecast: [Batch]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.cell not in ("lstm", "gru"):
            raise ValueError(f"unknown cell {cfg.cell!r}")
        self.cfg = cfg

        # Select either standard LSTM or GRU class
        rnn = nn.LSTM if cfg.cell == "lstm" else nn.GRU
        self.rnn = rnn(
            cfg.input_size,
            cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,  # Formats inputs as [Batch, Time, Features]
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        # Optional Port Embedding: Allows the network to learn unique characteristics
        # for each specific port (e.g., Singapore has different operational rhythms than Rotterdam)
        self.embedding = nn.Embedding(cfg.n_ports, cfg.port_embedding_dim) if cfg.port_embedding_dim > 0 else None

        # Dropout regularisation to prevent overfitting
        self.dropout = nn.Dropout(cfg.dropout)

        # Final linear prediction head: Combines recurrent hidden state + port embedding
        head_in = cfg.hidden_size + cfg.port_embedding_dim
        self.head = nn.Linear(head_in, 1)

        # If residual EWMA is enabled, initialise weights and biases to 0.0!
        # This guarantees that at epoch 0 (before any training), the model outputs the exact
        # EWMA baseline, and only learns corrections on top of it.
        if cfg.residual_ewma_alpha is not None:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, port_index: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass for recurrent dwell forecasting.

        Parameters:
          x: Input tensor of shape [Batch, 12 Hours, 8 Features]
          port_index: Integer tensor [Batch] indicating which port each sample belongs to.
        """
        # Validate input dimensions
        if x.dim() != 3 or x.shape[-1] != self.cfg.input_size:
            raise ValueError(f"expected x of shape [batch, time, {self.cfg.input_size}], got {tuple(x.shape)}")

        # Step 1: Run through the recurrent network (LSTM or GRU)
        _, state = self.rnn(x)

        # For LSTM, state is a tuple: (hidden_state, cell_state). We want the hidden state.
        # For GRU, state is directly the hidden_state.
        h = state[0] if self.cfg.cell == "lstm" else state  # shape: [layers, batch, hidden]

        # Extract the hidden state from the topmost layer: [batch, hidden]
        z = h[-1]

        # Step 2: If port embeddings are enabled, concatenate the port vector to the hidden state
        if self.embedding is not None:
            if port_index is None:
                raise ValueError("this model has a port embedding; port_index is required")
            z = torch.cat([z, self.embedding(port_index.long())], dim=-1)

        # Step 3 & 4: Apply dropout and pass through linear head
        out = self.head(self.dropout(z)).squeeze(-1)

        # Step 5: If residual learning is enabled, add the EWMA baseline calculation
        if self.cfg.residual_ewma_alpha is not None:
            out = out + ewma_last(x[..., self.cfg.dwell_channel], self.cfg.residual_ewma_alpha)

        return out

    def n_parameters(self) -> int:
        """Count total trainable weights in the model."""
        return sum(p.numel() for p in self.parameters())
