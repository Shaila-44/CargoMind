"""CargoMind Stage 02 forecasting models (inference-safe: no training imports).

=============================================================================
CARGOMIND DEEP LEARNING MODELS PACKAGE (`dl.models`)
=============================================================================
This package provides the neural network architectures for predicting next-hour
container dwell time across ports.

SUPPORTED ARCHITECTURES:
------------------------
1. TransformerForecaster (`transformer.py`)
   Self-attention encoder with sinusoidal positional encodings. Processes all 12
   input hours simultaneously to discover long-range temporal interactions.

2. RecurrentForecaster (`recurrent.py`)
   LSTM and GRU sequential models with optional port embeddings and EWMA
   (Exponentially Weighted Moving Average) residual connections.

3. GCNLSTMForecaster (`gnn.py`)
   Spatio-temporal Graph Neural Network combining Graph Convolution (over maritime
   shipping lanes) with LSTM temporal sequence modeling.

CHECKPOINT MANAGEMENT:
----------------------
• load_checkpoint / save_checkpoint (`checkpoint.py`)
   Saves and restores trained model weights, accompanied by cryptographic SHA-256
   checksums to verify dataset consistency and prevent data corruption.
=============================================================================
"""
from .checkpoint import load_checkpoint, save_checkpoint
from .gnn import GCNLSTMForecaster, GNNConfig
from .recurrent import ModelConfig, RecurrentForecaster
from .transformer import TransformerConfig, TransformerForecaster

__all__ = [
    "ModelConfig",
    "RecurrentForecaster",
    "TransformerConfig",
    "TransformerForecaster",
    "GNNConfig",
    "GCNLSTMForecaster",
    "load_checkpoint",
    "save_checkpoint",
]
