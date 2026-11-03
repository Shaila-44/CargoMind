"""Seeded, deterministic training of one configuration on one seed.

* Loss: MSE on the scaled target (shapes asserted equal: [batch] vs [batch]).
* Model selection / early stopping: validation RMSE in hours (via the dataset's inverse transform).
* Training windows are shuffled within the training split only (seeded generator); validation is
  evaluated in stored order. The test split is never loaded here.
"""
# =============================================================================
# WHAT HAPPENS DURING DEEP LEARNING MODEL TRAINING?
# =============================================================================
# Training a neural network is an iterative cycle of practice, feedback, and correction:
#
# In each "Epoch" (one full pass through all training data):
#   1. BATCHING: We take small groups of shipments (e.g. 64 windows at a time).
#   2. FORWARD PASS: The model makes a guess on what the next hour's dwell time will be.
#   3. LOSS (ERROR) CALCULATION: We compare the guess to reality using Mean Squared
#      Error (MSE). Larger mistakes get penalised more heavily.
#   4. BACKPROPAGATION: PyTorch calculates how much each neural weight contributed to the
#      mistake (calculating gradients).
#   5. GRADIENT CLIPPING: We cap the updates to prevent numerical explosions.
#   6. OPTIMIZER (ADAM): Gently nudges all weights in the direction of smaller error.
#   7. VALIDATION EVALUATION: At the end of each epoch, we test the model on held-out
#      validation data, converting predictions back into REAL HOURS (RMSE).
#   8. EARLY STOPPING: If validation error doesn't improve for `patience` (e.g. 15) epochs,
#      we stop training automatically to prevent overfitting and keep the best checkpoint!
# =============================================================================

from __future__ import annotations

# --- Standard library imports ---
import random  # Python's built-in random number generator
import time    # For measuring training speed (seconds per epoch)
from dataclasses import dataclass

# --- Third-party library imports ---
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --- Internal model imports ---
from dl.models import ModelConfig, RecurrentForecaster, TransformerConfig, TransformerForecaster

from . import data
from .config import TrainConfig


def set_seed(seed: int, num_threads: int) -> None:
    """Lock down all sources of randomness for 100% scientific reproducibility.

    Ensures that running the training script twice produces the exact same numbers,
    weight updates, and final metrics every single time.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)


def build_model(
    cfg: TrainConfig,
    input_size: int,
    n_ports: int = 10,
    dwell_channel: int = 0,
) -> RecurrentForecaster | TransformerForecaster:
    """Factory function: Instantiates either a Transformer or Recurrent (LSTM/GRU) forecaster."""
    if cfg.cell == "transformer":
        # Transformer architecture
        if cfg.port_embedding_dim or cfg.residual_ewma_alpha is not None:
            raise ValueError("the Transformer has no port embedding or residual head")
        return TransformerForecaster(
            TransformerConfig(
                input_size=input_size,
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                num_layers=cfg.num_layers,
            )
        )

    # Recurrent architecture (LSTM or GRU)
    return RecurrentForecaster(
        ModelConfig(
            input_size=input_size,
            cell=cfg.cell,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            n_ports=n_ports,
            port_embedding_dim=cfg.port_embedding_dim,
            residual_ewma_alpha=cfg.residual_ewma_alpha,
            dwell_channel=dwell_channel,
        )
    )


def sample_weights(cfg: TrainConfig, split: data.SplitData) -> np.ndarray:
    """Calculate sample loss weights so the training objective matches real-world hours.

    Why this matters:
    Because targets are scaled per-port into z-scores, a 1-unit z-score error at a port with
    high variance represents many more actual delay hours than at a low-variance port.
    Weighting samples by each port's historical dwell variance aligns the loss objective
    with minimizing real-world delay hours!
    """
    if cfg.loss_weighting == "none":
        return np.ones(len(split.y), np.float32)
    if cfg.loss_weighting == "hours":
        var = data.dwell_std_by_port_index() ** 2
        w = var[split.port_index]
        # Normalise so the average weight is 1.0
        return (w / w.mean()).astype(np.float32)
    raise ValueError(f"unknown loss_weighting {cfg.loss_weighting!r}")


def mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted Mean Squared Error (MSE) loss function.

    Formula: Loss = Mean( weight * (prediction - target)^2 )
    """
    if pred.shape != target.shape or (weight is not None and weight.shape != target.shape):
        raise ValueError(f"prediction {tuple(pred.shape)} and target {tuple(target.shape)} shapes differ (would broadcast)")
    sq = (pred - target) ** 2
    return torch.mean(sq if weight is None else weight * sq)


@torch.no_grad()
def predict(
    model: RecurrentForecaster | TransformerForecaster,
    X: np.ndarray,
    port_index: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    """Generate model predictions over an entire dataset split in efficient batches."""
    model.eval()  # Put into evaluation mode (disables dropout)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size])
        pb = torch.from_numpy(port_index[i:i + batch_size])
        out.append(model(xb, pb).numpy())
    return np.concatenate(out)


def rmse_hours(model: RecurrentForecaster | TransformerForecaster, split: data.SplitData) -> float:
    """Evaluate model performance in Root Mean Squared Error (RMSE) in ACTUAL HOURS.

    Converts model z-scores back to real hours and computes: sqrt(mean((pred_hours - actual_hours)^2)).
    """
    yh = data.to_hours(predict(model, split.X, split.port_index), split.port_index)
    return float(np.sqrt(np.mean((yh - split.y_hours) ** 2)))


@dataclass
class RunResult:
    """Container holding the complete outcome of a training experiment."""
    config: TrainConfig
    history: list[dict]
    best_epoch: int
    best_val_rmse_h: float
    best_state: dict
    channels: list[str]
    n_parameters: int
    seconds: float


def train(
    cfg: TrainConfig,
    train_split: data.SplitData | None = None,
    val_split: data.SplitData | None = None,
    max_epochs: int | None = None,
) -> RunResult:
    """Main training loop for Deep Learning forecasters.

    Coordinates:
      - Deterministic seed initialization.
      - Loading train and validation data.
      - Adam optimizer and ReduceLROnPlateau learning rate scheduler.
      - Epoch-by-epoch training and validation logging.
      - Early stopping and saving the best checkpoint.
    """
    # Step 1: Set random seed
    set_seed(cfg.seed, cfg.num_threads)

    # Step 2: Load data splits (test split is strictly NEVER loaded here)
    tr = train_split or data.load("train", cfg.mask_channels, cfg.drop_features)
    va = val_split or data.load("val", cfg.mask_channels, cfg.drop_features)
    if tr.channels != va.channels:
        raise ValueError("train and validation channels differ")
    if "dwell" not in tr.channels:
        raise ValueError("the dwell channel is required")

    # Step 3: Instantiate model architecture
    model = build_model(cfg, input_size=tr.X.shape[-1], dwell_channel=tr.channels.index("dwell"))

    # Step 4: Setup Adam optimizer and learning rate decay scheduler
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr
    )

    # Step 5: Prepare PyTorch DataLoader
    g = torch.Generator().manual_seed(cfg.seed)
    w = torch.from_numpy(sample_weights(cfg, tr))
    loader = DataLoader(
        TensorDataset(*tr.tensors(), w),
        batch_size=cfg.batch_size,
        shuffle=True,  # Shuffle training windows to break temporal correlation across batches
        generator=g,
        num_workers=0,
    )

    # Step 6: Training loop with early stopping tracking
    history, best, best_epoch, best_state, stale = [], float("inf"), -1, None, 0
    t0 = time.time()

    for epoch in range(1, (max_epochs or cfg.max_epochs) + 1):
        model.train()  # Enable training mode (dropout active)
        total, n = 0.0, 0

        # Iterate over mini-batches
        for xb, yb, pb, wb in loader:
            opt.zero_grad()                         # Reset gradients
            loss = mse_loss(model(xb, pb), yb, wb)  # Forward pass + weighted loss
            loss.backward()                         # Backpropagation
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip) # Prevent gradient explosions
            opt.step()                              # Update weights
            total += loss.item() * len(yb)
            n += len(yb)

        # Validation evaluation at end of epoch
        val_pred = predict(model, va.X, va.port_index)
        val_loss = float(np.mean((val_pred - va.y) ** 2))
        val_rmse = float(np.sqrt(np.mean((data.to_hours(val_pred, va.port_index) - va.y_hours) ** 2)))
        train_rmse = rmse_hours(model, tr)
        lr = opt.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_objective": total / n,
            "train_rmse_h": train_rmse,
            "val_loss_scaled_mse": val_loss,
            "val_rmse_h": val_rmse,
            "lr": lr,
        })

        # Step learning rate scheduler based on validation RMSE
        sched.step(val_rmse)

        # Check for improvement (early stopping check)
        if val_rmse < best - 1e-6:
            best, best_epoch, stale = val_rmse, epoch, 0
            # Clone best weights
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                # Early stopping triggered: model stopped improving
                break

    return RunResult(
        config=cfg,
        history=history,
        best_epoch=best_epoch,
        best_val_rmse_h=best,
        best_state=best_state,
        channels=tr.channels,
        n_parameters=model.n_parameters(),
        seconds=time.time() - t0,
    )


def restore(result: RunResult) -> RecurrentForecaster | TransformerForecaster:
    """Rebuild a model and restore its best saved weights from a training RunResult."""
    model = build_model(result.config, input_size=len(result.channels), dwell_channel=result.channels.index("dwell"))
    model.load_state_dict(result.best_state)
    model.eval()
    return model
