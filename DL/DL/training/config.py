"""Training / experiment configuration. Every experiment is a TrainConfig; ablations change one field."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from cargomind_dl import config as data_config

DL_ROOT = Path(__file__).resolve().parents[1]  # DL/DL (modelling stage)
ARTIFACTS = DL_ROOT / "artifacts"  # checkpoints, selection, experiment_results.csv (training records)
LOGS = DL_ROOT / "training" / "logs"
REPORTS = DL_ROOT / "reports"  # training report
DATASET = data_config.DATASET  # DL/Dataset/processed
EVALUATION = data_config.EVALUATION_STAGE  # DL/Evaluation
EVAL_RESULTS = EVALUATION / "results"  # evaluation tables / predictions
EVAL_FIGURES = EVALUATION / "figures"

SEEDS = (0, 1, 2, 3, 4)
SELECTION_TOLERANCE_H = 0.05  # validation RMSE differences below this are treated as ties (simpler model wins)


@dataclass(frozen=True)
class TrainConfig:
    name: str = "A_lstm"
    # model
    cell: str = "lstm"
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    port_embedding_dim: int = 0
    residual_ewma_alpha: float | None = None  # output = EWMA(window dwell) + learned correction
    # transformer only (cell="transformer"); hidden_size / num_layers above apply to recurrent cells
    d_model: int = 32
    nhead: int = 4
    dim_feedforward: int = 64
    # loss: "none" = plain MSE on the per-port z-scored target; "hours" = the same MSE weighted per sample by the
    # port's dwell variance (from scalers.json), which equals MSE in hours up to a constant - the selection metric
    loss_weighting: str = "none"
    # inputs (names from the loader; see cargomind_dl.dataset.load_split)
    mask_channels: tuple[str, ...] = ("dwell",)  # appends dwell_reconstructed
    drop_features: tuple[str, ...] = ()
    # optimisation
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 150
    patience: int = 15  # early stopping on validation RMSE (hours)
    grad_clip: float = 1.0
    # ReduceLROnPlateau on validation RMSE: the target is noisy, so a fixed LR tends to oscillate near the optimum
    lr_factor: float = 0.5
    lr_patience: int = 5
    min_lr: float = 1e-5
    num_threads: int = 4
    seed: int = 0
    notes: str = ""

    def with_seed(self, seed: int) -> "TrainConfig":
        return replace(self, seed=seed)

    def to_dict(self) -> dict:
        return asdict(self)


A = TrainConfig(name="A_lstm", notes="LSTM, 8 channels, no port embedding")
B = replace(A, name="B_lstm_port_embedding", port_embedding_dim=4, notes="A + 4-d port embedding")

# Round 2 (after round-1 validation showed scaled MSE and hour RMSE disagree; see training report)
A_W = replace(A, name="A_w_lstm_hour_weighted", loss_weighting="hours", notes="A with hour-weighted loss")
B_W = replace(B, name="B_w_lstm_port_embedding_hour_weighted", loss_weighting="hours", notes="B with hour-weighted loss")
C_W = replace(A_W, name="C_w_lstm_ewma_residual_hour_weighted", residual_ewma_alpha=0.2,
              notes="A_w predicting a correction to the window EWMA (alpha 0.2, the dataset baseline's definition)")
# simplicity order used by the selection rule (earlier = simpler)
CANDIDATES = (A, A_W, B, B_W, C_W)

# GRU challenger: hidden 37 gives a parameter count comparable to the 32-unit LSTM (see report)
GRU = replace(A, name="GRU_challenger", cell="gru", hidden_size=37, notes="GRU with a comparable parameter budget")


# Transformer (dl/models/transformer.py): one encoder layer, d_model 32, 4 heads, ff 64, dropout 0.1, no port embedding.
# T matches the basic LSTM protocol (plain MSE on the scaled target); T_w matches the GRU / C_w protocol (the same MSE
# weighted to equal MSE in hours), so the architecture is compared under both existing training protocols.
T = replace(A, name="T_transformer", cell="transformer", notes="Transformer, plain scaled MSE (protocol of A)")
T_W = replace(T, name="T_w_transformer_hour_weighted", loss_weighting="hours", notes="Transformer, hour-weighted MSE (protocol of A_w / C_w)")
TRANSFORMER_ARTIFACTS = ARTIFACTS / "transformer"
TRANSFORMER_LOGS = LOGS / "transformer"


def gru_like(base: TrainConfig) -> TrainConfig:
    """GRU challenger sharing every setting of the selected configuration except the cell (hidden 37 ~ same parameters)."""
    return replace(base, name=f"GRU__{base.name}", cell="gru", hidden_size=37, notes=f"GRU version of {base.name}")


def ablations(base: TrainConfig) -> list[TrainConfig]:
    """One factor at a time, relative to the selected configuration."""
    out = [
        replace(base, name=f"{base.name}__no_labour_strike", drop_features=("labour_strike",), notes="drop labour_strike"),
        replace(base, name=f"{base.name}__no_dwell_reconstructed", mask_channels=(), notes="drop dwell_reconstructed indicator"),
        replace(base, name=f"{base.name}__no_vessels_at_anchor", drop_features=("vessels_at_anchor",), notes="drop vessels_at_anchor"),
        # beyond the required minimum: the EDA's strongest partial-correlation signals, and history alone
        replace(base, name=f"{base.name}__no_utilization", drop_features=("utilization",), notes="drop utilization"),
        replace(base, name=f"{base.name}__no_disruption", drop_features=("disruption_active", "disruption_hours_elapsed"),
                notes="drop both disruption channels (one factor: disruption information)"),
        replace(base, name=f"{base.name}__no_wind_gust", drop_features=("wind_gust",), notes="drop wind_gust"),
        replace(base, name=f"{base.name}__dwell_history_only", mask_channels=(),
                drop_features=("utilization", "labour_strike", "wind_gust", "disruption_active", "disruption_hours_elapsed", "vessels_at_anchor"),
                notes="dwell history only (all operational channels removed)"),
    ]
    return out


EXPERIMENTS = {c.name: c for c in CANDIDATES + (GRU,)}
__all__ = ["TrainConfig", "A", "B", "GRU", "ablations", "SEEDS", "field"]
