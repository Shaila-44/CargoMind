"""Figures for the training / evaluation reports, built only from saved logs and result CSVs.

    .venv/bin/python -m dl.evaluation.report_figures
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dl.training.config import ARTIFACTS, EVAL_FIGURES, LOGS  # noqa: E402

FIG = EVAL_FIGURES
EWMA_VAL = 4.2964  # DL/Dataset/processed/baseline_metrics.csv, validation, EWMA alpha 0.2
COLORS = {"A_lstm": "#8a8985", "B_lstm_port_embedding": "#b07aa1", "A_w_lstm_hour_weighted": "#2a78d6",
          "B_w_lstm_port_embedding_hour_weighted": "#e34948", "C_w_lstm_ewma_residual_hour_weighted": "#1b9e77",
          "GRU__B_w_lstm_port_embedding_hour_weighted": "#eda100"}
SHORT = {"A_lstm": "A", "B_lstm_port_embedding": "B", "A_w_lstm_hour_weighted": "A_w", "B_w_lstm_port_embedding_hour_weighted": "B_w (selected)",
         "C_w_lstm_ewma_residual_hour_weighted": "C_w", "GRU__B_w_lstm_port_embedding_hour_weighted": "GRU"}


def curves() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), sharey="row")
    for col, exp in enumerate(["A_lstm", "B_w_lstm_port_embedding_hour_weighted", "C_w_lstm_ewma_residual_hour_weighted"]):
        for seed in range(5):
            h = pd.read_csv(LOGS / exp / f"seed_{seed}.csv")
            axes[0, col].plot(h["epoch"], h["val_rmse_h"], color=COLORS[exp], alpha=0.8, lw=1, label="validation" if seed == 0 else None)
            axes[0, col].plot(h["epoch"], h["train_rmse_h"], color="#0b0b0b", alpha=0.35, lw=0.8, ls="--", label="training" if seed == 0 else None)
            axes[1, col].plot(h["epoch"], h["val_loss_scaled_mse"], color=COLORS[exp], alpha=0.8, lw=1)
        axes[0, col].axhline(EWMA_VAL, color="#1b9e77", ls=":", lw=1.2, label="EWMA, validation")
        axes[0, col].set(title=f"{SHORT[exp]}: RMSE in hours (5 seeds)", xlabel="epoch", ylabel="RMSE (h)")
        axes[1, col].set(title=f"{SHORT[exp]}: validation loss, unweighted scaled MSE", xlabel="epoch", ylabel="MSE (z-units²)")
        axes[0, col].legend(fontsize=8)
    axes[0, 0].set_ylim(3.9, 5.0)
    fig.suptitle("Training curves. Round 1 (A): scaled loss falls while hour RMSE rises. Round 2 (B_w, C_w): hour-weighted loss", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=140)
    plt.close(fig)


def seeds_and_ablations(res: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1.4]})
    main = [e for e in COLORS if e in set(res["experiment"])]
    for i, e in enumerate(main):
        v = res.loc[res["experiment"] == e, "val_rmse_h"]
        axes[0].scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=COLORS[e], s=28, zorder=3)
        axes[0].hlines(v.mean(), i - 0.25, i + 0.25, color="#0b0b0b", lw=1.5)
    axes[0].axhline(EWMA_VAL, color="#1b9e77", ls=":", lw=1.2, label="EWMA (validation)")
    axes[0].set_xticks(range(len(main)), [SHORT[e] for e in main], rotation=20)
    axes[0].set(title="Best validation RMSE per seed (bar = mean of 5 seeds)", ylabel="validation RMSE (h)")
    axes[0].legend(fontsize=8)

    base = "B_w_lstm_port_embedding_hour_weighted"
    abl = [e for e in res["experiment"].unique() if e.startswith(base + "__")]
    g = res[res["experiment"].isin([base] + abl)].groupby("experiment")["val_rmse_h"].agg(["mean", "std"])
    ref = g.loc[base, "mean"]
    g = g.drop(base).assign(delta=lambda d: d["mean"] - ref).sort_values("delta")
    labels = [e.replace(base + "__", "").replace("_", " ") for e in g.index]
    axes[1].barh(labels, g["delta"], xerr=g["std"], color=["#2a78d6" if d < 0 else "#e34948" for d in g["delta"]], capsize=3)
    axes[1].axvline(0, color="#0b0b0b", lw=0.8)
    for x in (-0.05, 0.05):
        axes[1].axvline(x, color="#8a8985", ls="--", lw=0.8)
    axes[1].set(title="Ablations of B_w: change in mean validation RMSE (± seed SD); dashed = ±0.05 h tie band",
                xlabel="validation RMSE − B_w (h); negative = better without it")
    fig.tight_layout()
    fig.savefig(FIG / "seeds_and_ablations.png", dpi=140)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(ARTIFACTS / "experiment_results.csv")
    curves()
    seeds_and_ablations(res)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
