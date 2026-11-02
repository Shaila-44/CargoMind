"""Validation-only comparison of the Transformer with the existing models and baselines.

    .venv/bin/python -m dl.evaluation.compare_transformer

Uses the existing evaluation context (frames.context) and metrics (metrics.mae / rmse / r2); adds mean bias and
two definitions of "extreme" dwell. The test split is not loaded. Outputs go to DL/Evaluation/results/transformer/ and
DL/Evaluation/figures/transformer/.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dl.training.config import ARTIFACTS, EVAL_FIGURES, EVAL_RESULTS, TRANSFORMER_ARTIFACTS, TRANSFORMER_LOGS  # noqa: E402

from . import frames, metrics  # noqa: E402
from .evaluate import model_predictions  # noqa: E402

EXPERIMENTS = {  # label -> checkpoint directory
    "LSTM (A, basic)": ARTIFACTS / "runs" / "A_lstm",
    "LSTM (A_w, hour-weighted)": ARTIFACTS / "runs" / "A_w_lstm_hour_weighted",
    "GRU (B_w settings)": ARTIFACTS / "runs" / "GRU__B_w_lstm_port_embedding_hour_weighted",
    "C_w (EWMA-residual LSTM)": ARTIFACTS / "runs" / "C_w_lstm_ewma_residual_hour_weighted",
    "Transformer (T)": TRANSFORMER_ARTIFACTS / "runs" / "T_transformer",
    "Transformer (T_w, hour-weighted)": TRANSFORMER_ARTIFACTS / "runs" / "T_w_transformer_hour_weighted",
}
PROTOCOL = {"LSTM (A, basic)": "plain MSE", "LSTM (A_w, hour-weighted)": "hour-weighted", "GRU (B_w settings)": "hour-weighted + port emb.",
            "C_w (EWMA-residual LSTM)": "hour-weighted + EWMA residual", "Transformer (T)": "plain MSE",
            "Transformer (T_w, hour-weighted)": "hour-weighted"}
FIG = EVAL_FIGURES / "transformer"
OUT = EVAL_RESULTS / "transformer"  # comparison tables (the checkpoints stay in TRANSFORMER_ARTIFACTS)
EWMA = "EWMA α=0.2"


def masks(f: pd.DataFrame) -> dict[str, np.ndarray]:
    m = {k: v for k, v in metrics.subgroups(f).items() if not k.startswith("port ")}
    # realised extremes: top decile of the actual target within each port (subject to regression to the mean)
    m["extreme: top-decile actual dwell (per port)"] = (f["y_hours"] >= f.groupby("port_id")["y_hours"].transform(lambda v: v.quantile(0.9))).values
    # known-at-t extremes: top decile of the window's EWMA level within each port (no selection on the outcome)
    lvl = f["baseline_ewma_alpha_0.2_hours"]
    m["extreme: top-decile window level (known at t, per port)"] = (lvl >= lvl.groupby(f["port_id"]).transform(lambda v: v.quantile(0.9))).values
    return m


def score(y, p) -> dict:
    return {"rmse_h": metrics.rmse(y, p), "mae_h": metrics.mae(y, p), "r2": metrics.r2(y, p), "bias_h": float(np.mean(p - y))}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    preds, frame = {}, None
    for label, d in EXPERIMENTS.items():
        for ck in sorted(d.glob("seed_*.pt")):
            p, sd = model_predictions(ck, "val")
            preds[(label, int(ck.stem.split("_")[1]))] = p
            if frame is None:
                frame = frames.context(sd.window_id, sd.y_hours)
    base = frames.baseline_predictions(frame)
    y = frame["y_hours"].values
    port = frame["port_id"].values
    ports = sorted(np.unique(port))

    seed_rows, port_rows, sub_rows = [], [], []
    for (label, seed), p in preds.items():
        seed_rows.append({"model": label, "seed": seed, **score(y, p), "port_balanced_rmse_h": metrics.port_balanced_rmse(y, p, port)})
    for name, p in base.items():
        seed_rows.append({"model": name, "seed": None, **score(y, p), "port_balanced_rmse_h": metrics.port_balanced_rmse(y, p, port)})
    seedwise = pd.DataFrame(seed_rows)
    seedwise.to_csv(OUT / "validation_seedwise.csv", index=False)

    summ = seedwise.groupby("model", sort=False).agg(seeds=("seed", "count"), rmse_mean=("rmse_h", "mean"), rmse_sd=("rmse_h", "std"),
                                                       mae_mean=("mae_h", "mean"), r2_mean=("r2", "mean"), bias_mean=("bias_h", "mean"),
                                                       port_balanced_rmse=("port_balanced_rmse_h", "mean")).reset_index()
    ewma_rmse = metrics.rmse(y, base[EWMA])
    summ["minus_ewma_h"] = summ["rmse_mean"] - ewma_rmse
    summ["seeds_beating_ewma"] = [int((seedwise[(seedwise["model"] == m)]["rmse_h"] < ewma_rmse).sum()) if m in EXPERIMENTS else None
                                  for m in summ["model"]]
    summ["training_protocol"] = summ["model"].map(PROTOCOL)
    params = {}
    for label, d in EXPERIMENTS.items():
        from dl.models import load_checkpoint

        model, _ = load_checkpoint(next(iter(sorted(d.glob("seed_*.pt")))))
        params[label] = sum(p.numel() for p in model.parameters())
    summ["n_parameters"] = summ["model"].map(params)
    summ.to_csv(OUT / "validation_comparison.csv", index=False)

    models = {**{k: [v] for k, v in base.items()}, **{lab: [preds[(lab, s)] for s in range(5)] for lab in EXPERIMENTS}}
    for q in ports:
        mk = port == q
        for name, plist in models.items():
            r = [metrics.rmse(y[mk], p[mk]) for p in plist]
            port_rows.append({"port": q, "model": name, "n": int(mk.sum()), "rmse_mean_h": float(np.mean(r)),
                              "rmse_sd_h": float(np.std(r, ddof=1)) if len(r) > 1 else None,
                              "bias_mean_h": float(np.mean([np.mean(p[mk] - y[mk]) for p in plist]))})
    per_port = pd.DataFrame(port_rows)
    per_port.to_csv(OUT / "validation_per_port.csv", index=False)

    for g, mk in masks(frame).items():
        n = int(mk.sum())
        if n == 0:
            continue
        for name, plist in models.items():
            r = [score(y[mk], p[mk]) for p in plist]
            sub_rows.append({"subgroup": g, "n": n, "ports": int(frame.loc[mk, "port_id"].nunique()), "small_sample": n < metrics.MIN_N,
                             "model": name, "rmse_mean_h": float(np.mean([x["rmse_h"] for x in r])),
                             "rmse_sd_h": float(np.std([x["rmse_h"] for x in r], ddof=1)) if len(r) > 1 else None,
                             "mae_mean_h": float(np.mean([x["mae_h"] for x in r])), "bias_mean_h": float(np.mean([x["bias_h"] for x in r]))})
    sub = pd.DataFrame(sub_rows)
    sub["minus_ewma_h"] = sub["rmse_mean_h"] - sub.groupby("subgroup")["rmse_mean_h"].transform(
        lambda s: s[sub.loc[s.index, "model"] == EWMA].iloc[0])
    sub.to_csv(OUT / "validation_subgroups.csv", index=False)
    plots(seedwise, per_port, sub)
    print(summ.round(4).to_string())
    (OUT / "comparison_info.json").write_text(json.dumps(
        {"split": "validation only", "n_windows": int(len(y)), "ewma_val_rmse_h": ewma_rmse,
         "note": "test split not loaded; existing test results are diagnostic only and were not used"}, indent=2))


def plots(seedwise, per_port, sub) -> None:
    colors = {"LSTM (A, basic)": "#8a8985", "LSTM (A_w, hour-weighted)": "#2a78d6", "GRU (B_w settings)": "#eda100",
              "C_w (EWMA-residual LSTM)": "#1b9e77", "Transformer (T)": "#b07aa1", "Transformer (T_w, hour-weighted)": "#e34948"}
    ewma = seedwise.loc[seedwise["model"] == EWMA, "rmse_h"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (m, c) in enumerate(colors.items()):
        v = seedwise.loc[seedwise["model"] == m, "rmse_h"]
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=c, s=30, zorder=3)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color="#0b0b0b", lw=1.5)
    ax.axhline(ewma, color="#1b9e77", ls=":", lw=1.3, label=f"EWMA α=0.2 ({ewma:.3f} h)")
    ax.set_xticks(range(len(colors)), [m.replace(" (", "\n(") for m in colors], fontsize=8)
    ax.set(title="Validation RMSE per seed (bar = 5-seed mean)", ylabel="validation RMSE (h)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "validation_rmse_by_model.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    ports = sorted(per_port["port"].unique())
    x = np.arange(len(ports))
    show = [EWMA, "GRU (B_w settings)", "C_w (EWMA-residual LSTM)", "Transformer (T_w, hour-weighted)"]
    for i, m in enumerate(show):
        v = per_port[per_port["model"] == m].set_index("port").loc[ports, "rmse_mean_h"]
        axes[0].bar(x + (i - 1.5) * 0.2, v, width=0.2, color=colors.get(m, "#1b9e77" if m == EWMA else "#555"),
                    label=m, alpha=0.6 if m == EWMA else 1)
    axes[0].set_xticks(x, ports, rotation=45)
    axes[0].set(title="Validation RMSE by port (5-seed mean)", ylabel="RMSE (h)", ylim=(3.5, 5.2))
    axes[0].legend(fontsize=7)
    groups = ["all", "disruption active at t", "disruption 1–6 h old", "disruption 7–24 h old", "disruption > 24 h old",
              "extreme: top-decile window level (known at t, per port)", "extreme: top-decile actual dwell (per port)"]
    s = sub[sub["model"].isin(show[1:])]
    for i, m in enumerate(show[1:]):
        v = s[s["model"] == m].set_index("subgroup").loc[groups, "minus_ewma_h"]
        axes[1].barh(np.arange(len(groups)) + (i - 1) * 0.25, v, height=0.25, color=colors[m], label=m)
    axes[1].axvline(0, color="#0b0b0b", lw=0.8)
    n = sub[sub["model"] == EWMA].set_index("subgroup").loc[groups, "n"]
    axes[1].set_yticks(range(len(groups)), [f"{g} (n={k})" for g, k in zip(groups, n)], fontsize=7)
    axes[1].set(title="Validation RMSE minus EWMA by subgroup (negative = better)", xlabel="Δ RMSE (h)")
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "validation_ports_and_subgroups.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, exp in zip(axes, ["T_transformer", "T_w_transformer_hour_weighted"]):
        for seed in range(5):
            h = pd.read_csv(TRANSFORMER_LOGS / exp / f"seed_{seed}.csv")
            ax.plot(h["epoch"], h["val_rmse_h"], color="#e34948", lw=1, alpha=0.8, label="validation" if seed == 0 else None)
            ax.plot(h["epoch"], h["train_rmse_h"], color="#0b0b0b", lw=0.8, ls="--", alpha=0.4, label="training" if seed == 0 else None)
        ax.axhline(ewma, color="#1b9e77", ls=":", lw=1.2, label="EWMA (validation)")
        ax.set(title=f"{exp}: RMSE in hours (5 seeds)", xlabel="epoch", ylabel="RMSE (h)", ylim=(3.9, 4.9))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "transformer_training_curves.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
