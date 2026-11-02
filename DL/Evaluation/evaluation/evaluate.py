"""Evaluation of trained forecasters in original hours.

    .venv/bin/python -m dl.evaluation.evaluate validation   # diagnostics on validation for every trained experiment
    .venv/bin/python -m dl.evaluation.evaluate test         # ONE final test evaluation of the selected configuration

The test command refuses to run before DL/DL/artifacts/selection.json exists (the validation-based choice),
evaluates only the selected configuration (its 5 seeds and the designated lstm_best.pt), and records
every invocation in DL/DL/artifacts/test_evaluation_log.json. Nothing it produces feeds back into training.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dl.models import load_checkpoint  # noqa: E402
from dl.training import data  # noqa: E402
from dl.training.config import ARTIFACTS, DATASET, EVAL_FIGURES, EVAL_RESULTS  # noqa: E402
from dl.training.trainer import predict  # noqa: E402

from . import frames, metrics  # noqa: E402

OUT = EVAL_RESULTS / "evaluation"
FIG = EVAL_FIGURES


def model_predictions(ckpt: Path, split: str) -> tuple[np.ndarray, data.SplitData]:
    model, meta = load_checkpoint(ckpt)
    cfg = meta["extra"]["train_config"]
    sd = data.load(split, tuple(cfg["mask_channels"]), tuple(cfg["drop_features"]))
    if sd.channels != meta["input_channels"]:
        raise ValueError(f"{ckpt}: channel mismatch {sd.channels} vs {meta['input_channels']}")
    return data.to_hours(predict(model, sd.X, sd.port_index), sd.port_index), sd


def headline(frame: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    y, port = frame["y_hours"].values, frame["port_id"].values
    return pd.DataFrame([{"model": k, "n": len(y), "mae_h": metrics.mae(y, p), "rmse_h": metrics.rmse(y, p), "r2": metrics.r2(y, p),
                          "port_balanced_rmse_h": metrics.port_balanced_rmse(y, p, port)} for k, p in preds.items()])


def evaluate_experiment(exp: str, split: str) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """Per-seed predictions of one experiment on one split, plus the seed-mean metrics."""
    preds, sd = {}, None
    for ckpt in sorted((ARTIFACTS / "runs" / exp).glob("seed_*.pt")):
        p, sd = model_predictions(ckpt, split)
        preds[f"{exp} {ckpt.stem}"] = p
    frame = frames.context(sd.window_id, sd.y_hours)
    return frame, preds, headline(frame, preds)


def validation() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exps = sorted(p.name for p in (ARTIFACTS / "runs").iterdir() if p.is_dir())
    heads, breaks = [], []
    for exp in exps:
        frame, preds, head = evaluate_experiment(exp, "val")
        base = frames.baseline_predictions(frame)
        head["experiment"] = exp
        heads.append(head)
        b = metrics.breakdown(frame, {**base, **preds})
        b["experiment"] = exp
        breaks.append(b)
    pd.concat(heads).to_csv(OUT / "validation_headline_by_seed.csv", index=False)
    br = pd.concat(breaks)
    br = br[~br["model"].isin(frames.BASELINES) | (br["experiment"] == exps[0])]
    br.to_csv(OUT / "validation_subgroups_by_seed.csv", index=False)
    agg = br.assign(exp=np.where(br["model"].isin(frames.BASELINES), br["model"], br["experiment"])) \
        .groupby(["subgroup", "exp"]).agg(n=("n", "first"), rmse_mean=("rmse_h", "mean"), rmse_sd=("rmse_h", "std"),
                                          diff_vs_ewma_mean=("rmse_minus_reference_h", "mean")).reset_index()
    agg.to_csv(OUT / "validation_subgroups_summary.csv", index=False)
    print(agg[agg["subgroup"].isin(["all", "high-signal ports", "noise-dominated ports", "disruption active at t", "labour strike at t"])]
          .round(4).to_string())


def test() -> None:
    sel_path = ARTIFACTS / "selection.json"
    if not sel_path.exists():
        raise SystemExit("selection.json missing: run `python -m dl.training.run_experiments main` first (validation-based selection)")
    sel = json.loads(sel_path.read_text())
    exp = sel["chosen_experiment"]
    log_path = ARTIFACTS / "test_evaluation_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    log.append({"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "experiment": exp,
                "selection_sha_basis": sel["reason"], "invocation": len(log) + 1})
    log_path.write_text(json.dumps(log, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    frame, seed_preds, _ = evaluate_experiment(exp, "test")
    best_pred, _ = model_predictions(ARTIFACTS / "lstm_best.pt", "test")
    base = frames.baseline_predictions(frame)
    stored = pd.read_csv(DATASET_BASELINES)
    stored = stored[(stored["split"] == "test") & (stored["port_id"] == "ALL")].set_index("baseline")["rmse_h"]
    recomputed = {"persistence": metrics.rmse(frame["y_hours"].values, base["persistence"]),
                  "trailing_mean_12h": metrics.rmse(frame["y_hours"].values, base["12 h mean"]),
                  "ewma_alpha_0.2": metrics.rmse(frame["y_hours"].values, base["EWMA α=0.2"])}
    for k, v in recomputed.items():
        if abs(stored[k] - v) > 1e-3:
            raise ValueError(f"baseline {k} on these windows ({v:.4f}) differs from baseline_metrics.csv ({stored[k]:.4f})")

    lstm_label = f"LSTM ({exp}, best-val seed {sel['best_seed']})"
    preds = {**base, lstm_label: best_pred, **seed_preds}
    head = headline(frame, preds)
    seeds = head[head["model"].str.startswith(exp)]
    head = pd.concat([head, pd.DataFrame([{"model": f"{exp}: mean of 5 seeds", "n": len(frame),
                                           **{c: seeds[c].mean() for c in ("mae_h", "rmse_h", "r2", "port_balanced_rmse_h")}},
                                          {"model": f"{exp}: SD over 5 seeds", "n": len(frame),
                                           **{c: seeds[c].std() for c in ("mae_h", "rmse_h", "r2", "port_balanced_rmse_h")}}])])
    head.to_csv(OUT / "test_headline.csv", index=False)
    br = metrics.breakdown(frame, preds)
    br.to_csv(OUT / "test_subgroups.csv", index=False)
    boots = [{"model": name, **metrics.paired_bootstrap(frame, p, base["EWMA α=0.2"], b)}
             for name, p in [(lstm_label, best_pred)] + list(seed_preds.items()) for b in ("day", "port_day")]
    (OUT / "test_paired_bootstrap_vs_ewma.json").write_text(json.dumps(boots, indent=2))
    out = frame[["window_id", "port_id", "target_ts", "y_hours", "disruption_active_t", "disruption_hours_elapsed_t", "labour_strike_t",
                 "n_reconstructed_dwell_inputs"]].copy()
    for k, p in preds.items():
        out[f"pred_{k}"] = p
    out.to_csv(OUT / "test_predictions.csv", index=False)
    plot_test(frame, base, best_pred, lstm_label, br)
    print(head.round(4).to_string())
    print(json.dumps(boots[:2], indent=1))


DATASET_BASELINES = DATASET / "baseline_metrics.csv"


def plot_test(frame, base, best_pred, label, br) -> None:
    ports = sorted(frame["port_id"].unique())
    sub = br[br["subgroup"].str.startswith("port ")]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(ports))
    for i, (m, col) in enumerate([("persistence", "#8a8985"), ("EWMA α=0.2", "#1b9e77"), (label, "#2a78d6")]):
        v = [sub[(sub["subgroup"] == f"port {p}") & (sub["model"] == m)]["rmse_h"].iloc[0] for p in ports]
        ax.bar(x + (i - 1) * 0.27, v, width=0.27, color=col, label=m)
    ax.set_xticks(x, ports)
    ax.set(title="Test RMSE by port (final evaluation, evaluated once)", xlabel="port", ylabel="RMSE (h)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "test_rmse_by_port.png", dpi=140)
    plt.close(fig)

    # the two test ports with disruption / strike episodes: actual vs forecasts
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    for ax, p in zip(axes, ("MYPKG", "INNSA")):
        m = (frame["port_id"] == p).values
        t = pd.to_datetime(frame["target_ts"][m])
        ax.plot(t, frame["y_hours"][m], color="#0b0b0b", lw=0.8, label="actual (observed targets)")
        ax.plot(t, base["EWMA α=0.2"][m], color="#1b9e77", lw=1.2, label="EWMA α=0.2")
        ax.plot(t, best_pred[m], color="#2a78d6", lw=1.2, label=label)
        dis = frame["disruption_active_t"].values[m] == 1
        ax.fill_between(t, ax.get_ylim()[0], ax.get_ylim()[1], where=dis, color="#eda100", alpha=0.15, label="disruption active at t")
        ax.set(title=f"{p}: next-hour dwell, test period", ylabel="dwell (h)")
    axes[0].legend(fontsize=8, ncol=4)
    axes[1].set_xlabel("target hour (UTC)")
    fig.tight_layout()
    fig.savefig(FIG / "test_disruption_ports_timeseries.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["validation", "test"])
    args = ap.parse_args()
    validation() if args.split == "validation" else test()


if __name__ == "__main__":
    main()
