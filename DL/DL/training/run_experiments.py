"""Run experiments (train + validation only) and the validation-based model selection.

From the project root:

    .venv/bin/python -m dl.training.run_experiments main        # round 1: A (no embedding) and B (embedding), 5 seeds each
    .venv/bin/python -m dl.training.run_experiments round2      # round 2: hour-weighted loss (A_w, B_w) and EWMA-residual C_w; final selection
    .venv/bin/python -m dl.training.run_experiments ablations   # one-factor ablations of the selected configuration, 5 seeds
    .venv/bin/python -m dl.training.run_experiments gru         # GRU version of the selected configuration, 5 seeds

Outputs: DL/DL/training/logs/<experiment>/seed_<s>.csv (per-epoch), DL/DL/artifacts/runs/<experiment>/seed_<s>.pt,
DL/DL/artifacts/experiment_results.csv (one row per experiment x seed, validation metrics only),
DL/DL/artifacts/selection.json, lstm_best.pt, lstm_config.json, experiment_metadata.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from dl.models import save_checkpoint
from dl.models.checkpoint import dataset_reference

from . import config as C
from . import data
from .trainer import predict, restore, train

RESULTS_CSV = C.ARTIFACTS / "experiment_results.csv"


def val_metrics(model, va: data.SplitData) -> dict:
    yh = data.to_hours(predict(model, va.X, va.port_index), va.port_index)
    e = yh - va.y_hours
    return {"val_rmse_h": float(np.sqrt(np.mean(e ** 2))), "val_mae_h": float(np.mean(np.abs(e))),
            "val_r2": float(1 - np.sum(e ** 2) / np.sum((va.y_hours - va.y_hours.mean()) ** 2))}


def run(cfg: C.TrainConfig, seeds=C.SEEDS, runs_dir=None, logs_dir=None, results_csv=None) -> pd.DataFrame:
    runs_dir = runs_dir or C.ARTIFACTS / "runs"
    logs_dir = logs_dir or C.LOGS
    tr = data.load("train", cfg.mask_channels, cfg.drop_features)
    va = data.load("val", cfg.mask_channels, cfg.drop_features)
    rows = []
    for seed in seeds:
        c = cfg.with_seed(seed)
        res = train(c, tr, va)
        model = restore(res)
        hist = pd.DataFrame(res.history)
        (logs_dir / cfg.name).mkdir(parents=True, exist_ok=True)
        hist.to_csv(logs_dir / cfg.name / f"seed_{seed}.csv", index=False)
        best_row = hist.loc[hist["epoch"] == res.best_epoch].iloc[0]
        m = val_metrics(model, va)
        assert abs(m["val_rmse_h"] - res.best_val_rmse_h) < 1e-4, "restored model must reproduce its best validation RMSE"
        extra = {"train_config": c.to_dict(), "best_epoch": res.best_epoch, "best_val_rmse_h": res.best_val_rmse_h}
        save_checkpoint(runs_dir / cfg.name / f"seed_{seed}.pt", model, res.channels, extra)
        rows.append({"experiment": cfg.name, "seed": seed, "cell": cfg.cell, "port_embedding_dim": cfg.port_embedding_dim,
                     "channels": "|".join(res.channels), "n_channels": len(res.channels), "n_parameters": res.n_parameters,
                     "best_epoch": res.best_epoch, "epochs_run": len(hist), "train_rmse_h_at_best": best_row["train_rmse_h"],
                     **m, "seconds": round(res.seconds, 1), "notes": cfg.notes})
        print(f"  {cfg.name} seed {seed}: val RMSE {m['val_rmse_h']:.4f} h at epoch {res.best_epoch} ({res.seconds:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    _merge_results(df, results_csv or RESULTS_CSV)
    return df


def _merge_results(df: pd.DataFrame, results_csv=RESULTS_CSV) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    if results_csv.exists():
        old = pd.read_csv(results_csv)
        old = old[~old.set_index(["experiment", "seed"]).index.isin(df.set_index(["experiment", "seed"]).index)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(results_csv, index=False)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("experiment").agg(seeds=("seed", "count"), val_rmse_mean=("val_rmse_h", "mean"), val_rmse_sd=("val_rmse_h", "std"),
                                        val_rmse_min=("val_rmse_h", "min"), val_rmse_max=("val_rmse_h", "max"),
                                        val_mae_mean=("val_mae_h", "mean"), best_epoch_mean=("best_epoch", "mean"),
                                        n_parameters=("n_parameters", "first"))


def select(res: pd.DataFrame) -> dict:
    """Pre-registered rule: lower mean validation RMSE wins unless within SELECTION_TOLERANCE_H, then the simpler A."""
    s = summary(res[res["experiment"].isin([C.A.name, C.B.name])])
    a, b = s.loc[C.A.name, "val_rmse_mean"], s.loc[C.B.name, "val_rmse_mean"]
    diff = b - a
    chosen = C.B if diff < -C.SELECTION_TOLERANCE_H else C.A
    reason = (f"B mean val RMSE {b:.4f} vs A {a:.4f} (B − A = {diff:+.4f} h); "
              + ("B better by more than the tolerance" if chosen is C.B else f"|difference| or sign does not clear the {C.SELECTION_TOLERANCE_H} h tolerance -> simpler A"))
    runs = res[res["experiment"] == chosen.name].sort_values("val_rmse_h")
    best_seed = int(runs.iloc[0]["seed"])
    return {"rule": f"choose B only if its mean validation RMSE is lower than A's by more than {C.SELECTION_TOLERANCE_H} h; otherwise A",
            "chosen_experiment": chosen.name, "reason": reason, "best_seed": best_seed,
            "best_seed_val_rmse_h": float(runs.iloc[0]["val_rmse_h"]),
            "summary": s.reset_index().to_dict("records"), "decided_without_test_data": True}


def select_final(res: pd.DataFrame) -> dict:
    """Final rule over all candidates (fixed before round 2 ran): find the lowest mean validation RMSE; choose the
    simplest candidate (order in config.CANDIDATES) whose mean is within SELECTION_TOLERANCE_H of it."""
    names = [c.name for c in C.CANDIDATES]
    s = summary(res[res["experiment"].isin(names)]).reindex(names)
    if s["seeds"].isna().any():
        raise ValueError(f"missing candidate runs: {s.index[s['seeds'].isna()].tolist()}")
    best = s["val_rmse_mean"].min()
    chosen = next(n for n in names if s.loc[n, "val_rmse_mean"] <= best + C.SELECTION_TOLERANCE_H)
    runs = res[res["experiment"] == chosen].sort_values("val_rmse_h")
    return {"rule": f"lowest mean validation RMSE over 5 seeds; among candidates within {C.SELECTION_TOLERANCE_H} h of it, the simplest "
                    f"in the order {names}", "chosen_experiment": chosen,
            "reason": f"best mean {best:.4f} h ({s['val_rmse_mean'].idxmin()}); {chosen} mean {s.loc[chosen, 'val_rmse_mean']:.4f} h",
            "best_seed": int(runs.iloc[0]["seed"]), "best_seed_val_rmse_h": float(runs.iloc[0]["val_rmse_h"]),
            "summary": s.reset_index().to_dict("records"), "decided_without_test_data": True}


def environment() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=C.DL_ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        commit = f"unavailable ({e.__class__.__name__})"
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "platform": platform.platform(), "device": "cpu", "git_commit_at_run": commit}


def finalize_selection(sel: dict) -> dict:
    chosen = C.EXPERIMENTS[sel["chosen_experiment"]]
    src = C.ARTIFACTS / "runs" / chosen.name / f"seed_{sel['best_seed']}.pt"
    shutil.copyfile(src, C.ARTIFACTS / "lstm_best.pt")
    payload = torch.load(src, map_location="cpu", weights_only=False)
    cfg = {"model_config": payload["model_config"], "input_channels": payload["input_channels"], "target": payload["target"],
           "train_config": payload["extra"]["train_config"], "best_epoch": payload["extra"]["best_epoch"],
           "best_val_rmse_h": payload["extra"]["best_val_rmse_h"], "port_embedding": {
               "used": payload["model_config"]["port_embedding_dim"] > 0, "dim": payload["model_config"]["port_embedding_dim"],
               "port_order": json.loads((C.DATASET / "feature_schema.json").read_text())["arrays"]["port_index"]["ports"]},
           "dataset_reference": payload["dataset_reference"],
           "inference": "from dl.models import load_checkpoint; model, meta = load_checkpoint('DL/DL/artifacts/lstm_best.pt'); "
                        "x from cargomind_dl.dataset.load_split (channels as listed); hours = cargomind_dl.dataset.to_hours(model(x, port_index), port_index)"}
    (C.ARTIFACTS / "lstm_config.json").write_text(json.dumps(cfg, indent=2))
    (C.ARTIFACTS / "selection.json").write_text(json.dumps(sel, indent=2))
    return sel


def write_metadata() -> None:
    meta = {"environment": environment(), "seeds": list(C.SEEDS), "selection_tolerance_h": C.SELECTION_TOLERANCE_H,
            "experiments": {k: v.to_dict() for k, v in C.EXPERIMENTS.items()}, "dataset_reference": dataset_reference(),
            "test_split_used_for": "final evaluation only (python -m dl.evaluation.evaluate test), after selection.json was written"}
    (C.ARTIFACTS / "experiment_metadata.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["main", "round2", "ablations", "gru", "transformer"])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(C.SEEDS))
    args = ap.parse_args()
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.phase == "transformer":  # separate artifacts, logs and results; existing ones are not touched
        out_csv = C.TRANSFORMER_ARTIFACTS / "experiment_results.csv"
        for cfg in (C.T, C.T_W):
            run(cfg, args.seeds, runs_dir=C.TRANSFORMER_ARTIFACTS / "runs", logs_dir=C.TRANSFORMER_LOGS, results_csv=out_csv)
        print(summary(pd.read_csv(out_csv)).round(4).to_string())
        return
    write_metadata()
    if args.phase == "main":
        for cfg in (C.A, C.B):
            run(cfg, args.seeds)
        sel = select(pd.read_csv(RESULTS_CSV))
        (C.ARTIFACTS / "selection_round1.json").write_text(json.dumps(sel, indent=2))
        print(json.dumps({k: sel[k] for k in ("chosen_experiment", "reason", "best_seed", "best_seed_val_rmse_h")}, indent=2))
    elif args.phase == "round2":
        for cfg in (C.A_W, C.B_W, C.C_W):
            run(cfg, args.seeds)
        sel = finalize_selection(select_final(pd.read_csv(RESULTS_CSV)))
        print(json.dumps({k: sel[k] for k in ("chosen_experiment", "reason", "best_seed", "best_seed_val_rmse_h")}, indent=2))
    elif args.phase == "ablations":
        base = C.EXPERIMENTS[json.loads((C.ARTIFACTS / "selection.json").read_text())["chosen_experiment"]]
        for cfg in C.ablations(base):
            run(cfg, args.seeds)
    else:
        base = C.EXPERIMENTS[json.loads((C.ARTIFACTS / "selection.json").read_text())["chosen_experiment"]]
        run(C.gru_like(base), args.seeds)
    print(summary(pd.read_csv(RESULTS_CSV)).round(4).to_string())


if __name__ == "__main__":
    main()

__all__ = ["run", "select", "summary", "replace"]
