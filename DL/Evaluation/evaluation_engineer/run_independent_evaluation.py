"""Independent evaluation of the CargoMind Stage 02 DL component.

Run from the project root:

    .venv/bin/python -m dl.evaluation_engineer.run_independent_evaluation

Reads DL/Dataset/processed/, DL/DL/artifacts/ and DL/Evaluation/results/ (never writes to them). All outputs go to
DL/Evaluation/evaluation_engineer/. Metrics, baselines and bootstraps are re-implemented here rather than imported
from dl.evaluation, so the reported numbers are checked against separate code. The model checkpoints are
loaded with the project's own loader (that loader is itself one of the things under audit).

Test-split use in this audit: the selected model's test metrics are re-derived (verification). Test
metrics for the other experiments are computed as DIAGNOSTICS ONLY, to check whether validation rankings
held; they must not be used to choose a future model (see report, section 10).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cargomind_dl import config as data_config  # noqa: E402

ROOT = data_config.PROJECT_ROOT
OUT = Path(__file__).resolve().parent  # DL/Evaluation/evaluation_engineer
DATA = data_config.DATASET  # DL/Dataset/processed
ART = data_config.DL_STAGE / "artifacts"  # DL/DL/artifacts
EVAL_RESULTS = data_config.EVALUATION_STAGE / "results"
TESTS = data_config.EVALUATION_STAGE / "tests"

from dl.models import load_checkpoint  # noqa: E402

PORTS = json.loads((DATA / "feature_schema.json").read_text())["arrays"]["port_index"]["ports"]
HIGH = ["NLRTM", "USLAX", "CNSHA", "INNSA", "SGSIN"]
NOISE = ["MYPKG", "DEHAM", "AEJEA", "INMAA"]
SEED = 20261102
REPS = 5000


# ----------------------------------------------------------------------------------------------
# Independent metric implementations (hours)
# ----------------------------------------------------------------------------------------------
def metrics(y, p, port=None) -> dict:
    e = p - y
    out = {"n": int(len(y)), "mae_h": float(np.abs(e).mean()), "rmse_h": float(np.sqrt((e ** 2).mean())),
           "bias_h": float(e.mean()), "r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())}
    if port is not None:
        per = [(np.abs(e[port == q]).mean(), np.sqrt((e[port == q] ** 2).mean())) for q in np.unique(port)]
        out["port_balanced_mae_h"] = float(np.mean([a for a, _ in per]))
        out["port_balanced_rmse_h"] = float(np.mean([b for _, b in per]))
    return out


# ----------------------------------------------------------------------------------------------
# Data, provenance and integrity
# ----------------------------------------------------------------------------------------------
def load_npz(split):
    return dict(np.load(DATA / f"{split}.npz"))


def integrity(panel: pd.DataFrame, windows: pd.DataFrame) -> list[dict]:
    """Leakage / split / scaler checks re-derived from the hourly panel, not from the build code."""
    res = []

    def add(area, check, ok, evidence):
        res.append({"area": area, "check": check, "result": "PASS" if ok else "FAIL", "evidence": evidence})

    man = json.loads((DATA / "manifest.json").read_text())
    bad = [k for k, h in man["outputs_sha256"].items() if hashlib.sha256((DATA / k).read_bytes()).hexdigest() != h]
    src = [k for k, h in man["sources_sha256"].items() if hashlib.sha256(data_config.resolve_recorded_path(k).read_bytes()).hexdigest() != h]
    add("data", "DL/Dataset/processed outputs and their sources match manifest.json", not bad and not src, f"changed: {bad + src or 'none'}")
    meta = json.loads((DATA / "split_metadata.json").read_text())
    b = meta["bounds_hour_index"]
    add("split", "split boundaries are global hour indices shared by all ports (not per-port)",
        b == {"train": [0, 700], "val": [700, 850], "test": [850, 1000]}, json.dumps(b))
    k = windows[windows["keep"]]
    first = k["t_hour_index"] - 11
    lo = k["split"].map({s: v[0] for s, v in b.items()})
    hi = k["split"].map({s: v[1] for s, v in b.items()})
    add("leakage", "every kept window: inputs t-11..t and target t+1 all inside the target's split",
        bool(((first >= lo) & (k["target_hour_index"] < hi) & (k["target_hour_index"] == k["t_hour_index"] + 1)).all()), f"{len(k)} windows")
    add("leakage", "no kept window has a reconstructed target", bool((k["target_status"] != "reconstructed").all()),
        f"{int((windows['target_status'] == 'reconstructed').sum())} reconstructed-target candidates excluded")
    counts = k.groupby("split").size().to_dict()
    add("split", "window counts 6,579 / 1,306 / 1,318", counts == {"train": 6579, "val": 1306, "test": 1318}, json.dumps(counts))
    # scalers: refit independently on training-hour, non-reconstructed cells and compare (no artifact is changed)
    sc = json.loads((DATA / "scalers.json").read_text())
    tr = panel[(panel["hour_index"] < 700)]
    worst = 0.0
    for p in PORTS:
        v = tr[(tr["port_id"] == p) & (tr["prov_dwell"] != 2)]["dwell"]
        worst = max(worst, abs(v.mean() - sc["dwell"]["per_port"][p]["mean"]), abs(v.std() - sc["dwell"]["per_port"][p]["std"]))
    alt = panel[panel["port_id"] == PORTS[0]]["dwell"]
    add("scaling", "dwell scalers equal statistics of training hours only (reconstructed cells excluded)", worst < 1e-9,
        f"max |Δ| {worst:.1e} h; for contrast, {PORTS[0]} all-hours mean would be {alt.mean():.3f} vs saved {sc['dwell']['per_port'][PORTS[0]]['mean']:.3f}")
    # baselines: recompute EWMA / mean / persistence from the hourly panel, using only hours t-11..t
    pv = panel.set_index(["port_id", "hour_index"])["dwell"]
    dw = np.stack([pv.loc[list(zip(k["port_id"], k["t_hour_index"] - j))].values for j in range(11, -1, -1)], 1)
    s = dw[:, 0].copy()
    for j in range(1, 12):
        s = 0.2 * dw[:, j] + 0.8 * s
    tgt = pv.loc[list(zip(k["port_id"], k["target_hour_index"]))].values
    add("baselines", "stored EWMA / 12 h mean / persistence = recomputation from hours t-11..t only (target hour not used)",
        bool(np.allclose(s, k["baseline_ewma_alpha_0.2_hours"], atol=1e-3) and np.allclose(dw.mean(1), k["baseline_trailing_mean_12h_hours"], atol=1e-3)
             and np.allclose(dw[:, -1], k["baseline_persistence_hours"], atol=1e-6)),
        f"max |Δ| EWMA {np.abs(s - k['baseline_ewma_alpha_0.2_hours']).max():.1e} h; target equals panel hour t+1: {np.allclose(tgt, k['target_hours'])}")
    # EWMA alpha chosen on train
    schema = json.loads((DATA / "feature_schema.json").read_text())
    add("baselines", "EWMA alpha selected on training windows", schema["baselines"]["alpha_selected_on"].startswith("training"),
        f"alpha {schema['baselines']['ewma_alpha']}; grid {schema['baselines']['alpha_grid_rmse']}")
    # model inputs: every X cell re-derived for a random sample
    rng = np.random.default_rng(SEED)
    d = load_npz("test")
    kk = k.set_index("window_id")
    worst = 0.0
    for i in rng.choice(len(d["y"]), 300, replace=False):
        w = kk.loc[d["window_id"][i]]
        raw = pv.loc[[(w["port_id"], h) for h in range(w["t_hour_index"] - 11, w["t_hour_index"] + 1)]].values
        st = sc["dwell"]["per_port"][w["port_id"]]
        worst = max(worst, float(np.abs((raw - st["mean"]) / st["std"] - d["X"][i, :, 0]).max()))
    add("leakage", "test X dwell channel = scaled dwell of hours t-11..t (300 random windows)", worst < 1e-4, f"max |Δ| {worst:.1e} z")
    # test-set usage log
    log = json.loads((ART / "test_evaluation_log.json").read_text())
    sel_time = (ART / "selection.json").stat().st_mtime
    add("test discipline", "test evaluated once, after selection.json; only the selected experiment",
        len(log) == 1 and log[0]["experiment"] == json.loads((ART / "selection.json").read_text())["chosen_experiment"],
        f"{len(log)} logged invocation(s): {log}; selection.json mtime {time.strftime('%Y-%m-%d %H:%M', time.gmtime(sel_time))} UTC")
    return res


# ----------------------------------------------------------------------------------------------
# Predictions for every checkpoint
# ----------------------------------------------------------------------------------------------
def predict_checkpoint(path: Path, split: str, cache: dict) -> tuple[np.ndarray, dict]:
    from cargomind_dl import dataset as ds

    model, meta = load_checkpoint(path)
    cfg = meta["extra"]["train_config"]
    key = (split, tuple(cfg["mask_channels"]))
    if key not in cache:
        cache[key] = ds.load_split(split, mask_channels=tuple(cfg["mask_channels"]))
    d = cache[key]
    keep = [i for i, c in enumerate(d["channels"]) if c not in cfg["drop_features"]]
    channels = [d["channels"][i] for i in keep]
    if channels != meta["input_channels"]:
        raise ValueError(f"{path}: channels {channels} != checkpoint {meta['input_channels']}")
    with torch.no_grad():
        z = model(torch.from_numpy(np.ascontiguousarray(d["X"][..., keep])), torch.from_numpy(d["port_index"].astype(np.int64))).numpy()
    sc = json.loads((DATA / "scalers.json").read_text())["dwell"]["per_port"]
    mu = np.array([sc[p]["mean"] for p in PORTS])[d["port_index"]]
    sd = np.array([sc[p]["std"] for p in PORTS])[d["port_index"]]
    return z * sd + mu, {"channels": channels, "model_config": meta["model_config"], "train_config": cfg,
                         "best_epoch": meta["extra"]["best_epoch"], "best_val_rmse_h_logged": meta["extra"]["best_val_rmse_h"]}


def frame_for(split: str, windows: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    d = load_npz(split)
    w = windows.set_index("window_id").loc[d["window_id"]].reset_index()
    assert np.allclose(w["target_hours"].values, d["y_hours"], atol=1e-3)
    pv = panel.set_index(["port_id", "hour_index"])
    at_t = pv.loc[list(zip(w["port_id"], w["t_hour_index"]))]
    w["dis_t"] = at_t["disruption_active"].values
    w["elapsed_t"] = at_t["disruption_hours_elapsed"].values
    w["strike_t"] = at_t["labour_strike"].values
    w["y"] = d["y_hours"].astype(float)
    w["day"] = pd.to_datetime(w["target_ts"], utc=True).dt.floor("D").astype(str)
    return w


def subgroup_masks(f: pd.DataFrame, dev_strike_ports: set) -> dict:
    m = {"all": np.ones(len(f), bool), "normal (no disruption, no strike at t)": (f["dis_t"] == 0).values & (f["strike_t"] == 0).values,
         "disruption active at t": (f["dis_t"] == 1).values, "disruption onset (1-6 h old)": f["elapsed_t"].between(1, 6).values,
         "disruption 7-24 h old": f["elapsed_t"].between(7, 24).values, "disruption > 24 h old": (f["elapsed_t"] > 24).values,
         "labour strike at t": (f["strike_t"] == 1).values,
         "strike at a port with no development strikes": ((f["strike_t"] == 1) & ~f["port_id"].isin(dev_strike_ports)).values,
         "strike at a port with development strikes": ((f["strike_t"] == 1) & f["port_id"].isin(dev_strike_ports)).values,
         "high-signal ports": f["port_id"].isin(HIGH).values, "noise-dominated ports": f["port_id"].isin(NOISE).values,
         "top decile of target (extreme dwell, per port)": (f["y"] >= f.groupby("port_id")["y"].transform(lambda v: v.quantile(0.9))).values,
         ">= 1 reconstructed dwell input": (f["n_reconstructed_dwell_inputs"] > 0).values}
    for p in PORTS:
        m[f"port {p}"] = (f["port_id"] == p).values
    return m


# ----------------------------------------------------------------------------------------------
# Statistical comparison vs EWMA
# ----------------------------------------------------------------------------------------------
def boot(f: pd.DataFrame, pm: np.ndarray, pr: np.ndarray, scheme: str, reps=REPS, block_len=24, seed=SEED) -> dict:
    """RMSE(model) - RMSE(ref), paired on identical windows. Schemes:
    iid (ignores dependence), day, port_day (cluster), mbb (moving blocks of `block_len` consecutive target hours
    within each port, resampled independently per port: keeps within-port autocorrelation)."""
    y = f["y"].values
    rng = np.random.default_rng(seed)
    if scheme == "iid":
        groups = [np.array([i]) for i in range(len(y))]
    elif scheme in ("day", "port_day"):
        key = f["day"].values if scheme == "day" else (f["day"] + "|" + f["port_id"]).values
        groups = [np.flatnonzero(key == k) for k in np.unique(key)]
    diffs = np.empty(reps)
    for r in range(reps):
        if scheme == "mbb":
            idx = []
            for p in PORTS:
                pi = np.flatnonzero(f["port_id"].values == p)
                pi = pi[np.argsort(f["target_hour_index"].values[pi])]
                n = len(pi)
                starts = rng.integers(0, n - block_len + 1, int(np.ceil(n / block_len)))
                idx.append(np.concatenate([pi[s:s + block_len] for s in starts])[:n])
            idx = np.concatenate(idx)
        elif scheme == "iid":
            idx = rng.integers(0, len(y), len(y))
        else:
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        e1, e0 = pm[idx] - y[idx], pr[idx] - y[idx]
        diffs[r] = np.sqrt((e1 ** 2).mean()) - np.sqrt((e0 ** 2).mean())
    point = float(np.sqrt(((pm - y) ** 2).mean()) - np.sqrt(((pr - y) ** 2).mean()))
    return {"scheme": scheme if scheme != "mbb" else f"moving-block within port, L={block_len} h", "reps": reps, "point_h": point,
            "ci95_h": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], "share_model_better": float((diffs < 0).mean())}


def diebold_mariano(f: pd.DataFrame, pm: np.ndarray, pr: np.ndarray, max_lag: int = 24) -> dict:
    """DM test on squared-error differentials aggregated per target hour (sum over ports), Newey-West HAC."""
    d = pd.Series((pm - f["y"].values) ** 2 - (pr - f["y"].values) ** 2).groupby(f["target_hour_index"].values).mean().values
    n, dbar = len(d), d.mean()
    u = d - dbar
    lrv = u @ u / n
    for lag in range(1, max_lag + 1):
        w = 1 - lag / (max_lag + 1)
        lrv += 2 * w * (u[lag:] @ u[:-lag]) / n
    stat = dbar / np.sqrt(lrv / n)
    from scipy.stats import norm, t

    return {"hours": int(n), "mean_loss_diff_h2": float(dbar), "dm_stat": float(stat), "p_two_sided_normal": float(2 * norm.sf(abs(stat))),
            "p_two_sided_t": float(2 * t.sf(abs(stat), n - 1)), "hac_lag": max_lag}


# ----------------------------------------------------------------------------------------------
# Reproducibility: retrain two runs in memory (no artifact written)
# ----------------------------------------------------------------------------------------------
def retrain_checks(results: pd.DataFrame) -> list[dict]:
    from dl.training import config as C
    from dl.training.trainer import train

    out = []
    for cfg, seed in [(C.B_W, 2), (C.C_W, 0)]:
        r = train(cfg.with_seed(seed))
        logged = results[(results["experiment"] == cfg.name) & (results["seed"] == seed)].iloc[0]
        out.append({"experiment": cfg.name, "seed": seed, "retrained_best_epoch": r.best_epoch, "logged_best_epoch": int(logged["best_epoch"]),
                    "retrained_val_rmse_h": r.best_val_rmse_h, "logged_val_rmse_h": float(logged["val_rmse_h"]),
                    "identical": r.best_epoch == int(logged["best_epoch"]) and abs(r.best_val_rmse_h - logged["val_rmse_h"]) < 1e-9})
    return out


# ----------------------------------------------------------------------------------------------
# Inference-readiness probes (no code is changed)
# ----------------------------------------------------------------------------------------------
def inference_probes() -> list[dict]:
    model, meta = load_checkpoint(ART / "lstm_best.pt")
    out = []

    def probe(name, fn):
        try:
            v = fn()
            out.append({"probe": name, "outcome": "returned", "detail": str(v)[:160]})
        except Exception as e:  # recorded, not swallowed: each failure mode is reported
            out.append({"probe": name, "outcome": f"raised {type(e).__name__}", "detail": str(e)[:160]})

    x = torch.zeros(1, 12, 8)
    probe("valid [1,12,8] + port 0", lambda: model(x, torch.tensor([0])).tolist())
    probe("NaN in input", lambda: model(torch.full((1, 12, 8), float("nan")), torch.tensor([0])).tolist())
    probe("sequence length 6 instead of 12", lambda: model(torch.zeros(1, 6, 8), torch.tensor([0])).tolist())
    probe("7 channels (missing reconstruction flag)", lambda: model(torch.zeros(1, 12, 7), torch.tensor([0])).tolist())
    probe("port index 10 (out of range)", lambda: model(x, torch.tensor([10])).tolist())
    probe("no port index", lambda: model(x).tolist())
    probe("raw-hours input not standardised (dwell=120 h)", lambda: model(torch.cat([torch.full((1, 12, 1), 120.0), torch.zeros(1, 12, 7)], -1), torch.tensor([0])).tolist())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(DATA / "hourly_panel.csv")
    windows = pd.read_csv(DATA / "windows.csv")
    results = pd.read_csv(ART / "experiment_results.csv")
    stored_base = pd.read_csv(DATA / "baseline_metrics.csv")
    dev = panel[panel["hour_index"] < 850]
    dev_strike_ports = set(dev.loc[dev["labour_strike"] == 1, "port_id"])

    checks = integrity(panel, windows)
    frames = {s: frame_for(s, windows, panel) for s in ("val", "test")}
    base = {s: {"persistence": f["baseline_persistence_hours"].values, "12 h mean": f["baseline_trailing_mean_12h_hours"].values,
                "EWMA α=0.2": f["baseline_ewma_alpha_0.2_hours"].values} for s, f in frames.items()}

    # every checkpoint on validation and (diagnostic) test
    cache, rows, preds, configs = {}, [], {"val": {}, "test": {}}, {}
    for exp_dir in sorted((ART / "runs").iterdir()):
        for ck in sorted(exp_dir.glob("seed_*.pt")):
            seed = int(ck.stem.split("_")[1])
            for s in ("val", "test"):
                p, info = predict_checkpoint(ck, s, cache)
                preds[s][(exp_dir.name, seed)] = p
                f = frames[s]
                rows.append({"experiment": exp_dir.name, "seed": seed, "split": s, **metrics(f["y"].values, p, f["port_id"].values)})
            configs[exp_dir.name] = {k: info[k] for k in ("channels", "model_config")} | {
                "train_config": {k: v for k, v in info["train_config"].items() if k != "seed"}}
            logged = results[(results["experiment"] == exp_dir.name) & (results["seed"] == seed)]
            rows[-2]["logged_val_rmse_h"] = float(logged["val_rmse_h"].iloc[0]) if len(logged) else None
            rows[-2]["checkpoint_matches_log"] = bool(len(logged) and abs(rows[-2]["rmse_h"] - logged["val_rmse_h"].iloc[0]) < 1e-4)
            rows[-2]["best_epoch"] = info["best_epoch"]
    seedwise = pd.DataFrame(rows)
    seedwise.to_csv(OUT / "seedwise_metrics.csv", index=False)
    agg = seedwise.groupby(["experiment", "split"]).agg(
        seeds=("seed", "count"), mae_mean=("mae_h", "mean"), mae_sd=("mae_h", "std"), rmse_mean=("rmse_h", "mean"), rmse_sd=("rmse_h", "std"),
        bias_mean=("bias_h", "mean"), r2_mean=("r2", "mean"), pb_rmse_mean=("port_balanced_rmse_h", "mean"),
        pb_mae_mean=("port_balanced_mae_h", "mean"), rmse_min=("rmse_h", "min"), rmse_max=("rmse_h", "max")).reset_index()
    agg.to_csv(OUT / "experiment_summary.csv", index=False)

    base_rows = []
    for s, f in frames.items():
        for name, p in base[s].items():
            base_rows.append({"model": name, "split": s, **metrics(f["y"].values, p, f["port_id"].values)})
    base_df = pd.DataFrame(base_rows)
    sb = stored_base[stored_base["port_id"] == "ALL"]
    m = {"persistence": "persistence", "12 h mean": "trailing_mean_12h", "EWMA α=0.2": "ewma_alpha_0.2"}
    base_df["stored_rmse_h"] = [float(sb[(sb["baseline"] == m[r.model]) & (sb["split"] == r.split)]["rmse_h"].iloc[0]) for r in base_df.itertuples()]

    sel = json.loads((ART / "selection.json").read_text())
    chosen, best_seed = sel["chosen_experiment"], sel["best_seed"]
    ft = frames["test"]
    lstm_best_pred, _ = predict_checkpoint(ART / "lstm_best.pt", "test", cache)
    same_ckpt = bool(np.allclose(lstm_best_pred, preds["test"][(chosen, best_seed)]))
    stored_test = pd.read_csv(EVAL_RESULTS / "evaluation" / "test_predictions.csv")
    stored_col = [c for c in stored_test.columns if c.startswith("pred_LSTM")][0]
    reproduce_stored = float(np.abs(stored_test[stored_col].values - lstm_best_pred).max())

    # baseline comparison: selected model (designated checkpoint and 5-seed spread) vs baselines on test & val
    comp = []
    for s, f in frames.items():
        y = f["y"].values
        sel_seeds = [preds[s][(chosen, k)] for k in range(5)]
        cand = {"selected checkpoint (seed %d)" % best_seed: preds[s][(chosen, best_seed)],
                "selected config, seed-mean prediction (ensemble; not the deployed model)": np.mean(sel_seeds, 0)}
        for bname, bp in base[s].items():
            rb = np.sqrt(((bp - y) ** 2).mean())
            for cname, cp in cand.items():
                rm = np.sqrt(((cp - y) ** 2).mean())
                comp.append({"split": s, "model": cname, "baseline": bname, "baseline_rmse_h": rb, "model_rmse_h": rm,
                             "improvement_h": rb - rm, "improvement_pct": 100 * (rb - rm) / rb})
            per_seed = [np.sqrt(((q - y) ** 2).mean()) for q in sel_seeds]
            comp.append({"split": s, "model": "selected config, 5 seeds (mean ± sd of RMSE)", "baseline": bname, "baseline_rmse_h": rb,
                         "model_rmse_h": float(np.mean(per_seed)), "model_rmse_sd_h": float(np.std(per_seed, ddof=1)),
                         "improvement_h": rb - np.mean(per_seed), "improvement_pct": 100 * (rb - np.mean(per_seed)) / rb,
                         "seeds_beating_baseline": int(sum(r_ < rb for r_ in per_seed))})
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "baseline_comparison.csv", index=False)

    # per-port and subgroup metrics: selected checkpoint, 5 seeds, baselines, and C_w / GRU (diagnostic)
    sub_rows = []
    for s, f in frames.items():
        masks = subgroup_masks(f, dev_strike_ports)
        models = {**base[s], f"{chosen} seed {best_seed}": preds[s][(chosen, best_seed)]}
        for k in range(5):
            models[f"{chosen} seed {k}"] = preds[s][(chosen, k)]
        for other in ("C_w_lstm_ewma_residual_hour_weighted", "GRU__B_w_lstm_port_embedding_hour_weighted", "A_w_lstm_hour_weighted"):
            for k in range(5):
                models[f"{other} seed {k}"] = preds[s][(other, k)]
        for g, mk in masks.items():
            if mk.sum() == 0:
                continue
            for name, p in models.items():
                r = metrics(f["y"].values[mk], p[mk])
                sub_rows.append({"split": s, "subgroup": g, "model": name, **r, "small_sample": bool(mk.sum() < 30),
                                 "episodes_or_ports": int(f.loc[mk, "port_id"].nunique())})
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(OUT / "subgroup_metrics.csv", index=False)

    # statistics vs EWMA on test for the selected checkpoint and each seed
    stats = []
    for k in [best_seed] + [s_ for s_ in range(5) if s_ != best_seed]:
        p = preds["test"][(chosen, k)]
        for scheme in ("iid", "day", "port_day", "mbb"):
            stats.append({"model": f"{chosen} seed {k}", **boot(ft, p, base["test"]["EWMA α=0.2"], scheme)})
        for L in (12, 48):
            stats.append({"model": f"{chosen} seed {k}", **boot(ft, p, base["test"]["EWMA α=0.2"], "mbb", block_len=L)})
        stats.append({"model": f"{chosen} seed {k}", "scheme": "Diebold-Mariano (hourly-aggregated, Newey-West lag 24)",
                      **diebold_mariano(ft, p, base["test"]["EWMA α=0.2"])})
    # also the validation comparison for the selected config (non-test evidence)
    for k in range(5):
        stats.append({"model": f"{chosen} seed {k} [validation]", **boot(frames["val"], preds["val"][(chosen, k)], base["val"]["EWMA α=0.2"], "mbb")})
    stats = pd.DataFrame(stats)
    stats.to_json(OUT / "statistical_tests.json", orient="records", indent=2)
    stored_boot = json.loads((EVAL_RESULTS / "evaluation" / "test_paired_bootstrap_vs_ewma.json").read_text())

    repro = retrain_checks(results)
    probes = inference_probes()

    # unit tests
    import subprocess

    t = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(TESTS)], cwd=ROOT, capture_output=True, text=True)
    tests = {"returncode": t.returncode, "summary": [ln for ln in t.stderr.splitlines() if ln.startswith(("Ran ", "OK", "FAILED"))]}

    payload = {
        "units": "hours unless stated (bias = mean(prediction - actual); r2 dimensionless; *_h2 = hours squared)",
        "independence_note": "Audit performed by the same agent session that built the pipeline; results re-derived with separate code.",
        "test_use_note": "Selected-model test metrics re-verified. Test metrics of other experiments are diagnostic only and must not drive model selection.",
        "selection": sel, "experiments": configs,
        "integrity_checks": checks,
        "baselines": base_df.to_dict("records"),
        "experiment_summary": agg.to_dict("records"),
        "seedwise": seedwise.to_dict("records"),
        "baseline_comparison": comp.to_dict("records"),
        "subgroups": sub.to_dict("records"),
        "statistics_vs_ewma": stats.to_dict("records"),
        "stored_bootstrap_from_dl_reports": stored_boot,
        "verification": {"lstm_best_equals_selected_run_checkpoint": same_ckpt,
                         "max_abs_diff_vs_stored_test_predictions_h": reproduce_stored,
                         "retrain_checks": repro, "unit_tests": tests,
                         "all_checkpoints_reproduce_logged_val_rmse": bool(seedwise.loc[seedwise["split"] == "val", "checkpoint_matches_log"].all())},
        "inference_probes": probes,
        "development_strike_ports": sorted(dev_strike_ports),
    }
    (OUT / "evaluation_results.json").write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("checks:", pd.DataFrame(checks)["result"].value_counts().to_dict(), "| tests:", tests, "| repro:", [r["identical"] for r in repro])


if __name__ == "__main__":
    main()
