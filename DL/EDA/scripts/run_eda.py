"""CargoMind Stage 02 - exploratory data analysis of the DL-ready dataset (no model training).

Run from the CargoMind project root:

    .venv/bin/python DL/EDA/scripts/run_eda.py

Reads DL/Dataset/processed/ (built by DL/Dataset/scripts/build_dl_dataset.py) and nothing else; never writes to it or to
ML/Dataset/Processed. Writes DL/EDA/plots/*.png, DL/EDA/reports/eda_summary.json and DL/EDA/reports/DL_EDA_Report.md.

Data-use rules followed here:
* Decision-relevant analysis uses the development period only: training + validation hours.
  Test-period statistics appear only in the distribution-shift section, labelled descriptive.
* Targets: only windows kept by the dataset build (observed or unit-converted target hours).
* No scaler or transformation is fitted. The saved scalers are applied or inverted only to check the
  stored arrays; all statistics are computed in original units (hours, %, km/h, vessels).
* Reconstructed input values are kept and analysed, never removed or replaced.
"""
import hashlib
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.tsa.stattools import pacf  # noqa: E402

from cargomind_dl import config as data_config  # noqa: E402
from cargomind_dl import dataset as dl_dataset  # noqa: E402

EDA = data_config.EDA_STAGE  # DL/EDA
DATA = data_config.DATASET  # DL/Dataset/processed
PLOTS = EDA / "plots"
REPORTS = EDA / "reports"
SEED = 42

warnings.filterwarnings("ignore", category=RuntimeWarning)
SPLITS = ("train", "val", "test")
DEV = ("train", "val")
FEATURE_TYPES = {"dwell": "continuous (h)", "utilization": "continuous, bounded 0-100 %, right-censored at 100",
                 "labour_strike": "binary", "wind_gust": "continuous (km/h)", "disruption_active": "binary",
                 "disruption_hours_elapsed": "count-like hours, zero-inflated", "vessels_at_anchor": "count (vessels)"}
UNITS = {"dwell": "h", "utilization": "%", "labour_strike": "0/1", "wind_gust": "km/h", "disruption_active": "0/1",
         "disruption_hours_elapsed": "h", "vessels_at_anchor": "vessels"}
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eda100", "test": "#8a8985"}


def r(x, n=3):
    """Round for JSON / report (handles numpy scalars and NaN)."""
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, n)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return f"../plots/{name}"  # relative to DL/EDA/reports/DL_EDA_Report.md


# =============================================================================================
# Load
# =============================================================================================
def load():
    arrays = {s: dict(np.load(DATA / f"{s}.npz")) for s in SPLITS}
    schema = json.loads((DATA / "feature_schema.json").read_text())
    scalers = json.loads((DATA / "scalers.json").read_text())
    split_meta = json.loads((DATA / "split_metadata.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    windows = pd.read_csv(DATA / "windows.csv")
    panel = pd.read_csv(DATA / "hourly_panel.csv")
    panel["ts"] = pd.to_datetime(panel["ts_hour"], utc=True)
    counts = pd.read_csv(DATA / "sequence_counts.csv")
    base = pd.read_csv(DATA / "baseline_metrics.csv")
    return arrays, schema, scalers, split_meta, manifest, windows, panel, counts, base


# =============================================================================================
# A. Structure and sanity checks
# =============================================================================================
def sanity(arrays, schema, scalers, split_meta, manifest, windows, panel, counts):
    checks = []

    def add(name, ok, evidence):
        checks.append({"check": name, "result": "PASS" if ok else "FAIL", "evidence": evidence})

    feats = schema["arrays"]["X"]["axis_2"]
    ports = schema["arrays"]["port_index"]["ports"]
    prov_names = schema["arrays"]["provenance"]["axis_2"]
    dims = {}
    for s, a in arrays.items():
        n = len(a["y"])
        dims[s] = {k: list(v.shape) for k, v in a.items()}
        add(f"{s}: all arrays have the same number of windows", all(v.shape[0] == n for v in a.values()), f"N = {n}")
        add(f"{s}: X is [N, 12, 7] and provenance [N, 12, 5]", a["X"].shape[1:] == (12, len(feats)) and a["provenance"].shape[1:] == (12, len(prov_names)),
            f"X {a['X'].shape}, provenance {a['provenance'].shape}")
        add(f"{s}: X and y finite (no NaN / inf)", bool(np.isfinite(a["X"]).all() and np.isfinite(a["y"]).all() and np.isfinite(a["y_hours"]).all()), "")
        add(f"{s}: window count matches split_metadata.json", n == split_meta["kept_windows"][s], f"{n} vs {split_meta['kept_windows'][s]}")
        add(f"{s}: provenance codes only in {{0,1,2}}", bool(np.isin(a["provenance"], [0, 1, 2]).all()), "")
    # constants per split and channel
    const = {s: [feats[j] for j in range(len(feats)) if np.ptp(a["X"][..., j]) == 0] for s, a in arrays.items()}
    add("no channel is constant within any split", not any(const.values()), json.dumps(const))
    # windows.csv agreement
    kept = windows[windows["keep"]].set_index("window_id")
    all_ids = np.concatenate([a["window_id"] for a in arrays.values()])
    add("every window_id is unique across splits", len(all_ids) == len(set(all_ids)), f"{len(all_ids)} windows")
    ok_split, ok_port, ok_tgt = True, True, True
    for s, a in arrays.items():
        k = kept.loc[a["window_id"]]
        ok_split &= bool((k["split"] == s).all())
        ok_port &= bool((k["port_index"].values == a["port_index"]).all() and (k["port_id"].values == np.array(ports)[a["port_index"]]).all())
        ok_tgt &= bool((k["target_hour_index"].values == a["target_hour_index"]).all() and (k["target_hour_index"] - k["t_hour_index"] == 1).all()
                       and (k["target_status"] != "reconstructed").all())
    add("each window maps to exactly one kept windows.csv row with the same split", ok_split, "")
    add("port_index agrees with windows.csv port_id and schema port list", ok_port, ", ".join(ports))
    add("target hour = last input hour + 1, and no reconstructed target", ok_tgt, "")
    per_port = {s: pd.Series(np.array(ports)[a["port_index"]]).value_counts().to_dict() for s, a in arrays.items()}
    cnt = counts.pivot(index="port_id", columns="split", values="kept")
    add("per-port window counts match sequence_counts.csv", all(per_port[s][p] == cnt.loc[p, s] for s in SPLITS for p in ports), "")
    # manifest hashes
    bad = [f for f, h in manifest["outputs_sha256"].items()
           if (DATA / f).exists() and hashlib.sha256((DATA / f).read_bytes()).hexdigest() != h]
    add("DL/Dataset/processed files match manifest.json SHA-256", not bad, f"mismatched: {bad or 'none'}")
    # inverse transform with the saved scaler (no fitting)
    inv_err, last_err = 0.0, 0.0
    for s, a in arrays.items():
        inv_err = max(inv_err, float(np.abs(dl_dataset.to_hours(a["y"], a["port_index"], scalers) - a["y_hours"]).max()))
        last_err = max(last_err, float(np.abs(dl_dataset.to_hours(a["X"][:, -1, 0], a["port_index"], scalers) - a["last_dwell_hours"]).max()))
    add("saved dwell scaler inverts y to y_hours", inv_err < 1e-3, f"max |Δ| {inv_err:.1e} h")
    add("saved dwell scaler inverts X[:, -1, 0] to last_dwell_hours", last_err < 1e-3, f"max |Δ| {last_err:.1e} h")
    # every X value re-derived from the unscaled hourly panel with the saved (not refitted) scalers
    pv = panel.set_index(["port_id", "hour_index"])
    max_diff, prov_ok_all = 0.0, True
    for s, a in arrays.items():
        k = kept.loc[a["window_id"]]
        hours = k["t_hour_index"].values[:, None] + np.arange(-11, 1)[None, :]
        port = np.repeat(k["port_id"].values[:, None], 12, 1)
        rows = pv.loc[list(zip(port.ravel(), hours.ravel()))]
        rec = np.empty_like(a["X"])
        for j, f in enumerate(feats):
            v = rows[f].to_numpy(float)
            pid = port.ravel()
            if f in ("dwell", "vessels_at_anchor"):
                st = scalers[f]["per_port"]
                v = np.log1p(v) if f == "vessels_at_anchor" else v
                v = (v - np.array([st[p]["mean"] for p in pid])) / np.array([st[p]["std"] for p in pid])
            elif f == "wind_gust":
                v = (v - scalers[f]["mean"]) / scalers[f]["std"]
            elif f == "disruption_hours_elapsed":
                v = (np.log1p(v) - scalers[f]["mean"]) / scalers[f]["std"]
            elif f == "utilization":
                v = v / 100
            rec[..., j] = v.reshape(-1, 12)
        max_diff = max(max_diff, float(np.abs(rec - a["X"]).max()))
        prov_ok = np.array_equal(rows[[f"prov_{p}" for p in prov_names]].to_numpy(np.int8).reshape(-1, 12, len(prov_names)), a["provenance"])
        prov_ok_all &= prov_ok
    add("all X values reproduce from hourly_panel.csv with the saved scalers (12 consecutive hours per window)", max_diff < 1e-4, f"max |Δ| {max_diff:.1e}")
    add("provenance arrays equal the hourly_panel provenance codes", prov_ok_all, "all windows, all splits")
    tr = dl_dataset.load_split("train")
    add("reconstruction indicator is NOT in saved X; dataset.load_split appends it as channel 8",
        arrays["train"]["X"].shape[-1] == 7 and tr["X"].shape[-1] == 8 and tr["channels"][-1] == "dwell_reconstructed",
        f"loader channels: {tr['channels']}")
    scaled = {s: {"y_mean": r(a["y"].mean()), "y_std": r(a["y"].std()),
                  "X_channel_mean": [r(v) for v in a["X"].reshape(-1, 7).mean(0)], "X_channel_std": [r(v) for v in a["X"].reshape(-1, 7).std(0)]}
              for s, a in arrays.items()}
    return pd.DataFrame(checks), dims, per_port, scaled, const


# =============================================================================================
# Helpers for the hourly series
# =============================================================================================
def dev_panel(panel, split_meta):
    end = split_meta["bounds_hour_index"]["val"][1]
    return panel[panel["hour_index"] < end].copy()


def observed_dwell(p):
    """Dwell with reconstructed hours set to NaN (for statistics that must use observations only)."""
    return p["dwell"].where(p["prov_dwell"] != 2)


def acf_pairwise(x, lags):
    """ACF using only pairs where both values are observed (NaN = reconstructed)."""
    x = x - np.nanmean(x)
    out = []
    v = np.nanmean(x * x)
    for k in lags:
        a, b = x[:-k], x[k:]
        m = ~np.isnan(a) & ~np.isnan(b)
        out.append(float(np.mean(a[m] * b[m]) / v))
    return np.array(out)


def noise_decomposition(dwell_train):
    """Hourly white-noise SD from first differences: if dwell = smooth level + independent hourly noise,
    cov(Δ_t, Δ_{t-1}) = -σ² (the level contributes little at lag 1). No model is fitted.
    signal_share = 1 - σ² / var(dwell): share of the port's dwell variance that is not hourly noise.
    Under that assumption no forecast can have a one-step RMSE below σ."""
    d = np.diff(dwell_train)
    c1 = np.mean((d[:-1] - d.mean()) * (d[1:] - d.mean()))
    s2 = max(-c1, 0.0)
    return {"noise_sd_h": np.sqrt(s2), "dwell_sd_h": np.std(dwell_train), "signal_share": 1 - s2 / np.var(dwell_train),
            "lag1_acf_of_change": c1 / np.var(d)}


# =============================================================================================
# B. Target
# =============================================================================================
def describe(v):
    v = np.asarray(v, float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    return {"n": int(len(v)), "mean": r(v.mean()), "median": r(med), "std": r(v.std(ddof=1)), "min": r(v.min()), "max": r(v.max()),
            "q1": r(q1), "q3": r(q3), "iqr": r(iqr), "skewness": r(stats.skew(v)), "excess_kurtosis": r(stats.kurtosis(v)),
            "outliers_iqr_rule": int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum())}


def target_analysis(arrays, ports, panel, split_meta):
    res = {"units": "hours (y_hours; y is the per-port z-score of the same value)"}
    tgt = {s: pd.DataFrame({"port": np.array(ports)[arrays[s]["port_index"]], "y": arrays[s]["y_hours"],
                            "hour": arrays[s]["target_hour_index"]}) for s in SPLITS}
    for s in DEV:
        res[s] = describe(tgt[s]["y"])
    res["per_port"] = {}
    for p in ports:
        row = {}
        for s in DEV:
            v = tgt[s].loc[tgt[s]["port"] == p, "y"].values
            row[s] = describe(v)
            # outliers judged within port (pooled IQR mixes port levels)
        a, b = tgt["train"].loc[tgt["train"]["port"] == p, "y"], tgt["val"].loc[tgt["val"]["port"] == p, "y"]
        ks = stats.ks_2samp(a, b)
        row["val_minus_train_mean_h"] = r(b.mean() - a.mean())
        row["val_shift_in_train_sd"] = r((b.mean() - a.mean()) / a.std())
        row["ks_train_vs_val"] = {"statistic": r(ks.statistic), "p_value": float(f"{ks.pvalue:.2e}")}
        res["per_port"][p] = row
    ks = stats.ks_2samp(tgt["train"]["y"], tgt["val"]["y"])
    res["train_vs_val_pooled_ks"] = {"statistic": r(ks.statistic), "p_value": float(f"{ks.pvalue:.2e}")}
    # between-port share of target variance (eta squared), development targets
    dev = pd.concat([tgt["train"], tgt["val"]])
    gm = dev["y"].mean()
    ss_b = sum(len(g) * (g["y"].mean() - gm) ** 2 for _, g in dev.groupby("port"))
    ss_t = ((dev["y"] - gm) ** 2).sum()
    res["between_port_variance_share"] = r(ss_b / ss_t)
    res["share_of_total_sum_of_squares_by_port"] = {p: r(((g["y"] - gm) ** 2).sum() / ss_t) for p, g in dev.groupby("port")}
    res["within_port_outliers_iqr_rule"] = {p: int(sum(res["per_port"][p][s]["outliers_iqr_rule"] for s in DEV)) for p in ports}

    # ---- plots ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    bins = np.linspace(dev["y"].min(), dev["y"].max(), 60)
    for s in DEV:
        ax[0].hist(tgt[s]["y"], bins=bins, density=True, alpha=0.55, color=SPLIT_COLORS[s], label=f"{s} (n={len(tgt[s])})")
    ax[0].set(title="Next-hour dwell target, all ports pooled", xlabel="container dwell time (h)", ylabel="density")
    ax[0].legend()
    z = dev.assign(z=dev.groupby("port")["y"].transform(lambda v: (v - v.mean()) / v.std()))
    ax[1].hist(z["z"], bins=60, density=True, color="#2a78d6", alpha=0.7, label="within-port standardised")
    xx = np.linspace(-4, 6, 200)
    ax[1].plot(xx, stats.norm.pdf(xx), color="#0b0b0b", lw=1, label="standard normal")
    ax[1].set(title="Target after removing each port's level and spread (train + val)", xlabel="within-port z-score", ylabel="density")
    ax[1].legend()
    plots = [savefig(fig, "B1_target_distribution.png")]

    fig, ax = plt.subplots(figsize=(12, 4.6))
    order = dev.groupby("port")["y"].median().sort_values().index
    sns.boxplot(data=dev.assign(split=np.where(dev["hour"] < split_meta["bounds_hour_index"]["val"][0], "train", "val")),
                x="port", y="y", hue="split", order=order, palette=SPLIT_COLORS, fliersize=1.5, ax=ax)
    ax.set(title="Next-hour dwell target by port and split (development period)", xlabel="port", ylabel="container dwell time (h)")
    plots.append(savefig(fig, "B2_target_by_port_boxplot.png"))

    # time series with rolling mean / sd (24 h) for representative ports
    sd = dev.groupby("port")["y"].std().sort_values()
    reps = [sd.index[0], sd.index[len(sd) // 2], sd.index[-1]]
    dp = dev_panel(panel, split_meta)
    fig, axes = plt.subplots(len(reps), 1, figsize=(13, 2.9 * len(reps)), sharex=True)
    for ax, p in zip(axes, reps):
        g = dp[dp["port_id"] == p].set_index("ts")
        obs = observed_dwell(g)
        ax.plot(g.index, g["dwell"], color="#2a78d6", lw=0.6, alpha=0.8, label="hourly dwell")
        rec = g[g["prov_dwell"] == 2]
        ax.scatter(rec.index, rec["dwell"], s=7, color="#e34948", zorder=3, label="reconstructed hour")
        rm, rs = obs.rolling(24, min_periods=18).mean(), obs.rolling(24, min_periods=18).std()
        ax.plot(rm.index, rm, color="#0b0b0b", lw=1.4, label="24 h rolling mean (observed hours)")
        ax.fill_between(rm.index, rm - rs, rm + rs, color="#0b0b0b", alpha=0.12, label="± 24 h rolling SD")
        ax.axvline(pd.Timestamp(split_meta["bounds_utc"]["val"][0]), color="#eda100", ls="--", lw=1.2, label="validation starts")
        ax.set(ylabel="dwell (h)", title=f"{p}: hourly dwell, train + validation (within-port target SD {sd[p]:.1f} h)")
    axes[0].legend(fontsize=7.5, ncol=5, loc="upper left")
    axes[-1].set_xlabel("time (UTC)")
    plots.append(savefig(fig, "B3_target_timeseries_rolling.png"))
    res["representative_ports"] = {"lowest_variability": reps[0], "median_variability": reps[1], "highest_variability": reps[2]}

    fig, ax = plt.subplots(figsize=(12, 4.2))
    roll = []
    for p in ports:
        g = dp[dp["port_id"] == p].set_index("ts")
        s = observed_dwell(g).rolling(24, min_periods=18).std()
        ax.plot(s.index, s, lw=1, label=p)
        roll.append({"port": p, "rolling24_sd_median": r(s.median()), "rolling24_sd_max": r(s.max())})
    ax.axvline(pd.Timestamp(split_meta["bounds_utc"]["val"][0]), color="#0b0b0b", ls="--", lw=1)
    ax.set(title="24-hour rolling standard deviation of dwell (observed hours, train + val)", xlabel="time (UTC)", ylabel="rolling SD (h)")
    ax.legend(ncol=5, fontsize=8)
    plots.append(savefig(fig, "B4_rolling_sd_by_port.png"))
    res["rolling_24h_sd"] = roll
    return res, plots, tgt


# =============================================================================================
# C. Temporal
# =============================================================================================
def temporal_analysis(panel, split_meta, ports, windows, schema):
    dp = dev_panel(panel, split_meta)
    train_end = split_meta["bounds_hour_index"]["train"][1]
    lags = np.arange(1, 73)
    res = {"acf": {}, "acf_of_hourly_change": {}, "pacf": {}, "noise_decomposition_train": {}, "lag_corr_1_12": {}}
    acfs, pacfs = {}, {}
    for p in ports:
        g = dp[dp["port_id"] == p].sort_values("hour_index")
        x_obs = observed_dwell(g).to_numpy(float)
        a = acf_pairwise(x_obs, lags)
        a_all = acf_pairwise(g["dwell"].to_numpy(float), lags)
        acfs[p] = a
        res["acf"][p] = {f"lag_{k}": r(a[k - 1]) for k in (1, 2, 3, 6, 12, 24, 48, 72)}
        res["acf"][p]["max_abs_diff_vs_including_reconstructed"] = r(np.abs(a - a_all).max())
        d = np.diff(x_obs)  # NaN where either hour reconstructed
        ad = acf_pairwise(d, np.arange(1, 25))
        res["acf_of_hourly_change"][p] = {f"lag_{k}": r(ad[k - 1]) for k in (1, 2, 3, 24)}
        pa = pacf(g["dwell"].to_numpy(float), nlags=24, method="ywm")[1:]  # PACF needs a gap-free series
        pacfs[p] = pa
        res["pacf"][p] = {f"lag_{k}": r(pa[k - 1]) for k in (1, 2, 3, 6, 12, 24)}
        tr = g[g["hour_index"] < train_end]["dwell"].to_numpy(float)
        res["noise_decomposition_train"][p] = {k: r(v) for k, v in noise_decomposition(tr).items()}
        # lag correlations: corr(y_{t+1}, dwell_{t+1-k}), observed values only
        s = pd.Series(x_obs)
        res["lag_corr_1_12"][p] = {f"lag_{k}": r(s.corr(s.shift(k))) for k in range(1, 13)}
    ci = 1.96 / np.sqrt(len(dp) / len(ports))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for p in ports:
        axes[0].plot(lags, acfs[p], lw=1.1, label=p)
        axes[1].plot(np.arange(1, 25), pacfs[p], lw=1.1, marker="o", ms=2.5, label=p)
        dd = np.diff(observed_dwell(dp[dp["port_id"] == p].sort_values("hour_index")).to_numpy(float))
        axes[2].plot(np.arange(1, 25), acf_pairwise(dd, np.arange(1, 25)), lw=1.1, marker="o", ms=2.5)
    for ax in axes:
        ax.axhline(0, color="#0b0b0b", lw=0.7)
        ax.axhspan(-ci, ci, color="#8a8985", alpha=0.18)
    axes[0].axvline(12, color="#e34948", ls="--", lw=1, label="12 h window")
    axes[0].set(title="ACF of hourly dwell (observed pairs)", xlabel="lag (h)", ylabel="autocorrelation")
    axes[1].set(title="PACF of hourly dwell, lags 1–24", xlabel="lag (h)", ylabel="partial autocorrelation")
    axes[2].set(title="ACF of hour-to-hour dwell change", xlabel="lag (h)", ylabel="autocorrelation")
    axes[0].legend(fontsize=7, ncol=2)
    plots = [savefig(fig, "C1_acf_pacf_by_port.png")]

    # lag scatter for a quiet and an episodic port + lag correlation curve
    nd = res["noise_decomposition_train"]
    quiet = min(nd, key=lambda p: nd[p]["signal_share"])
    episodic = max(nd, key=lambda p: nd[p]["signal_share"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, p in zip(axes[:2], (quiet, episodic)):
        s = observed_dwell(dp[dp["port_id"] == p].sort_values("hour_index")).reset_index(drop=True)
        ax.scatter(s.shift(1), s, s=4, alpha=0.4, color="#2a78d6")
        lo, hi = np.nanmin(s), np.nanmax(s)
        ax.plot([lo, hi], [lo, hi], color="#0b0b0b", lw=0.8, ls="--")
        ax.set(title=f"{p}: dwell(t+1) vs dwell(t), r = {s.corr(s.shift(1)):.2f}", xlabel="dwell at hour t (h)", ylabel="dwell at hour t+1 (h)")
    for p in ports:
        v = res["lag_corr_1_12"][p]
        axes[2].plot(range(1, 13), [v[f"lag_{k}"] for k in range(1, 13)], marker="o", ms=3, lw=1, label=p)
    axes[2].set(title="corr(dwell(t+1), dwell(t+1−k)), k = 1..12", xlabel="lag k (h)", ylabel="Pearson r")
    axes[2].legend(fontsize=7, ncol=2)
    plots.append(savefig(fig, "C2_lag_relationships.png"))
    res["quietest_port"], res["most_episodic_port"] = quiet, episodic

    # diurnal / weekly profile of dwell deviations from a centred 24 h mean (descriptive)
    meta_tz = pd.read_csv(data_config.PROCESSED / "port_metadata.csv").set_index("port_id")["local_timezone"]
    prof = []
    for p in ports:
        g = dp[dp["port_id"] == p].sort_values("hour_index").copy()
        obs = observed_dwell(g)
        g["dev"] = obs - obs.rolling(25, center=True, min_periods=18).mean()
        g["local_hour"] = g["ts"].dt.tz_convert(meta_tz[p]).dt.hour
        g["dow"] = g["ts"].dt.tz_convert(meta_tz[p]).dt.dayofweek
        prof.append(g[["port_id", "local_hour", "dow", "dev"]])
    prof = pd.concat(prof).dropna()
    hr = prof.groupby("local_hour")["dev"].mean()
    f_hour = stats.f_oneway(*[g["dev"].values for _, g in prof.groupby("local_hour")])
    f_dow = stats.f_oneway(*[g["dev"].values for _, g in prof.groupby("dow")])
    eta_hour = prof.groupby("local_hour")["dev"].apply(lambda v: len(v) * (v.mean() - prof["dev"].mean()) ** 2).sum() / ((prof["dev"] - prof["dev"].mean()) ** 2).sum()
    res["diurnal"] = {"amplitude_h_max_minus_min_hourly_mean": r(hr.max() - hr.min()), "anova_p": float(f"{f_hour.pvalue:.2e}"),
                      "eta_squared": r(eta_hour, 4)}
    res["weekly"] = {"anova_p": float(f"{f_dow.pvalue:.2e}"), "weeks_covered": r(len(dp["hour_index"].unique()) / 168, 2),
                     "note": "fewer than 6 weeks of development data: a weekly pattern cannot be separated from individual episodes"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.lineplot(data=prof, x="local_hour", y="dev", errorbar=("ci", 95), ax=axes[0], color="#2a78d6")
    axes[0].set(title="Dwell deviation from centred 24 h mean, by local hour", xlabel="local hour of day", ylabel="deviation (h)")
    sns.lineplot(data=prof, x="dow", y="dev", errorbar=("ci", 95), ax=axes[1], color="#eda100")
    axes[1].set(title="Same deviation by local day of week", xlabel="day of week (0 = Monday)", ylabel="deviation (h)")
    for ax in axes:
        ax.axhline(0, color="#0b0b0b", lw=0.7)
    plots.append(savefig(fig, "C3_daily_weekly_profile.png"))

    # how much past is useful? descriptive trailing-mean / EWMA error on development targets (no fitting)
    rows = []
    kept = windows[windows["keep"] & windows["split"].isin(DEV)]
    for p in ports:
        g = dp[dp["port_id"] == p].sort_values("hour_index").set_index("hour_index")["dwell"]
        y = g.shift(-1)
        k_idx = kept.loc[kept["port_id"] == p, "t_hour_index"].values
        k_idx = k_idx[k_idx >= 47]  # same targets for every horizon length
        for k in (1, 2, 3, 6, 12, 24, 48):
            pred = g.rolling(k).mean()
            e = (pred - y).loc[k_idx]
            rows.append({"port": p, "summary": f"trailing mean {k} h", "length_h": k, "rmse": float(np.sqrt((e ** 2).mean()))})
        for a in (0.05, 0.1, 0.2, 0.3, 0.5):
            pred = g.ewm(alpha=a, adjust=False).mean()
            e = (pred - y).loc[k_idx]
            rows.append({"port": p, "summary": f"EWMA alpha {a} (full history)", "length_h": round(1 / a, 1), "rmse": float(np.sqrt((e ** 2).mean()))})
    wl = pd.DataFrame(rows)
    pooled = wl.groupby("summary")["rmse"].apply(lambda v: np.sqrt((v ** 2).mean())).sort_values()
    res["history_length_rmse_dev"] = {k: r(v) for k, v in pooled.items()}
    res["history_length_note"] = "Descriptive smoothing errors on development targets (t >= 47 h so all lengths are comparable); nothing fitted."
    fig, ax = plt.subplots(figsize=(10, 4.2))
    tm = wl[wl["summary"].str.startswith("trailing")]
    for p in ports:
        g = tm[tm["port"] == p]
        ax.plot(g["length_h"], g["rmse"], marker="o", ms=3, lw=1, label=p)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 6, 12, 24, 48], ["1", "2", "3", "6", "12", "24", "48"])
    ax.axvline(12, color="#e34948", ls="--", lw=1)
    ax.set(title="Next-hour RMSE of a trailing mean of k past hours (train + val targets)", xlabel="k (hours averaged)", ylabel="RMSE (h)")
    ax.legend(fontsize=7, ncol=5)
    plots.append(savefig(fig, "C4_history_length.png"))

    # does information older than the 12 h window add anything beyond the in-window EWMA?
    rows = []
    alpha = schema["baselines"]["ewma_alpha"]
    for p in ports:
        g = dp[dp["port_id"] == p].sort_values("hour_index").set_index("hour_index")["dwell"]
        k_idx = kept.loc[(kept["port_id"] == p) & (kept["t_hour_index"] >= 47), "t_hour_index"].values
        ew = kept.set_index(["port_id", "t_hour_index"]).loc[[(p, t) for t in k_idx], "baseline_ewma_alpha_0.2_hours"].values
        old = g.rolling(36).mean().shift(12).loc[k_idx].values  # mean of hours t-47 .. t-12
        y = g.shift(-1).loc[k_idx].values
        rows.append(pd.DataFrame({"port": p, "y": y, "ewma": ew, "old": old}))
    d = pd.concat(rows)
    w = lambda c: d[c] - d.groupby("port")[c].transform("mean")  # noqa: E731
    e_y = w("y") - np.polyfit(w("ewma"), w("y"), 1)[0] * w("ewma")
    e_o = w("old") - np.polyfit(w("ewma"), w("old"), 1)[0] * w("ewma")
    res["older_history_partial_corr"] = {"partial_r_mean_hours_t47_to_t12_given_window_ewma": r(np.corrcoef(e_y, e_o)[0, 1]),
                                         "n": int(len(d)), "ewma_alpha": alpha}

    # representative windows: inputs and next-hour target, in hours
    rng = np.random.default_rng(SEED)
    sample = kept[kept["split"] == "train"]
    pick = [sample[sample["port_id"] == quiet].sample(1, random_state=SEED), sample[sample["port_id"] == episodic].sample(1, random_state=SEED + 1),
            sample[(sample["n_reconstructed_dwell_inputs"] > 0)].sample(1, random_state=SEED + 2),
            sample.iloc[[rng.integers(len(sample))]]]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    pv = panel.set_index(["port_id", "hour_index"])
    for ax, w_ in zip(axes, pick):
        w_ = w_.iloc[0]
        hrs = np.arange(w_["t_hour_index"] - 11, w_["t_hour_index"] + 1)
        x = pv.loc[[(w_["port_id"], h) for h in hrs]]
        rel = np.arange(-11, 1)
        ax.plot(rel, x["dwell"], marker="o", ms=3.5, color="#2a78d6", label="input dwell")
        rr = x["prov_dwell"].values == 2
        ax.scatter(rel[rr], x["dwell"].values[rr], s=45, facecolors="none", edgecolors="#e34948", label="reconstructed input")
        ax.scatter([1], [w_["target_hours"]], color="#0b0b0b", marker="*", s=90, zorder=3, label="target (t+1)")
        ax.scatter([1], [w_["baseline_ewma_alpha_0.2_hours"]], color="#1b9e77", marker="x", s=50, label="EWMA forecast")
        ax.set(title=f"{w_['port_id']} · t = {w_['t_ts'][:16]}", xlabel="hour relative to t", ylabel="dwell (h)")
    axes[0].legend(fontsize=7)
    plots.append(savefig(fig, "C5_example_windows.png"))
    return res, plots


# =============================================================================================
# D. Features
# =============================================================================================
def feature_analysis(panel, split_meta, ports, feats):
    dp = dev_panel(panel, split_meta)
    res = {}
    for f in feats:
        v = dp[f].astype(float)
        bt = dp.groupby("port_id")[f]
        eta = (bt.size() * (bt.mean() - v.mean()) ** 2).sum() / ((v - v.mean()) ** 2).sum() if v.var() > 0 else np.nan
        d = {"type": FEATURE_TYPES[f], "unit": UNITS[f], "min": r(v.min()), "p01": r(v.quantile(0.01)), "median": r(v.median()),
             "p99": r(v.quantile(0.99)), "max": r(v.max()), "mean": r(v.mean()), "std": r(v.std()), "n_unique": int(v.nunique()),
             "share_zero": r((v == 0).mean()), "between_port_variance_share": r(eta),
             "median_within_port_std": r(bt.std().median()),
             "per_port_mean": {p: r(m) for p, m in bt.mean().items()}, "per_port_std": {p: r(m) for p, m in bt.std().items()}}
        if f == "utilization":
            d["share_at_100"] = r((v >= 99.99).mean())
            d["per_port_share_at_100"] = {p: r((g >= 99.99).mean()) for p, g in bt}
        if f in ("labour_strike", "disruption_active"):
            d["prevalence"] = r(v.mean(), 4)
            d["per_port_prevalence"] = {p: r(g.mean(), 4) for p, g in bt}
            d["episodes"] = int(((dp.sort_values(["port_id", "hour_index"]).groupby("port_id")[f].diff() == 1)).sum()
                                + (dp[dp["hour_index"] == 0][f] == 1).sum())
        if f == "disruption_hours_elapsed":
            act = v[dp["disruption_active"] == 1]
            d["when_active"] = {"n_hours": int(len(act)), "median": r(act.median()), "p90": r(act.quantile(0.9)), "max": r(act.max())}
        if f == "vessels_at_anchor":
            d["per_port_share_zero"] = {p: r((g == 0).mean()) for p, g in bt}
            d["index_of_dispersion_median"] = r((bt.var() / bt.mean()).median())
        if f == "wind_gust":
            d["clipped_negative_hours"] = int(dp["wind_gust_clipped"].sum())
        res[f] = d

    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for ax, f in zip(axes.ravel(), feats):
        v = dp[f].astype(float)
        if f in ("labour_strike", "disruption_active"):
            share = dp.groupby("port_id")[f].mean().reindex(ports)
            ax.bar(ports, share * 100, color="#e34948")
            ax.set(title=f"{f}: share of hours = 1", xlabel="port", ylabel="% of development hours")
            ax.tick_params(axis="x", rotation=60, labelsize=7)
        elif f == "disruption_hours_elapsed":
            ax.hist(v[dp["disruption_active"] == 1], bins=40, color="#7b3fb5")
            ax.set(title=f"{f} (active hours only; {100 * (v == 0).mean():.0f} % of all hours are 0)", xlabel="hours since disruption began",
                   ylabel="hours")
            ax.title.set_fontsize(9)
        else:
            ax.hist(v, bins=50, color="#2a78d6")
            ax.set(title=f"{f} (all ports, train + val)", xlabel=f"{f} ({UNITS[f]})", ylabel="hours")
    axes.ravel()[-1].axis("off")
    plots = [savefig(fig, "D1_feature_distributions.png")]

    cont = ["dwell", "utilization", "wind_gust", "vessels_at_anchor"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    for ax, f in zip(axes, cont):
        sns.boxplot(data=dp, x="port_id", y=f, order=ports, ax=ax, color="#2a78d6", fliersize=1)
        ax.set(title=f"{f} by port", xlabel="port", ylabel=f"{f} ({UNITS[f]})")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
    plots.append(savefig(fig, "D2_features_by_port.png"))

    day = dp.assign(day=dp["ts"].dt.floor("D"))
    fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)
    for ax, f, cmap in zip(axes, ["disruption_active", "labour_strike"], ["Reds", "Purples"]):
        m = day.groupby(["port_id", "day"])[f].mean().unstack().reindex(ports)
        ax.imshow(m.values, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(ports)), ports, fontsize=7)
        ax.set(title=f"{f}: share of each day's hours (train + val)", ylabel="port")
    m = day.groupby(["port_id", "day"])["wind_gust"].max().unstack().reindex(ports)
    im = axes[2].imshow(m.values, aspect="auto", cmap="Blues", interpolation="nearest")
    axes[2].set_yticks(range(len(ports)), ports, fontsize=7)
    axes[2].set(title="wind_gust: daily maximum (km/h)", ylabel="port", xlabel="day")
    days = m.columns
    axes[2].set_xticks(range(0, len(days), 4), [str(d.date()) for d in days[::4]], rotation=45, fontsize=7)
    fig.colorbar(im, ax=axes[2], fraction=0.02, label="km/h")
    plots.append(savefig(fig, "D3_exposure_over_time.png"))
    return res, plots


# =============================================================================================
# E. Feature-target relationships
# =============================================================================================
def relationship_frame(windows, panel, feats):
    k = windows[windows["keep"] & windows["split"].isin(DEV)].copy()
    pv = panel.set_index(["port_id", "hour_index"])
    for lag in range(0, 12):
        rows = pv.loc[list(zip(k["port_id"], k["t_hour_index"] - lag)), feats]
        for f in feats:
            k[f"{f}_lag{lag}"] = rows[f].values
    k["ewma"] = k["baseline_ewma_alpha_0.2_hours"]
    k["resid"] = k["target_hours"] - k["ewma"]
    return k


def _within(d, c):
    return d[c] - d.groupby("port_id")[c].transform("mean")


def _partial(d, x, y, control):
    xc, yc, cc = _within(d, x), _within(d, y), _within(d, control)
    ex = xc - np.polyfit(cc, xc, 1)[0] * cc
    ey = yc - np.polyfit(cc, yc, 1)[0] * cc
    return float(np.corrcoef(ex, ey)[0, 1]) if ex.std() > 0 else np.nan


def relationships(windows, panel, ports, feats):
    k = relationship_frame(windows, panel, feats)
    res = {"n_windows": int(len(k)), "control": "EWMA (alpha 0.2) of the 12 input dwell hours, within port", "features": {}}
    ops = [f for f in feats if f != "dwell"]
    for f in feats:
        c = f"{f}_lag0"
        d = {"pearson_pooled": r(k[c].corr(k["target_hours"])), "spearman_pooled": r(stats.spearmanr(k[c], k["target_hours"])[0]),
             "pearson_within_port": r(_within(k, c).corr(_within(k, "target_hours"))),
             "partial_given_last_dwell_within_port": r(_partial(k, c, "target_hours", "dwell_lag0")) if f != "dwell" else None,
             "partial_given_window_ewma_within_port": r(_partial(k, c, "target_hours", "ewma")),
             "partial_by_lag_given_ewma": {f"lag{l}": r(_partial(k, f"{f}_lag{l}", "target_hours", "ewma")) for l in range(12)}}
        per = {}
        for p, g in k.groupby("port_id"):
            pr = _partial(g, c, "target_hours", "ewma")
            n = len(g)
            if np.isfinite(pr):
                z, se = np.arctanh(pr), 1 / np.sqrt(n - 4)
                per[p] = {"partial_r": r(pr), "ci95_naive": [r(np.tanh(z - 1.96 * se)), r(np.tanh(z + 1.96 * se))], "n": n}
            else:
                per[p] = {"partial_r": None, "ci95_naive": None, "n": n, "note": "feature constant at this port in train + val"}
        d["partial_given_ewma_by_port"] = per
        vals = [v["partial_r"] for v in per.values() if v["partial_r"] is not None]
        d["by_port_range"] = [r(min(vals)), r(max(vals))] if vals else None
        res["features"][f] = d
    # grouped summaries for binary features and elapsed bins
    grp = {}
    for f in ("labour_strike", "disruption_active"):
        g = k.groupby(f"{f}_lag0").agg(n=("resid", "size"), target_mean_h=("target_hours", "mean"),
                                       residual_mean_h=("resid", "mean"), residual_sd_h=("resid", "std"))
        grp[f] = {str(int(i)): {c: r(v) for c, v in row.items()} for i, row in g.iterrows()}
        a, b = k.loc[k[f"{f}_lag0"] == 1, "resid"], k.loc[k[f"{f}_lag0"] == 0, "resid"]
        grp[f]["mann_whitney_p_residual"] = float(f"{stats.mannwhitneyu(a, b).pvalue:.2e}")
    bins = pd.cut(k["disruption_hours_elapsed_lag0"], [-0.1, 0, 6, 24, 48, 200], labels=["0 (none)", "1-6", "7-24", "25-48", ">48"])
    k["elapsed_bin"] = bins
    g = k.groupby("elapsed_bin", observed=True).agg(n=("resid", "size"), residual_mean_h=("resid", "mean"))
    grp["disruption_hours_elapsed_bins"] = {str(i): {c: r(v) for c, v in row.items()} for i, row in g.iterrows()}
    res["grouped"] = grp
    res["note"] = ("Naive per-port intervals assume independent windows; consecutive windows overlap and share episodes, "
                   "so true uncertainty is wider.")

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
    tab = pd.DataFrame({"pooled Pearson": [res["features"][f]["pearson_pooled"] for f in feats],
                        "within-port Pearson": [res["features"][f]["pearson_within_port"] for f in feats],
                        "partial | last dwell": [res["features"][f]["partial_given_last_dwell_within_port"] for f in feats],
                        "partial | window EWMA": [res["features"][f]["partial_given_window_ewma_within_port"] for f in feats]},
                       index=feats).astype(float)
    sns.heatmap(tab, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[0], cbar_kws={"label": "correlation"})
    axes[0].set(title="Feature at hour t vs dwell at t+1 (train + val)", xlabel="measure", ylabel="feature")
    per = pd.DataFrame({f: {p: res["features"][f]["partial_given_ewma_by_port"][p]["partial_r"] for p in ports} for f in feats}).T.astype(float)
    sns.heatmap(per, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6, ax=axes[1], cbar_kws={"label": "partial r"})
    axes[1].set(title="Partial r with next-hour dwell given window EWMA, by port", xlabel="port", ylabel="feature")
    plots = [savefig(fig, "E1_feature_target_correlations.png")]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for f in ops:
        v = res["features"][f]["partial_by_lag_given_ewma"]
        ax.plot(range(12), [v[f"lag{l}"] for l in range(12)], marker="o", ms=3, label=f)
    ax.axhline(0, color="#0b0b0b", lw=0.7)
    ax.set(title="Partial r of feature at hour t−lag with dwell at t+1, given window EWMA", xlabel="lag (h before t)", ylabel="partial r (within port)")
    ax.legend(fontsize=8)
    plots.append(savefig(fig, "E2_lagged_partial_correlations.png"))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, f in zip(axes[:2], ("labour_strike", "disruption_active")):
        sns.boxplot(data=k, x=f"{f}_lag0", y="resid", ax=ax, color="#2a78d6", fliersize=1)
        ax.axhline(0, color="#0b0b0b", lw=0.7)
        ax.set(title=f"Target minus EWMA by {f} at t", xlabel=f"{f} at hour t", ylabel="dwell(t+1) − EWMA (h)")
    sns.boxplot(data=k, x="elapsed_bin", y="resid", ax=axes[2], color="#7b3fb5", fliersize=1)
    axes[2].axhline(0, color="#0b0b0b", lw=0.7)
    axes[2].set(title="Target minus EWMA by hours since disruption began", xlabel="hours since disruption began", ylabel="dwell(t+1) − EWMA (h)")
    x = _within(k, "vessels_at_anchor_lag0")
    axes[3].hexbin(x, k["resid"], gridsize=35, cmap="Blues", mincnt=1)
    axes[3].set(title="Vessels at anchor (port-demeaned) vs target minus EWMA", xlabel="vessels at anchor − port mean", ylabel="dwell(t+1) − EWMA (h)")
    plots.append(savefig(fig, "E3_binary_and_anchor_vs_residual.png"))
    return res, plots, k


# =============================================================================================
# F. Reconstruction
# =============================================================================================
def reconstruction(panel, windows, ports, split_meta, k_dev):
    res = {}
    b = split_meta["bounds_hour_index"]
    panel = panel.copy()
    kept = windows[windows["keep"]]
    res["windows_with_reconstructed_dwell_input_pct"] = {s: r(100 * (kept.loc[kept["split"] == s, "n_reconstructed_dwell_inputs"] > 0).mean(), 1) for s in SPLITS}
    res["windows_with_any_reconstructed_input_pct"] = {s: r(100 * (kept.loc[kept["split"] == s, "n_reconstructed_input_cells"] > 0).mean(), 1) for s in SPLITS}
    cells = panel.groupby(["port_id", "split_of_hour"])["prov_dwell"].apply(lambda v: 100 * (v == 2).mean()).unstack()[list(SPLITS)]
    res["reconstructed_dwell_hours_pct_by_port_split"] = {p: {s: r(cells.loc[p, s], 2) for s in SPLITS} for p in ports}
    wpct = kept.groupby(["port_id", "split"])["n_reconstructed_dwell_inputs"].apply(lambda v: 100 * (v > 0).mean()).unstack()[list(SPLITS)]
    res["windows_with_reconstructed_dwell_pct_by_port_split"] = {p: {s: r(wpct.loc[p, s], 1) for s in SPLITS} for p in ports}
    # temporal concentration: per-day counts vs a uniform rate (chi-square), development hours
    dp = dev_panel(panel, split_meta)
    daily = dp.assign(day=dp["ts"].dt.floor("D")).groupby("day")["prov_dwell"].apply(lambda v: int((v == 2).sum()))
    exp = daily.sum() / len(daily)
    chi = stats.chisquare(daily.values, np.full(len(daily), exp))
    res["daily_concentration_dev"] = {"days": int(len(daily)), "mean_per_day": r(exp, 2), "max_per_day": int(daily.max()),
                                      "chi2_uniform_p": float(f"{chi.pvalue:.2e}")}
    port_counts = dp.groupby("port_id")["prov_dwell"].apply(lambda v: int((v == 2).sum()))
    res["port_concentration_dev"] = {"counts": port_counts.to_dict(),
                                     "chi2_uniform_p": float(f"{stats.chisquare(port_counts.values).pvalue:.2e}")}
    # are reconstructed values distributed like observed ones? compare deviation from the median of the
    # observed hours within ±6 h (excluding the hour itself), per port
    devs = []
    for p in ports:
        g = dp[dp["port_id"] == p].sort_values("hour_index").reset_index(drop=True)
        obs = observed_dwell(g)
        for i in range(len(g)):
            nb = obs.iloc[max(0, i - 6):i].tolist() + obs.iloc[i + 1:i + 7].tolist()
            nb = [v for v in nb if np.isfinite(v)]
            if len(nb) >= 6:
                devs.append({"port": p, "reconstructed": g.loc[i, "prov_dwell"] == 2, "dev": g.loc[i, "dwell"] - np.median(nb)})
    devs = pd.DataFrame(devs)
    a, o = devs.loc[devs["reconstructed"], "dev"], devs.loc[~devs["reconstructed"], "dev"]
    ks = stats.ks_2samp(a, o)
    res["value_check_vs_local_observed_median"] = {
        "n_reconstructed": int(len(a)), "n_observed": int(len(o)),
        "reconstructed_mean_sd_h": [r(a.mean()), r(a.std())], "observed_mean_sd_h": [r(o.mean()), r(o.std())],
        "ks_statistic": r(ks.statistic), "ks_p": float(f"{ks.pvalue:.2e}"),
        "levene_p_equal_spread": float(f"{stats.levene(a, o).pvalue:.2e}")}
    # target behaviour: EWMA error in windows with vs without reconstructed dwell inputs (train + val)
    k = k_dev.copy()
    k["has_rec"] = k["n_reconstructed_dwell_inputs"] > 0
    pv = panel.set_index(["port_id", "hour_index"])
    k["last_rec"] = pv.loc[list(zip(k["port_id"], k["t_hour_index"])), "prov_dwell"].values == 2
    k["abs_err_ewma"] = k["resid"].abs()
    k["abs_err_persist"] = (k["target_hours"] - k["baseline_persistence_hours"]).abs()
    grp = {}
    for col, name in [("has_rec", "any reconstructed dwell input"), ("last_rec", "dwell at hour t reconstructed")]:
        g = k.groupby(col).agg(n=("resid", "size"), target_mean_h=("target_hours", "mean"),
                               ewma_rmse_h=("resid", lambda v: np.sqrt((v ** 2).mean())),
                               persistence_rmse_h=("abs_err_persist", lambda v: np.sqrt((v ** 2).mean())))
        grp[name] = {str(bool(i)): {c: r(v) for c, v in row.items()} for i, row in g.iterrows()}
        grp[name]["mann_whitney_p_abs_ewma_error"] = float(f"{stats.mannwhitneyu(k.loc[k[col], 'abs_err_ewma'], k.loc[~k[col], 'abs_err_ewma']).pvalue:.2e}")
    # same comparison within port (removes port-mix effects)
    within = k.assign(sq=k["resid"] ** 2).groupby(["port_id", "has_rec"])["sq"].mean().unstack()
    res["ewma_rmse_by_port_with_vs_without_reconstructed"] = {p: [r(np.sqrt(within.loc[p, False])), r(np.sqrt(within.loc[p, True]))] for p in ports}
    res["target_behaviour_dev"] = grp

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    sns.heatmap(cells.reindex(ports), annot=True, fmt=".1f", cmap="Reds", ax=axes[0], cbar_kws={"label": "% of hours"})
    axes[0].set(title="Reconstructed dwell hours (% of hours)", xlabel="split", ylabel="port")
    sns.heatmap(wpct.reindex(ports), annot=True, fmt=".0f", cmap="Reds", ax=axes[1], cbar_kws={"label": "% of windows"})
    axes[1].set(title="Kept windows with ≥ 1 reconstructed dwell input (%)", xlabel="split", ylabel="port")
    axes[2].bar(range(len(daily)), daily.values, color="#e34948")
    axes[2].axhline(exp, color="#0b0b0b", ls="--", lw=1, label="uniform expectation")
    axes[2].set(title="Reconstructed dwell hours per day, all ports (train + val)", xlabel="day index from 2025-01-01", ylabel="reconstructed hours")
    axes[2].legend()
    plots = [savefig(fig, "F1_reconstruction_by_port_split_time.png")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    bins = np.linspace(-20, 20, 50)
    axes[0].hist(o, bins=bins, density=True, alpha=0.6, color="#2a78d6", label=f"observed (n={len(o)})")
    axes[0].hist(a, bins=bins, density=True, alpha=0.6, color="#e34948", label=f"reconstructed (n={len(a)})")
    axes[0].set(title="Dwell minus median of observed neighbours (±6 h)", xlabel="deviation (h)", ylabel="density")
    axes[0].legend()
    sns.boxplot(data=k.assign(group=np.where(k["has_rec"], "≥1 reconstructed dwell input", "all dwell inputs observed")),
                x="group", y="resid", ax=axes[1], color="#2a78d6", fliersize=1)
    axes[1].axhline(0, color="#0b0b0b", lw=0.7)
    axes[1].set(title="Target minus EWMA, by window input provenance (train + val)", xlabel="window group", ylabel="dwell(t+1) − EWMA (h)")
    plots.append(savefig(fig, "F2_reconstructed_vs_observed.png"))
    return res, plots


# =============================================================================================
# G. Ports
# =============================================================================================
def port_comparison(panel, split_meta, ports, target_res, temporal_res, feat_res, recon_res, arrays):
    dp = dev_panel(panel, split_meta)
    rows = []
    for p in ports:
        g = dp[dp["port_id"] == p]
        tp = target_res["per_port"][p]["train"]
        nd = temporal_res["noise_decomposition_train"][p]
        rows.append({"port": p, "windows_train": int((np.array(ports)[arrays["train"]["port_index"]] == p).sum()),
                     "windows_val": int((np.array(ports)[arrays["val"]["port_index"]] == p).sum()),
                     "target_mean_train_h": tp["mean"], "target_sd_train_h": tp["std"],
                     "acf_lag1": temporal_res["acf"][p]["lag_1"], "acf_lag12": temporal_res["acf"][p]["lag_12"],
                     "noise_sd_h": nd["noise_sd_h"], "signal_share": nd["signal_share"],
                     "disruption_hours_pct": r(100 * g["disruption_active"].mean(), 1), "strike_hours_pct": r(100 * g["labour_strike"].mean(), 1),
                     "anchor_mean": r(g["vessels_at_anchor"].mean(), 2), "utilization_mean_pct": r(g["utilization"].mean(), 1),
                     "utilization_at_100_pct": r(100 * (g["utilization"] >= 99.99).mean(), 1),
                     "reconstructed_dwell_pct_dev": r(100 * (g["prov_dwell"] == 2).mean(), 2)})
    tab = pd.DataFrame(rows)
    res = {"table": tab.to_dict("records"),
           "window_share_per_port_train": r(tab["windows_train"].max() / tab["windows_train"].sum(), 3),
           "target_sum_of_squares_share_top_port": max(target_res["share_of_total_sum_of_squares_by_port"].items(), key=lambda kv: kv[1])}

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    t = tab.set_index("port")
    axes[0].bar(t.index, t["noise_sd_h"], color="#8a8985", label="hourly noise SD")
    axes[0].bar(t.index, t["target_sd_train_h"], color="none", edgecolor="#2a78d6", lw=1.5, label="total target SD")
    axes[0].set(title="Total dwell SD vs estimated hourly noise SD (train)", xlabel="port", ylabel="hours")
    axes[0].legend(fontsize=8)
    axes[1].bar(t.index, t["signal_share"] * 100, color="#1b9e77")
    axes[1].set(title="Share of dwell variance that is not hourly noise (train)", xlabel="port", ylabel="signal share (%)")
    x = np.arange(len(t))
    axes[2].bar(x - 0.2, t["disruption_hours_pct"], width=0.4, color="#e34948", label="disruption active")
    axes[2].bar(x + 0.2, t["strike_hours_pct"], width=0.4, color="#7b3fb5", label="labour strike")
    axes[2].set_xticks(x, t.index)
    axes[2].set(title="Exposure: % of development hours", xlabel="port", ylabel="% of hours")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", rotation=60, labelsize=8)
    plots = [savefig(fig, "G1_port_comparison.png")]
    return res, plots


# =============================================================================================
# H. Distribution shift
# =============================================================================================
def shift(panel, ports, feats, arrays, tgt):
    res = {"train_vs_val": {}, "test_descriptive_only": {}}
    ph = panel.copy()
    for other, key in (("val", "train_vs_val"), ("test", "test_descriptive_only")):
        out = {}
        for f in feats:
            a = ph.loc[ph["split_of_hour"] == "train", f].astype(float)
            b = ph.loc[ph["split_of_hour"] == other, f].astype(float)
            ks = stats.ks_2samp(a, b)
            # standardised mean difference within port, averaged
            smd = []
            for p in ports:
                pa = ph.loc[(ph["split_of_hour"] == "train") & (ph["port_id"] == p), f].astype(float)
                pb = ph.loc[(ph["split_of_hour"] == other) & (ph["port_id"] == p), f].astype(float)
                sd = pa.std()
                smd.append((pb.mean() - pa.mean()) / sd if sd > 0 else np.nan)
            out[f] = {"train_mean": r(a.mean()), f"{other}_mean": r(b.mean()), "ks_statistic": r(ks.statistic), "ks_p": float(f"{ks.pvalue:.2e}"),
                      "per_port_smd": {p: r(v) for p, v in zip(ports, smd)}, "mean_abs_per_port_smd": r(np.nanmean(np.abs(smd)))}
        ta, tb = tgt["train"], tgt[other]
        out["target"] = {"train_mean_h": r(ta["y"].mean()), f"{other}_mean_h": r(tb["y"].mean()),
                         "per_port_mean_shift_in_train_sd": {p: r((tb.loc[tb["port"] == p, "y"].mean() - ta.loc[ta["port"] == p, "y"].mean())
                                                                  / ta.loc[ta["port"] == p, "y"].std()) for p in ports},
                         "scaled_y_mean": r(arrays[other]["y"].mean()), "scaled_y_train_mean": r(arrays["train"]["y"].mean())}
        res[key] = out
    prev = ph.groupby(["port_id", "split_of_hour"])[["labour_strike", "disruption_active"]].mean().unstack()
    res["strike_hours_pct_by_port_split"] = {p: {sp: r(100 * prev.loc[p, ("labour_strike", sp)], 1) for sp in SPLITS} for p in ports}
    res["disruption_hours_pct_by_port_split"] = {p: {sp: r(100 * prev.loc[p, ("disruption_active", sp)], 1) for sp in SPLITS} for p in ports}
    res["test_note"] = "Test statistics are descriptive warnings only; no feature or modelling decision in this EDA uses them."

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    smd = pd.DataFrame({f: res["train_vs_val"][f]["per_port_smd"] for f in feats}).T
    smd.loc["target (dwell t+1)"] = pd.Series(res["train_vs_val"]["target"]["per_port_mean_shift_in_train_sd"])
    sns.heatmap(smd.astype(float), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1.5, vmax=1.5, ax=axes[0],
                cbar_kws={"label": "(val mean − train mean) / train SD"})
    axes[0].set(title="Train → validation shift by port (standardised mean difference)", xlabel="port", ylabel="feature")
    x = np.arange(len(ports))
    for i, (sp, col) in enumerate((("train", "#2a78d6"), ("val", "#eda100"))):
        share = ph[ph["split_of_hour"] == sp].groupby("port_id")[["disruption_active", "labour_strike"]].mean().reindex(ports) * 100
        axes[1].bar(x + (i - 0.5) * 0.4, share["disruption_active"], width=0.4, color=col, label=f"disruption active, {sp}")
        axes[1].scatter(x + (i - 0.5) * 0.4, share["labour_strike"], color="#0b0b0b", marker="_" if sp == "train" else "x", s=60,
                        label=f"labour strike, {sp}", zorder=3)
    axes[1].set_xticks(x, ports, rotation=60, fontsize=8)
    axes[1].set(title="Disruption (bars) and strike (markers) exposure: train vs validation", xlabel="port", ylabel="% of hours")
    axes[1].legend(fontsize=7)
    plots = [savefig(fig, "H1_train_val_shift.png")]
    return res, plots


# =============================================================================================
# Main
# =============================================================================================
def analyse():
    PLOTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False})
    arrays, schema, scalers, split_meta, manifest, windows, panel, counts, base = load()
    ports = schema["arrays"]["port_index"]["ports"]
    feats = schema["arrays"]["X"]["axis_2"]
    checks, dims, per_port, scaled, const = sanity(arrays, schema, scalers, split_meta, manifest, windows, panel, counts)
    target_res, p_b, tgt = target_analysis(arrays, ports, panel, split_meta)
    temporal_res, p_c = temporal_analysis(panel, split_meta, ports, windows, schema)
    feat_res, p_d = feature_analysis(panel, split_meta, ports, feats)
    rel_res, p_e, k_dev = relationships(windows, panel, ports, feats)
    rec_res, p_f = reconstruction(panel, windows, ports, split_meta, k_dev)
    port_res, p_g = port_comparison(panel, split_meta, ports, target_res, temporal_res, feat_res, rec_res, arrays)
    shift_res, p_h = shift(panel, ports, feats, arrays, tgt)
    bl = base[base["port_id"] == "ALL"].pivot(index="baseline", columns="split", values="rmse_h").round(3)
    summary = {
        "dataset_dimensions": {"arrays": dims, "windows_per_port": per_port, "window_hours": schema["window_hours"],
                               "horizon_hours": schema["horizon_hours"], "ports": ports, "scaled_array_stats": scaled,
                               "split_bounds_hour_index": split_meta["bounds_hour_index"], "split_bounds_utc": split_meta["bounds_utc"]},
        "sanity_checks": checks.to_dict("records"),
        "features": {f: {"position": i, "type": FEATURE_TYPES[f], "unit": UNITS[f],
                         "scaling": next(x["scaling"] for x in schema["features"] if x["name"] == f)} for i, f in enumerate(feats)},
        "loader_default_channels": dl_dataset.load_split("val")["channels"],
        "target": target_res, "temporal": temporal_res, "feature_analysis": feat_res, "feature_target_relationships": rel_res,
        "reconstruction": rec_res, "ports": port_res, "distribution_shift": shift_res,
        "baseline_rmse_h_from_dataset_stage": bl.to_dict(),
    }
    plots = {"target": p_b, "temporal": p_c, "features": p_d, "relationships": p_e, "reconstruction": p_f, "ports": p_g, "shift": p_h}
    return summary, plots, checks


if __name__ == "__main__":
    summary, plots, checks = analyse()
    from report import write_report  # noqa: E402  (DL/EDA/scripts/report.py)

    summary["recommendations"], report_text = write_report(summary, plots)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (REPORTS / "DL_EDA_Report.md").write_text(report_text)
    print(checks["result"].value_counts().to_dict(), "| plots:", sum(len(v) for v in plots.values()))
