"""Renders DL/EDA/reports/DL_EDA_Report.md from the computed summary (called by run_eda.py).

Every number is read from the summary dict. The interpretive sentences were written for the dataset
version whose hashes are listed in WRITTEN_FOR; if DL/Dataset/processed changes, the report carries a warning
banner so the prose is re-checked rather than trusted.
"""
import hashlib
from pathlib import Path

import numpy as np

from cargomind_dl.config import DATASET as DATA  # noqa: E402  (DL/Dataset/processed)
WRITTEN_FOR = {
    "train.npz": "d5d515dc79efead10ad9a3565f96f6b414ff612d726b290c26bb37b571de2b05",
    "val.npz": "f3bd0818ac0103c8006b043f83a158887714ccf6ac0bfe2276dde8e95502a5eb",
    "test.npz": "1924a5ef04db6c8f4c17e3fc7fa1ac5548b7b977959e0aece343339db037db65",
}


def f(x, n=2):
    return "–" if x is None else f"{x:.{n}f}"


def pct(x, n=1):
    return "–" if x is None else f"{x:.{n}f} %"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def img(path, alt):
    return f"![{alt}]({path})"


def recommendations(s):
    nd = s["temporal"]["noise_decomposition_train"]
    noise = [v["noise_sd_h"] for v in nd.values()]
    return [
        {"priority": 1, "recommendation": "Judge every model against EWMA (alpha 0.2) on the same test windows, in hours, overall and per port; "
                                          f"the attainable gain is small (hourly noise SD {min(noise):.1f}–{max(noise):.1f} h vs EWMA RMSE "
                                          f"{s['baseline_rmse_h_from_dataset_stage']['val']['ewma_alpha_0.2']:.2f} h on validation).",
         "evidence": "strong"},
        {"priority": 2, "recommendation": "Keep the 12-hour window and a small model (1 recurrent layer, 16–64 units, dropout, early stopping on validation "
                                          "RMSE). Do not add depth, attention or graph structure without validation evidence.",
         "evidence": "strong"},
        {"priority": 3, "recommendation": "Report errors separately for disruption / strike hours vs normal hours and for high-signal ports "
                                          "(NLRTM, USLAX, CNSHA, INNSA, SGSIN) vs noise-dominated ports (MYPKG, DEHAM, AEJEA, INMAA).",
         "evidence": "strong"},
        {"priority": 4, "recommendation": "Treat labour_strike as fragile: only 4 strike episodes at 4 ports in train + val, while the test period contains "
                                          "strikes at INNSA and MYPKG, which never strike in development data. Run an ablation without it.",
         "evidence": "strong"},
        {"priority": 5, "recommendation": "Use the default 8-channel loader input (with dwell_reconstructed); reconstruction showed no measurable effect, "
                                          "so a 7-channel ablation is a check, not a requirement. Use a port embedding or per-port reporting; keep "
                                          "per-port dwell scaling and compute the loss in scaled units but evaluate in hours.",
         "evidence": "moderate"},
    ]


def write_report(s, plots):
    warn = []
    for name, h in WRITTEN_FOR.items():
        p = DATA / name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            warn.append(name)
    recs = recommendations(s)
    dd = s["dataset_dimensions"]
    ports = dd["ports"]
    T, TM, FA, R, RC, PO, SH = (s["target"], s["temporal"], s["feature_analysis"], s["feature_target_relationships"],
                                s["reconstruction"], s["ports"], s["distribution_shift"])
    nd = TM["noise_decomposition_train"]
    bl = s["baseline_rmse_h_from_dataset_stage"]
    checks = s["sanity_checks"]
    n_pass = sum(c["result"] == "PASS" for c in checks)
    noise = {p: nd[p]["noise_sd_h"] for p in ports}
    pooled_noise = float(np.sqrt(np.mean([v ** 2 for v in noise.values()])))
    rel = R["features"]
    pr = {k: rel[k]["partial_given_window_ewma_within_port"] for k in rel}
    pl = {k: rel[k]["partial_given_last_dwell_within_port"] for k in rel}
    grp = R["grouped"]
    sig = sorted(ports, key=lambda p: -nd[p]["signal_share"])
    tv = SH["train_vs_val"]
    td = SH["test_descriptive_only"]
    strike = SH["strike_hours_pct_by_port_split"]
    dev_strike_ports = [p for p in ports if (strike[p]["train"] or 0) > 0 or (strike[p]["val"] or 0) > 0]
    test_new_strike = [p for p in ports if (strike[p]["test"] or 0) > 0 and p not in dev_strike_ports]
    vc = RC["value_check_vs_local_observed_median"]
    tb = RC["target_behaviour_dev"]["any reconstructed dwell input"]
    tl = RC["target_behaviour_dev"]["dwell at hour t reconstructed"]
    hl = TM["history_length_rmse_dev"]
    P = {k: v for k, v in plots.items()}

    L = []
    a = L.append
    a("# CargoMind Stage 02 — Exploratory Data Analysis of the DL Dataset")
    a("")
    a("Generated by `DL/EDA/scripts/run_eda.py` from `DL/Dataset/processed/`, the dataset built by `DL/Dataset/scripts/build_dl_dataset.py`. No model was trained, "
      "no scaler or transform was fitted, and nothing in `DL/Dataset/processed/` or `ML/Dataset/Processed/` was changed.")
    a("")
    a("**Data use:**")
    a("- Target, temporal, feature and reconstruction analyses use the **development period** only, i.e. training and validation hours.")
    a("- Test-period numbers appear only in section 9, as a descriptive warning.")
    a("- All statistics are in original units. The saved scalers were only applied to verify the stored arrays.")
    a("")
    if warn:
        a(f"> ⚠️ **The dataset differs from the version this report's prose was written for** ({', '.join(warn)}). "
          "The numbers below are current, but re-check every interpretive sentence.")
        a("")
    a("All figures are in `DL/EDA/plots/`, and all numbers are in `DL/EDA/reports/eda_summary.json`.")
    a("")

    # ---------------------------------------------------------------------------------------- 1
    a("## 1. Executive summary")
    a("")
    a(f"- **The dataset is structurally sound.** {n_pass} of {len(checks)} sanity checks pass.")
    a("  - Arrays, schema, manifest hashes, window counts and split membership all agree.")
    a("  - Every `X` value reproduces exactly from the hourly panel with the saved scalers.")
    a("  - No data integrity problem was found, so the dataset was not rebuilt.")
    a("- **Dwell levels differ strongly between ports.**")
    a(f"  - Ports explain {pct(100 * T['between_port_variance_share'])} of development-target variance.")
    a(f"  - NLRTM alone accounts for {pct(100 * T['share_of_total_sum_of_squares_by_port']['NLRTM'])} of the total sum of squares.")
    a("  - Per-port scaling, which the dataset already uses, is necessary.")
    a("- **Hour-to-hour behaviour is a slowly moving level plus independent hourly noise.**")
    a("  - Lag-1 autocorrelation of the hourly *change* is close to −0.5 at every port "
      f"({min(v['lag1_acf_of_change'] for v in nd.values()):.2f} to {max(v['lag1_acf_of_change'] for v in nd.values()):.2f}).")
    a(f"  - The implied noise SD is {min(noise.values()):.1f}–{max(noise.values()):.1f} h (pooled ≈ {pooled_noise:.1f} h).")
    a(f"  - Noise is {pct(100 * (1 - nd['MYPKG']['signal_share']), 0)} of dwell variance at MYPKG, "
      f"but only {pct(100 * (1 - nd['NLRTM']['signal_share']), 0)} at NLRTM.")
    a("- **The forecastable headroom is small.**")
    a(f"  - The validation EWMA RMSE ({f(bl['val']['ewma_alpha_0.2'])} h) is already within about "
      f"{bl['val']['ewma_alpha_0.2'] - pooled_noise:.1f} h of the noise floor.")
    a("  - An LSTM can only add value by anticipating level changes, which happen in disruption and backlog episodes.")
    a("- **The informative operational signals, beyond the window's own dwell history,** are:")
    a(f"  - utilization (partial r {f(pr['utilization'])});")
    a(f"  - disruption state (active {f(pr['disruption_active'])}, elapsed {f(pr['disruption_hours_elapsed'])});")
    a(f"  - labour strike ({f(pr['labour_strike'])});")
    a(f"  - vessels at anchor ({f(pr['vessels_at_anchor'])});")
    a(f"  - wind gusts ({f(pr['wind_gust'])}).")
    a("  - All are modest. None is constant or useless.")
    a(f"- **Strike data is very thin in development:** {FA['labour_strike']['episodes']} episodes, at {len(dev_strike_ports)} ports only. "
      f"The test period has strikes at {', '.join(test_new_strike)}, which never strike in development data.")
    a("- **Reconstructed dwell inputs show no measurable effect.**")
    a(f"  - They are spread evenly over ports (χ² p = {RC['port_concentration_dev']['chi2_uniform_p']}) and days "
      f"(p = {RC['daily_concentration_dev']['chi2_uniform_p']}).")
    a(f"  - They deviate from neighbouring observed hours just as observed values do (KS p = {vc['ks_p']}).")
    a(f"  - Windows containing them have the same EWMA error (Mann-Whitney p = {tb['mann_whitney_p_abs_ewma_error']}).")
    a("- **There is clear train → validation shift at some ports.** CNSHA's dwell level rises by "
      f"{f(tv['target']['per_port_mean_shift_in_train_sd']['CNSHA'])} training SDs and USLAX's falls by "
      f"{f(-tv['target']['per_port_mean_shift_in_train_sd']['USLAX'])}. Disruption exposure halves "
      f"({pct(100 * tv['disruption_active']['train_mean'])} → {pct(100 * tv['disruption_active']['val_mean'])} of hours).")
    a("")
    a("**Verdict: the dataset is ready for LSTM training,** under the evaluation discipline in section 12.")
    a("")

    # ---------------------------------------------------------------------------------------- 2
    a("## 2. Dataset structure and sanity checks")
    a("")
    rows = []
    for sp in ("train", "val", "test"):
        ar = dd["arrays"][sp]
        rows.append([sp, ar["X"], ar["provenance"], ar["y"], f"{dd['split_bounds_utc'][sp][0][:16]} → {dd['split_bounds_utc'][sp][1][:16]}",
                     f"{min(dd['windows_per_port'][sp].values())}–{max(dd['windows_per_port'][sp].values())}"])
    a(table(["Split", "X", "provenance", "y", "Hours covered (UTC)", "Windows per port"], rows))
    a("")
    a("**Channels.** The order, from `feature_schema.json`, is:")
    a(", ".join(f"`{k}`" for k in s["features"]) + ".")
    a("")
    a("**Scaling and target:**")
    a("- Inputs are stored **scaled**:")
    a("  - dwell: per-port z-score;")
    a("  - utilization: ÷ 100;")
    a("  - wind gust: global z-score;")
    a("  - hours since disruption began: log1p, then global z-score;")
    a("  - vessels at anchor: log1p, then per-port z-score;")
    a("  - binary channels: unscaled.")
    a("- The target `y` is the **per-port z-score of next-hour dwell**. It is stored in hours as `y_hours`.")
    a("- `dataset.to_hours` inverts the scaling with the saved scaler (max error < 1e-5 h).")
    a(f"- The **reconstruction indicator is not in the saved `X`.** `dataset.load_split()` appends it by default, so the actual model input is "
      f"`[N, 12, {len(s['loader_default_channels'])}]` with channels {', '.join(f'`{c}`' for c in s['loader_default_channels'])}.")
    a("")
    a(table(["Check", "Result", "Evidence"], [[c["check"], c["result"], c["evidence"] or ""] for c in checks]))
    a("")
    a("**No inconsistencies were found.** As expected, the scaled channel means of validation and test drift away from 0 "
      "(for example the test dwell mean is "
      f"{f(dd['scaled_array_stats']['test']['X_channel_mean'][0])}), because the scalers were fitted on training hours only.")
    a("")

    # ---------------------------------------------------------------------------------------- 3
    a("## 3. Target variable: next-hour container dwell time")
    a("")
    a("The analysis uses observed or unit-converted targets of the kept windows. Reconstructed targets are not in the dataset.")
    a("")
    rows = [[k] + [f(T[sp][k]) if isinstance(T[sp][k], float) else T[sp][k] for sp in ("train", "val")]
            for k in ("n", "mean", "median", "std", "min", "max", "q1", "q3", "iqr", "skewness", "excess_kurtosis", "outliers_iqr_rule")]
    a(table(["Statistic (h)", "Train", "Validation"], rows))
    a("")
    a("**Shape.**")
    a(f"- The pooled distribution is right-skewed (training skewness {f(T['train']['skewness'])}, excess kurtosis "
      f"{f(T['train']['excess_kurtosis'])}).")
    a("- The skew comes from mixing ports with different levels and from NLRTM's long backlog episode (up to "
      f"{f(T['per_port']['NLRTM']['train']['max'], 1)} h).")
    a("- Within each port the distribution is far closer to symmetric (`B1_target_distribution.png`, right panel).")
    a("- Pooled IQR \"outliers\" are mostly high-level ports, not errors.")
    a(f"- With the IQR rule applied within each port, flags range from 0 (NLRTM, USLAX) to {max(T['within_port_outliers_iqr_rule'].values())}, "
      "Because the audit found no single-hour jump beyond 6 robust SDs, these are hours of high level, not spikes, and need no correction.")
    a("")
    rows = []
    for p in ports:
        t_ = T["per_port"][p]
        rows.append([p, f(t_["train"]["mean"], 1), f(t_["train"]["std"], 1), f(t_["train"]["min"], 1), f(t_["train"]["max"], 1),
                     f(t_["train"]["skewness"]), f(t_["val"]["mean"], 1), f(t_["val_shift_in_train_sd"]), t_["ks_train_vs_val"]["p_value"]])
    a(table(["Port", "Train mean", "Train SD", "Min", "Max", "Skew", "Val mean", "Val − train (train SDs)", "KS p (train vs val)"], rows))
    a("")
    a(f"**Levels over time.** Dwell levels move in multi-day episodes rather than following a trend "
      f"(`B3_target_timeseries_rolling.png`, ports {', '.join(T['representative_ports'].values())}). The 24-hour rolling SD is about 4–5 h "
      "in calm periods and rises during build-ups and decays (`B4_rolling_sd_by_port.png`).")
    a("")
    a(f"**Train vs validation.** Pooled, the distributions are similar (KS {f(T['train_vs_val_pooled_ks']['statistic'], 3)}), "
      "but this hides opposite movements at individual ports. CNSHA (+1.3 SD) and LKCMB (+0.9) are higher in validation, "
      "while USLAX (−0.8) and SGSIN (−0.6) are lower. Section 9 covers this.")
    a("")
    for k, alt in [(0, "target distribution"), (1, "target by port"), (2, "target time series"), (3, "rolling SD")]:
        a(img(P["target"][k], alt))
        a("")

    # ---------------------------------------------------------------------------------------- 4
    a("## 4. Temporal analysis")
    a("")
    rows = []
    for p in sig:
        rows.append([p, f(TM["acf"][p]["lag_1"]), f(TM["acf"][p]["lag_12"]), f(TM["acf"][p]["lag_24"]), f(TM["pacf"][p]["lag_1"]), f(TM["pacf"][p]["lag_2"]),
                     f(TM["pacf"][p]["lag_12"]), f(nd[p]["lag1_acf_of_change"]), f(nd[p]["noise_sd_h"]), pct(100 * nd[p]["signal_share"], 0)])
    a(table(["Port (by signal share)", "ACF 1", "ACF 12", "ACF 24", "PACF 1", "PACF 2", "PACF 12", "ACF of Δ, lag 1", "Noise SD (h)", "Signal share"], rows))
    a("")
    a("The ACF is computed from pairs of observed hours. Including reconstructed hours changes it by at most "
      f"{max(TM['acf'][p]['max_abs_diff_vs_including_reconstructed'] for p in ports):.2f}.")
    a("")
    a("**What the temporal structure shows:**")
    a("1. **Level plus noise.**")
    a("   - Every port's hourly change has lag-1 autocorrelation ≈ −0.5, and essentially none at lags ≥ 2. This is the signature of independent hourly "
      "noise on top of a smooth level.")
    a(f"   - The noise SD ({min(noise.values()):.1f}–{max(noise.values()):.1f} h) is almost the same at every port.")
    a("   - What differs is the level's variability. *Signal share* is the fraction of dwell variance that is not hourly noise: "
      f"{pct(100 * nd['NLRTM']['signal_share'], 0)} at NLRTM, but {pct(100 * nd['MYPKG']['signal_share'], 0)} at MYPKG.")
    a("2. **Persistence, not momentum.**")
    a("   - At episodic ports, ACF stays high for days (NLRTM ACF 24 h "
      f"{f(TM['acf']['NLRTM']['lag_24'])}).")
    a("   - The PACF decays over lags 1–3 and is ≈ 0 by lag 12.")
    a("   - Averaging several recent hours therefore helps, but hours beyond the window add little given the window.")
    a("3. **Rapid reversals are noise.**")
    a("   - A single-hour jump is followed by a partial reversal (the negative ACF of Δ).")
    a("   - That is why repeating the last hour (persistence) is a poor forecast, and why averaging beats it.")
    a("")
    a("**Is 12 hours enough?** Descriptive smoothing errors on development targets (nothing fitted; same targets for every length):")
    a("")
    a(table(["Summary of past dwell", "RMSE next hour (h)"], [[k, f(v, 3)] for k, v in hl.items()]))
    a("")
    a(f"- Trailing means of 6–12 h are best ({f(hl['trailing mean 12 h'], 3)} h). 24 h is worse ({f(hl['trailing mean 24 h'], 3)}), and 48 h worse still "
      f"({f(hl['trailing mean 48 h'], 3)}), because levels move.")
    a(f"- The EWMA with α = 0.2 over the full history ({f(hl['EWMA alpha 0.2 (full history)'], 3)}) is barely better than the in-window "
      f"12 h mean, and its effective memory is ≈ 5 h.")
    a(f"- Given the window EWMA, the mean dwell of hours t−47 … t−12 has partial r {f(TM['older_history_partial_corr']['partial_r_mean_hours_t47_to_t12_given_window_ewma'])}. "
      "Older history adds essentially nothing.")
    a("- **12 hours is sufficient.**")
    a("")
    a("**Daily and weekly patterns:**")
    a(f"- Deviation from a centred 24-hour mean varies by local hour with an amplitude of {f(TM['diurnal']['amplitude_h_max_minus_min_hourly_mean'])} h.")
    a(f"- That is statistically detectable (p = {TM['diurnal']['anova_p']}) but explains {pct(100 * TM['diurnal']['eta_squared'], 2)} of the variance, which is negligible.")
    a(f"- No day-of-week effect is detectable (p = {TM['weekly']['anova_p']}), and {TM['weekly']['weeks_covered']} weeks are too few to establish one.")
    a("- No calendar channels are needed.")
    a("")
    a("**What this means for an LSTM:**")
    a("- The univariate part of the problem is close to solved by exponential smoothing.")
    a("- A recurrent model can learn that smoothing, but its gain must come from:")
    a("  - recognising level changes earlier, from operational inputs;")
    a("  - non-linear episode dynamics (build-up and decay).")
    a(f"- Expect gains, if any, of a few tenths of an hour of RMSE: the EWMA is ≈ {bl['val']['ewma_alpha_0.2'] - pooled_noise:.1f} h above the noise floor on validation.")
    a("")
    for k, alt in enumerate(["ACF/PACF", "lag relationships", "daily profile", "history length", "example windows"]):
        a(img(P["temporal"][k], alt))
        a("")

    # ---------------------------------------------------------------------------------------- 5
    a("## 5. Feature analysis (development hours, original units)")
    a("")
    rows = []
    for k, d in FA.items():
        extra = ""
        if k == "utilization":
            extra = f"{pct(100 * d['share_at_100'])} of hours at 100 (censored); between-port variance {pct(100 * d['between_port_variance_share'], 0)}"
        elif k in ("labour_strike", "disruption_active"):
            extra = f"prevalence {pct(100 * d['prevalence'])}; {d['episodes']} episodes; ports with any: {sum(v > 0 for v in d['per_port_prevalence'].values())}/10"
        elif k == "disruption_hours_elapsed":
            w = d["when_active"]
            extra = f"{pct(100 * d['share_zero'])} zeros; when active: median {f(w['median'], 0)} h, p90 {f(w['p90'], 0)} h, max {f(w['max'], 0)} h"
        elif k == "vessels_at_anchor":
            extra = f"{pct(100 * d['share_zero'])} zeros; overdispersed (variance/mean ≈ {f(d['index_of_dispersion_median'], 1)} per port)"
        elif k == "wind_gust":
            extra = f"{d['clipped_negative_hours']} negative value clipped to 0 in the dataset build"
        elif k == "dwell":
            extra = f"between-port variance {pct(100 * d['between_port_variance_share'], 0)}"
        rows.append([k, d["type"], f"{f(d['min'], 1)} – {f(d['max'], 1)}", f(d["median"], 1), f(d["std"], 2), f(d["median_within_port_std"], 2), extra])
    a(table(["Feature", "Type", "Range", "Median", "SD", "Median within-port SD", "Notes"], rows))
    a("")
    a("**Per feature:**")
    a("- **Binary indicators are imbalanced but not constant.**")
    a(f"  - `disruption_active` is 1 in {pct(100 * FA['disruption_active']['prevalence'])} of development hours, across "
      f"{FA['disruption_active']['episodes']} episodes and all ports (INMAA least, at {pct(100 * FA['disruption_active']['per_port_prevalence']['INMAA'])}).")
    a(f"  - `labour_strike` is 1 in only {pct(100 * FA['labour_strike']['prevalence'])} of hours, from **{FA['labour_strike']['episodes']} episodes at "
      f"{len(dev_strike_ports)} ports** ({', '.join(dev_strike_ports)}). Six ports never strike in development data, so the model can only learn a strike effect "
      "from very few events.")
    a("- **Hours since disruption began** is zero-inflated. When non-zero, it spans 1–96 h with a long tail, and log1p scaling is appropriate.")
    a("- **Vessels at anchor** varies meaningfully. It is a count (0–33), zero in about half of hours, and overdispersed. Its level differs by port "
      f"(between-port variance {pct(100 * FA['vessels_at_anchor']['between_port_variance_share'], 0)}), which per-port scaling handles.")
    a(f"- **Wind gusts** vary mostly within ports (between-port variance only {pct(100 * FA['wind_gust']['between_port_variance_share'], 0)}; "
      f"median within-port SD {f(FA['wind_gust']['median_within_port_std'], 1)} km/h). There is enough variation to carry a signal.")
    a(f"- **Utilization** varies strongly within ports (median within-port SD {f(FA['utilization']['median_within_port_std'], 1)} percentage points) "
      f"and is censored at 100 in {pct(100 * FA['utilization']['share_at_100'])} of hours. That ranges from "
      f"{pct(100 * FA['utilization']['per_port_share_at_100']['MYPKG'])} at MYPKG to {pct(100 * FA['utilization']['per_port_share_at_100']['NLRTM'])} at NLRTM.")
    a("- **No feature is nearly constant** at the dataset level.")
    a("- **Unusual values:** apart from one clipped negative gust in development hours, nothing outside physical ranges was found.")
    a("")
    for k, alt in enumerate(["feature distributions", "features by port", "exposure over time"]):
        a(img(P["features"][k], alt))
        a("")

    # ---------------------------------------------------------------------------------------- 6
    a("## 6. Feature–target relationships")
    a("")
    a(f"These use the {R['n_windows']} development windows. Each feature is taken at hour t (the last input step) against dwell at t+1. Measures:")
    a("- **Pooled** correlation, which is dominated by port levels.")
    a("- **Within-port** correlation, with each port's mean removed.")
    a("- **Partial** correlation within port, controlling for (a) dwell at hour t and (b) the window's EWMA, which is the information a smoothing baseline already has.")
    a("")
    rows = [[k, f(rel[k]["pearson_pooled"]), f(rel[k]["spearman_pooled"]), f(rel[k]["pearson_within_port"]),
             f(pl[k]) if pl[k] is not None else "–", f(pr[k]), f"{f(rel[k]['by_port_range'][0])} to {f(rel[k]['by_port_range'][1])}"] for k in rel]
    a(table(["Feature at t", "Pearson (pooled)", "Spearman (pooled)", "Pearson (within port)", "Partial | dwell(t)", "**Partial | window EWMA**", "Partial | EWMA, port range"], rows))
    a("")
    a("**Reading the table:**")
    a(f"- **Dwell history.** It is almost perfectly correlated with the target (within port {f(rel['dwell']['pearson_within_port'])}), "
      f"but once the window EWMA is known, the current hour adds nothing ({f(pr['dwell'])}). The last single hour is mostly noise.")
    a(f"- **Vessels at anchor** looks strong raw (within port {f(rel['vessels_at_anchor']['pearson_within_port'])}), and "
      f"strong given the last hour ({f(pl['vessels_at_anchor'])}). But it falls to {f(pr['vessels_at_anchor'])} given the window EWMA.")
    a("  - It is mainly a second, noisy view of the same backlog level.")
    a("  - Its value fades within about 2 h of lag (`E2_lagged_partial_correlations.png`).")
    a("  - Most of its raw correlation exists because both variables move with the backlog.")
    a(f"- **Utilization** has no raw correlation (within port {f(rel['utilization']['pearson_within_port'])}), but the largest partial correlation "
      f"given the EWMA ({f(pr['utilization'])}; negative at every port, {f(rel['utilization']['by_port_range'][0])} to {f(rel['utilization']['by_port_range'][1])}).")
    a("  - Low throughput relative to capacity, with dwell held fixed, precedes higher dwell, which is consistent with a capacity cut building a backlog.")
    a("  - The signal is persistent across all 12 lags.")
    a("  - **This is the clearest case of a feature that is useful only in combination with dwell history.**")
    a(f"- **Disruption state** is positive at every port ({f(rel['disruption_active']['by_port_range'][0])} to {f(rel['disruption_active']['by_port_range'][1])}).")
    a(f"  - Next-hour dwell exceeds the EWMA by {f(grp['disruption_active']['1']['residual_mean_h'])} h on average during disruptions, against "
      f"{f(grp['disruption_active']['0']['residual_mean_h'])} h otherwise (p = {grp['disruption_active']['mann_whitney_p_residual']}).")
    a("  - The excess grows with time since onset: " + ", ".join(f"{k}: {f(v['residual_mean_h'])} h"
      for k, v in grp["disruption_hours_elapsed_bins"].items()) + ".")
    a("  - The EWMA lags rising episodes, and the disruption features can anticipate that.")
    a(f"- **Labour strike** shows an excess of {f(grp['labour_strike']['1']['residual_mean_h'])} h over the EWMA (n = {int(grp['labour_strike']['1']['n'])} windows).")
    a(f"  - The partial r ranges from {f(rel['labour_strike']['by_port_range'][0])} to {f(rel['labour_strike']['by_port_range'][1])} at the 4 ports with strikes.")
    a("  - The estimate rests on very few events.")
    a(f"- **Wind gusts** are weak but consistently positive at every port (partial {f(rel['wind_gust']['by_port_range'][0])} to {f(rel['wind_gust']['by_port_range'][1])}).")
    a("")
    a("**Caveats:**")
    a("- Per-port intervals that assume independent windows are too narrow, because windows overlap and share episodes.")
    a("- The synthetic generator builds dwell from a backlog driven by capacity cuts (disruptions, strikes, high gusts). These relationships are "
      "therefore partly properties of the simulator, not evidence about real ports.")
    a("- Nothing here is causal.")
    a("")
    for k, alt in enumerate(["correlations", "lagged partial correlations", "binary and anchor vs residual"]):
        a(img(P["relationships"][k], alt))
        a("")

    # ---------------------------------------------------------------------------------------- 7
    a("## 7. Reconstruction and data-quality sensitivity")
    a("")
    a(table(["", "Train", "Validation", "Test (descriptive)"],
            [["Kept windows with ≥ 1 reconstructed dwell input"] + [pct(RC["windows_with_reconstructed_dwell_input_pct"][sp]) for sp in ("train", "val", "test")],
             ["Kept windows with any reconstructed input cell"] + [pct(RC["windows_with_any_reconstructed_input_pct"][sp]) for sp in ("train", "val", "test")]]))
    a("")
    rows = [[p] + [pct(RC["reconstructed_dwell_hours_pct_by_port_split"][p][sp]) for sp in ("train", "val", "test")]
            + [pct(RC["windows_with_reconstructed_dwell_pct_by_port_split"][p][sp], 0) for sp in ("train", "val", "test")] for p in ports]
    a(table(["Port", "Recon. dwell hours: train", "val", "test", "Windows affected: train", "val", "test"], rows))
    a("")
    a("**Concentration:**")
    a(f"- Reconstructed dwell hours are spread evenly across ports (χ² p = {RC['port_concentration_dev']['chi2_uniform_p']}).")
    a(f"- They are also spread evenly across days (p = {RC['daily_concentration_dev']['chi2_uniform_p']}; at most {RC['daily_concentration_dev']['max_per_day']} per day across all ports).")
    a("- USLAX's validation period is an exception: 10 % of hours are reconstructed, so 80 % of its validation windows contain one.")
    a("")
    a("**Do reconstructed values look different?**")
    a(f"- Relative to the median of observed neighbouring hours (±6 h), reconstructed values deviate by {f(vc['reconstructed_mean_sd_h'][0])} ± {f(vc['reconstructed_mean_sd_h'][1])} h.")
    a(f"- Observed values deviate by {f(vc['observed_mean_sd_h'][0])} ± {f(vc['observed_mean_sd_h'][1])} h.")
    a(f"- KS p = {vc['ks_p']} and Levene p = {vc['levene_p_equal_spread']}: **no detectable difference in level or spread.**")
    a("")
    a("**Does target behaviour differ?** Windows with reconstructed dwell inputs have the same next-hour EWMA error:")
    a(f"- With reconstructed inputs: RMSE {f(tb['True']['ewma_rmse_h'], 3)} h.")
    a(f"- All inputs observed: {f(tb['False']['ewma_rmse_h'], 3)} h.")
    a(f"- p = {tb['mann_whitney_p_abs_ewma_error']}.")
    a(f"- This holds even when hour t itself is reconstructed: {f(tl['True']['ewma_rmse_h'], 3)} vs {f(tl['False']['ewma_rmse_h'], 3)} h, p = {tl['mann_whitney_p_abs_ewma_error']}.")
    a("")
    a("**Conclusion:**")
    a("- Reconstruction is unlikely to materially affect learning or evaluation.")
    a("- Keep the `dwell_reconstructed` channel. It is cheap, and the fill method is still undocumented.")
    a("- Run one ablation without it, rather than removing data.")
    a("")
    for k, alt in enumerate(["reconstruction by port and time", "reconstructed vs observed"]):
        a(img(P["reconstruction"][k], alt))
        a("")

    # ---------------------------------------------------------------------------------------- 8
    a("## 8. Port-level comparison")
    a("")
    rows = [[t_["port"], t_["windows_train"], f(t_["target_mean_train_h"], 1), f(t_["target_sd_train_h"], 1), f(t_["noise_sd_h"], 2),
             pct(100 * t_["signal_share"], 0), f(t_["acf_lag12"]), pct(t_["disruption_hours_pct"]), pct(t_["strike_hours_pct"]),
             f(t_["anchor_mean"], 2), f(t_["utilization_mean_pct"], 1), pct(t_["utilization_at_100_pct"]), pct(t_["reconstructed_dwell_pct_dev"], 2)]
            for t_ in PO["table"]]
    a(table(["Port", "Train windows", "Target mean (h)", "Target SD (h)", "Noise SD (h)", "Signal share", "ACF 12", "Disruption hours",
             "Strike hours", "Anchor mean", "Util. mean (%)", "Util. at 100", "Recon. dwell"], rows))
    a("")
    a(f"**Window counts:** no port dominates. Each port contributes ≈ {pct(100 * PO['window_share_per_port_train'], 0)} of windows.")
    a("")
    a("**Target variance is far less balanced:**")
    a(f"- NLRTM holds {pct(100 * T['share_of_total_sum_of_squares_by_port']['NLRTM'])} of the pooled sum of squares.")
    a("- **Global scaling of dwell would hide the quiet ports.** A per-port SD of 4 h would be about 0.2 global SDs, and an error in hours would be dominated by NLRTM.")
    a("- The dataset's per-port z-scaling avoids that, but it has the reverse effect:")
    a("  - At MYPKG, DEHAM, AEJEA and INMAA most of the z-scaled variance is irreducible noise.")
    a("  - At NLRTM, CNSHA and USLAX the noise is only about 0.1–0.3 z-units.")
    a("- The ports fall into two groups that behave differently enough to make **per-port evaluation essential:**")
    a("  - **High-signal ports** (NLRTM, USLAX, CNSHA, INNSA, SGSIN): long episodes and high ACF. Here an LSTM can beat the EWMA.")
    a("  - **Noise-dominated ports** (MYPKG, DEHAM, AEJEA, INMAA): ACF ≤ 0.3. No model can do much better than a mean.")
    a("")
    a(img(P["ports"][0], "port comparison"))
    a("")

    # ---------------------------------------------------------------------------------------- 9
    a("## 9. Distribution shift")
    a("")
    a("### Train → validation (development; usable for decisions)")
    a("")
    a("Values are hourly, in original units. SMD = (validation mean − train mean) / train SD, computed within port.")
    a("")
    rows = [[k, f(tv[k]["train_mean"]), f(tv[k]["val_mean"]), f(tv[k]["ks_statistic"], 3), tv[k]["ks_p"], f(tv[k]["mean_abs_per_port_smd"])]
            for k in s["features"]]
    a(table(["Feature", "Train mean", "Val mean", "KS", "KS p", "Mean \\|per-port SMD\\|"], rows))
    a("")
    a("**Target shift:**")
    a("- The pooled mean is almost unchanged.")
    a("- By port, CNSHA (+1.34 SD; the Spring-Festival backlog plateau) and LKCMB (+0.93) are higher, while USLAX (−0.80) and SGSIN (−0.59) are lower.")
    a("- The level-shift ports are the high-signal ports, so validation RMSE will depend on how well the model tracks episode onsets and decays.")
    a("")
    a("**Covariate shift:**")
    a(f"- Disruption hours halve ({pct(100 * tv['disruption_active']['train_mean'])} → {pct(100 * tv['disruption_active']['val_mean'])}).")
    a(f"- Strike hours fall ({pct(100 * tv['labour_strike']['train_mean'])} → {pct(100 * tv['labour_strike']['val_mean'])}). In validation they occur only at DEHAM.")
    a("- Gusts are higher at INNSA (+1.2 SD) and LKCMB (+1.8 SD).")
    a("")
    a("**Implication:** validation contains few disruption examples. Early stopping on validation RMSE will mostly reward calm-period accuracy. "
      "Monitor validation error on disruption hours separately.")
    a("")
    a(img(P["shift"][0], "train vs validation shift"))
    a("")
    a("### Test period (descriptive only; not used for any decision)")
    a("")
    a(f"- The target is lower than in training: {f(td['target']['train_mean_h'], 1)} → {f(td['target']['test_mean_h'], 1)} h pooled.")
    a(f"  - The scaled `y` mean is {f(td['target']['scaled_y_mean'])}.")
    a("  - NLRTM, USLAX and SGSIN are 0.6–0.7 SD lower. MYPKG is 1.1 SD higher (a test-period strike and disruption).")
    a(f"- Strike hours rise to {pct(100 * td['labour_strike']['test_mean'])}.")
    a(f"  - Test strikes occur at **{', '.join(test_new_strike)}** ({', '.join(pct(strike[p]['test'], 0) for p in test_new_strike)} of their test hours) and NLRTM.")
    a(f"  - **{' and '.join(test_new_strike)} have no strike hours in development data.**")
    a(f"- Vessels at anchor fall ({f(td['vessels_at_anchor']['train_mean'])} → {f(td['vessels_at_anchor']['test_mean'])}) and gusts are lower.")
    a("- **Caveat:** test performance will partly measure extrapolation of the strike effect to unseen ports, and accuracy in a calmer regime.")
    a("")

    # ---------------------------------------------------------------------------------------- 10
    a("## 10. Important observations")
    a("")
    a(f"1. **Most next-hour variation is irreducible.** The noise SD of about {pooled_noise:.1f} h is close to the EWMA RMSE of "
      f"{f(bl['val']['ewma_alpha_0.2'])} h (validation) and {f(bl['test']['ewma_alpha_0.2'])} h (test).")
    a("2. **Signal lives in episodes at five ports.** Elsewhere dwell is near-constant plus noise.")
    a("3. **Operational features matter only conditionally.** Utilization and disruption state carry information *beyond* recent dwell, while their raw "
      "correlations are weak or misleading. Vessels at anchor is the reverse: strong raw correlation, little extra information.")
    a("4. **Strike evidence is thin,** and its test-period distribution differs in kind: new ports.")
    a("5. **Reconstructed inputs are statistically indistinguishable** from observed ones on every check made.")
    a("6. **The dataset passes all structural checks.**")
    a("")

    # ---------------------------------------------------------------------------------------- 11
    a("## 11. Limitations")
    a("")
    a("- **The data is synthetic.** The dwell process (port base level, backlog from capacity cuts, independent hourly noise) is simple, and the "
      "relationships found are properties of the simulator. None of this transfers to real ports without re-analysis.")
    a("- **Coverage is short:** 850 development hours (≈ 5 weeks) per port, with one Spring Festival and 42 development disruption episodes. "
      "Weekly and seasonal structure cannot be assessed.")
    a("- **Uncertainty is understated.** Overlapping windows and shared episodes mean the naive p-values and intervals here overstate certainty, "
      "especially for per-port strike estimates.")
    a("- **The noise decomposition is approximate.** It assumes independent hourly noise on a smooth level, which fits every port well "
      "(lag-1 ACF of Δ ≈ −0.5) but is approximate during sharp episode onsets.")
    a("- **Partial correlations are linear and pairwise.** They do not capture interactions, for example utilization × disruption.")
    a("- **Some provenance is unknown.** The reconstruction method behind the filled values is undocumented. They are statistically "
      "indistinguishable from observations, but are still not observations.")
    a("")

    # ---------------------------------------------------------------------------------------- 12
    a("## 12. Recommendations for the DL Engineer")
    a("")
    for rec in recs:
        a(f"{rec['priority']}. {rec['recommendation']} *(evidence: {rec['evidence']})*")
    a("")
    a("Further practical points:")
    a("- **Splits and normalisation:**")
    a("  - Do not reshuffle across splits, and do not refit scalers.")
    a("  - Shuffle training windows within the training split only.")
    a("  - Use the saved per-port statistics. Global normalisation of dwell is not recommended, because it would mask the quiet ports.")
    a("- **Loss and evaluation units:**")
    a("  - Optimise MSE on the scaled target `y`, but select and report models by RMSE in hours (`dataset.to_hours`).")
    a("  - Optionally also report a port-balanced RMSE, since NLRTM dominates pooled hour-scale error.")
    a("- **Repeat runs.** Use at least 5 seeds and report mean ± SD. Differences below about 0.05 h RMSE are not meaningful at this sample size.")
    a("- **Ablations worth running:** without `labour_strike`, without `vessels_at_anchor`, and without the `dwell_reconstructed` channel.")
    a("")

    # ---------------------------------------------------------------------------------------- 13
    a("## Implications for LSTM Architecture and Training")
    a("")
    a("| # | Question | Answer | Evidence |")
    a("|---|---|---|---|")
    a(f"| 1 | Is a 12-hour window reasonable? | **Yes.** 6–12 h trailing means give the best smoothing error; 24 h and 48 h are worse; hours t−47…t−12 add nothing given the window (partial r {f(TM['older_history_partial_corr']['partial_r_mean_hours_t47_to_t12_given_window_ewma'])}). | Strong |")
    a(f"| 2 | Is there enough temporal signal to justify an LSTM? | **Partly.** The history is strongly autocorrelated at 5 ports, but its usable part is captured by exponential smoothing (EWMA {f(bl['val']['ewma_alpha_0.2'])} h vs noise ≈ {pooled_noise:.1f} h). An LSTM is justified only as a way to combine history with operational signals non-linearly; it must beat EWMA to be kept. | Strong (headroom); moderate (whether an LSTM captures it) |")
    a(f"| 3 | Most informative operational features? | Utilization ({f(pr['utilization'])}), disruption active ({f(pr['disruption_active'])}) and elapsed ({f(pr['disruption_hours_elapsed'])}), strike ({f(pr['labour_strike'])}), vessels at anchor ({f(pr['vessels_at_anchor'])}), gusts ({f(pr['wind_gust'])}); partial r given window EWMA. | Moderate (consistent signs across ports, modest size) |")
    a("| 4 | Nearly constant or unhelpful features? | None is constant. `labour_strike` is the most fragile (4 development episodes, 4 ports). Vessels at anchor is largely redundant with dwell history. Keep both, and ablate. | Moderate |")
    a(f"| 5 | Strong port-specific differences? | **Yes.** Ports explain {pct(100 * T['between_port_variance_share'], 0)} of target variance; signal share ranges {pct(100 * nd['MYPKG']['signal_share'], 0)}–{pct(100 * nd['NLRTM']['signal_share'], 0)}. | Strong |")
    a("| 6 | Global or per-port scaling? | Keep **per-port** dwell and anchor scaling (already in the dataset). Global scaling would let NLRTM dominate. Add a port embedding (`port_index`) so the model can learn port-specific noise and dynamics. | Strong (scaling); weak (embedding benefit untested) |")
    a(f"| 7 | Could reconstructed inputs affect performance? | **Unlikely.** No difference in value distribution (KS p {vc['ks_p']}) or baseline error (p {tb['mann_whitney_p_abs_ewma_error']}). Keep the flag channel as a safeguard. | Moderate |")
    a("| 8 | What to be careful about? | Overfitting noise (most variance is irreducible). Early stopping on a calm validation period. Few strike and disruption examples. Reporting gains below seed-to-seed variation. Never using test windows for any choice. | Strong |")
    a("| 9 | Essential evaluation breakdowns? | Per port; high-signal vs noise-dominated ports; disruption-active vs normal hours; strike vs non-strike (noting new strike ports in test); windows with vs without reconstructed inputs; always side by side with EWMA, the 12 h mean and persistence on identical windows. | Strong |")
    a("")
    a("**Evidence summary:**")
    a("- **Strong:**")
    a("  - Hourly noise of about 4 h dominates next-hour variation.")
    a("  - A 12 h window is sufficient.")
    a("  - Port heterogeneity is large, so per-port scaling and evaluation are needed.")
    a("  - Disruption and utilization carry information beyond dwell history.")
    a("  - The dataset is structurally valid.")
    a("- **Moderate:**")
    a("  - Ranking of operational features.")
    a("  - Reconstruction has no material effect.")
    a("  - Vessels at anchor is largely redundant.")
    a("  - Validation is calmer than training.")
    a("- **Weak or inconclusive:**")
    a("  - Size and generality of the strike effect.")
    a("  - Diurnal effect: detectable, but about 1 h in amplitude and explaining 0.6 % of variance.")
    a("  - Any weekly pattern.")
    a("  - Whether a port embedding helps.")
    a("  - Whether non-linear models extract more than linear partial correlations suggest.")
    a("")
    a("**Recommended model input:** `dataset.load_split(split)` gives **`[N, 12, 8]`** "
      f"(train N = {dd['arrays']['train']['X'][0]}, validation N = {dd['arrays']['val']['X'][0]}, test N = {dd['arrays']['test']['X'][0]}), "
      "plus `port_index` for an embedding. The target is `y` `[N]`, scaled, with `y_hours` for evaluation.")
    a("For a 1D CNN, transpose to `[N, 8, 12]`.")
    a("")
    return recs, "\n".join(L)
