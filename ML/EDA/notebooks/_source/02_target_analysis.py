# %% [markdown]
# # 02 · Target analysis: `delay_status` and `actual_delay_hours`
#
# **Questions:** How are the three classes distributed? Is the label consistent with the delay hours? What does the
# delay distribution look like (early arrivals, tails)? How does it vary by port, carrier, mode, cargo, priority and route?
# Does the problem need imbalance handling?
#
# Labels are **not** changed here. Thresholds: On-Time ≤ 12 h < At-Risk ≤ 48 h < Critical.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as ss

from cargomind_eda import io, stats, plotting as P
from cargomind_eda.config import CLASSES, CLASS_COLORS, THRESHOLDS

P.setup()
s = io.load("shipment_records")
flags = pd.read_csv(ROOT / "data" / "interim" / "shipment_lineage_flags.csv")
s = s.merge(flags[["shipment_id", "outcome_not_observed_in_raw"]], on="shipment_id", how="left", validate="1:1")
s["observed"] = ~s["outcome_not_observed_in_raw"]
print(s.shape, "| outcome not observed in raw:", s.outcome_not_observed_in_raw.sum())

# %% [markdown]
# ## B1. Label integrity

# %%
recomputed = pd.cut(s.actual_delay_hours, [-np.inf, THRESHOLDS["on_time_max"], THRESHOLDS["at_risk_max"], np.inf], labels=CLASSES, right=True)
delay_from_ts = (s.actual_arrival_ts - s.expected_arrival_ts).dt.total_seconds() / 3600
print("label disagrees with thresholds:", (recomputed.astype(str) != s.delay_status.astype(str)).sum())
print("max |delay_hours − (actual − expected)|:", round((delay_from_ts - s.actual_delay_hours).abs().max(), 4), "h")
print("delay_hours resolution:", np.unique(np.round(s.actual_delay_hours * 10 % 1, 6))[:5], "(multiples of 0.1 h)")

# %% [markdown]
# The label is a deterministic function of `actual_delay_hours`, which is itself exactly `actual_arrival_ts − expected_arrival_ts`.
# So all three outcome columns encode the same information; any one of them used as a feature would reveal the target.

# %% [markdown]
# ## B2. Class distribution and imbalance

# %%
cnt = s.delay_status.value_counts().reindex(CLASSES)
dist = pd.DataFrame({"n": cnt, "pct": (cnt / cnt.sum() * 100).round(2)})
dist["ci95"] = [tuple(round(v * 100, 1) for v in stats.wilson_ci(k, len(s))) for k in cnt]
obs = s[s.observed].delay_status.value_counts(normalize=True).reindex(CLASSES).mul(100).round(2)
dist["pct_observed_only"] = obs
dist["pct_sea_only"] = s[s.shipping_mode == "sea"].delay_status.value_counts(normalize=True).reindex(CLASSES).mul(100).round(2)
print("imbalance ratio (majority/minority):", round(cnt.max() / cnt.min(), 2))
dist

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
ax[0].bar(CLASSES, cnt.values, color=P.PALETTE, width=0.6)
for i, (n, p) in enumerate(zip(cnt.values, dist.pct.values)):
    ax[0].text(i, n + 80, f"{n:,}\n({p:.1f}%)", ha="center", fontsize=9)
ax[0].set_ylabel("Shipments")
ax[0].set_ylim(0, cnt.max() * 1.2)
ax[0].set_title("Class distribution (all 9,850 shipments)")
wk = s.assign(week=s.departure_ts.dt.tz_localize(None).dt.to_period("W").dt.start_time)
share = pd.crosstab(wk.week, wk.delay_status, normalize="index") * 100
share[CLASSES].plot.area(ax=ax[1], color=P.PALETTE, alpha=0.9, linewidth=0)
ax[1].set_ylabel("Share of departures (%)")
ax[1].set_xlabel("Departure week (UTC)")
ax[1].set_ylim(0, 100)
ax[1].set_title("Class mix by departure week")
ax[1].legend(title="delay_status", loc="lower left", fontsize=8)
P.save(fig, "02_class_distribution")
print(pd.crosstab(wk.week.dt.date, wk.delay_status, margins=True))

# %% [markdown]
# ## B3. Distribution of `actual_delay_hours`

# %%
d = s.actual_delay_hours
summ = d.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(2)
print(summ.to_string())
print("skewness:", round(ss.skew(d), 2), "| excess kurtosis:", round(ss.kurtosis(d), 2))
print("early arrivals (< 0 h):", (d < 0).sum(), f"({(d < 0).mean()*100:.1f}%)", "| exactly on time (0 h):", (d == 0).sum())

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
bins = np.arange(-30, 205, 2.5)
for c in CLASSES:
    ax[0].hist(d[s.delay_status == c], bins=bins, color=CLASS_COLORS[c], label=c, alpha=0.95)
for t in THRESHOLDS.values():
    ax[0].axvline(t, color="#52514e", ls="--", lw=1)
ax[0].axvline(0, color="#52514e", lw=0.8)
ax[0].set_xlabel("actual_delay_hours (h; negative = early)")
ax[0].set_ylabel("Shipments")
ax[0].set_title("Delay distribution, coloured by class (thresholds 12 h, 48 h)")
ax[0].legend()
for mode, col in [("sea", "#2a78d6"), ("air", "#52514e")]:
    x = np.sort(d[s.shipping_mode == mode])
    ax[1].plot(x, 1 - np.arange(1, len(x) + 1) / len(x), color=col, label=f"{mode} (n={len(x):,})")
ax[1].set_yscale("log")
ax[1].set_xlabel("actual_delay_hours (h)")
ax[1].set_ylabel("P(delay > x)  (log scale)")
ax[1].set_title("Survival curve of delay: air never exceeds 3.4 h")
for t in THRESHOLDS.values():
    ax[1].axvline(t, color="#52514e", ls="--", lw=1)
ax[1].legend()
P.save(fig, "02_delay_distribution")

# %% [markdown]
# ### Early arrivals and extreme delays

# %%
early = s[d < 0]
print("Early arrivals by mode (% of mode):")
print((s.assign(early=d < 0).groupby("shipping_mode").early.mean() * 100).round(1).to_string())
print("\nEarly-arrival magnitude (h):", early.actual_delay_hours.describe().round(1).to_dict())

q1, q3 = d.quantile([0.25, 0.75])
fence = q3 + 1.5 * (q3 - q1)
far = q3 + 3 * (q3 - q1)
print(f"\nTukey fences: 1.5·IQR = {fence:.1f} h → {(d > fence).sum()} shipments; 3·IQR = {far:.1f} h → {(d > far).sum()} shipments")
print("top 10 delays:\n", s.nlargest(10, "actual_delay_hours")[["shipment_id", "origin_port_id", "destination_port_id", "carrier", "departure_ts", "actual_delay_hours"]].to_string(index=False))

# %%
# Are the extremes clustered in time/place (suggesting operational events) or scattered (suggesting errors)?
ext = s[d > far]
print("extreme delays by origin:", ext.origin_port_id.value_counts().to_dict())
print("extreme delays by departure date:", ext.departure_ts.dt.date.value_counts().head(8).to_dict())

# %% [markdown]
# Extreme delays (> 3·IQR above Q3) are concentrated at a few origin ports and a few departure dates, which is the pattern
# expected from operational disruptions (verified against disruption events in notebook 08), not from random recording
# errors. They are timestamp-consistent and **valid extreme observations**; they define the Critical class and must not be
# removed or winsorised.

# %% [markdown]
# ### Threshold proximity (label sensitivity)
# How many shipments sit close to a class boundary, where a small timing error would flip the label?

# %%
for t in THRESHOLDS.values():
    for w in (1, 2, 4):
        k = ((d - t).abs() <= w).sum()
        print(f"within ±{w} h of {t:.0f} h: {k:4d} ({k/len(d)*100:.1f}%)")

# %% [markdown]
# ### Filled outcomes vs observed outcomes

# %%
comp = s.groupby("observed").actual_delay_hours.describe().round(2)
comp.index = comp.index.map({True: "observed in raw", False: "not observed in raw (filled)"})
display(comp)
ct = pd.crosstab(s.observed, s.delay_status)
print(ct, "\nchi2 p =", round(ss.chi2_contingency(ct)[1], 3), "| KS p =",
      round(ss.ks_2samp(s.loc[s.observed, "actual_delay_hours"], s.loc[~s.observed, "actual_delay_hours"]).pvalue, 3))

# %% [markdown]
# The 295 filled outcomes have a distribution indistinguishable from the observed ones. That does not make them observations:
# their provenance is unknown. Recommendation: exclude them from training and evaluation, and use them only for sensitivity checks.

# %% [markdown]
# ## B4. Delay by shipment and network dimensions
#
# For each grouping: class mix, median delay, % delayed (> 12 h) with a 95 % Wilson CI, and Cramér's V vs `delay_status`.
# Because **air shipments are always On-Time** (B3), grouping effects are also reported on the sea-only subset to avoid
# reading the air/sea mix as a carrier or port effect.

# %%
sea = s[s.shipping_mode == "sea"]
dims = ["origin_port_id", "destination_port_id", "carrier", "shipping_mode", "cargo_type", "shipment_priority"]
assoc = pd.concat(
    [stats.screen_categorical(s, dims)[["levels", "cramers_v", "p_value"]].add_suffix("_all"),
     stats.screen_categorical(sea, [c for c in dims if c != "shipping_mode"])[["cramers_v", "p_value"]].add_suffix("_sea")],
    axis=1,
).sort_values("cramers_v_all", ascending=False)
P.save_table(assoc, "02_group_association")
assoc

# %%
fig, axes = plt.subplots(3, 2, figsize=(13, 12))
for ax, c in zip(axes.flat, dims):
    order = s.groupby(c, observed=True).actual_delay_hours.median().sort_values().index
    sns.boxplot(data=s, y=c, x="actual_delay_hours", order=order, ax=ax, color="#b7d3f6", fliersize=1.5, linewidth=1)
    for t in THRESHOLDS.values():
        ax.axvline(t, color="#52514e", ls="--", lw=1)
    ax.set_xlabel("actual_delay_hours (h)")
    ax.set_title(f"Delay by {c}")
P.save(fig, "02_delay_boxplots_by_group")

# %%
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
for ax, c in zip(axes.flat, dims):
    P.stacked_class_bar(sea if c != "shipping_mode" else s, c, ax=ax, title=f"Class mix by {c}" + (" (sea only)" if c != "shipping_mode" else ""))
P.save(fig, "02_class_mix_by_group")

# %%
for c in dims:
    print(f"\n── {c} ({'sea only' if c != 'shipping_mode' else 'all'})")
    display(stats.class_rates(sea if c != "shipping_mode" else s, c).sort_values("pct_delayed(>12h)", ascending=False))

# %% [markdown]
# ### Route-level delay
# Routes have 20–476 sea shipments each (median 72). Ranking routes by raw delay rate over-states extremes for small routes, so routes are
# shown with their 95 % CI.

# %%
rr = stats.class_rates(sea, "route_id")
rr = rr.join(s.groupby("route_id")[["origin_port_id", "destination_port_id"]].first())
rr["median_delay_h"] = sea.groupby("route_id").actual_delay_hours.median()
print("shipments per route:", rr.n.describe().round(0).to_dict())
P.save_table(rr, "02_route_delay_rates")
top = rr.sort_values("pct_delayed(>12h)", ascending=False)
display(top.head(10))
display(top.tail(10))

# %%
fig, ax = plt.subplots(figsize=(12, 4.2))
r = rr.sort_values("pct_delayed(>12h)").reset_index()
ax.errorbar(range(len(r)), r["pct_delayed(>12h)"], yerr=[r["pct_delayed(>12h)"] - r.ci95_low, r.ci95_high - r["pct_delayed(>12h)"]],
            fmt="o", ms=4, color="#2a78d6", ecolor="#9ec5f4", elinewidth=1.5)
overall = (sea.delay_status != "On-Time").mean() * 100
ax.axhline(overall, color="#52514e", ls="--", lw=1)
ax.text(1, overall + 2, f"sea average {overall:.1f}%", fontsize=8, color="#52514e")
for i in list(range(3)) + list(range(len(r) - 3, len(r))):
    ax.annotate(f"{r.origin_port_id[i]}→{r.destination_port_id[i]}", (i, r['pct_delayed(>12h)'][i]), xytext=(3, 6), textcoords="offset points", fontsize=7)
ax.set_xticks([])
ax.set_xlabel("90 routes, sorted by delay rate")
ax.set_ylabel("% delayed > 12 h (95% CI)")
ax.set_title("Route delay rates (sea shipments): wide spread, many CIs exclude the average")
P.save(fig, "02_route_delay_rates")
print("routes whose CI lies entirely above the sea average:", (rr.ci95_low > overall).sum(), "| entirely below:", (rr.ci95_high < overall).sum())

# %% [markdown]
# ## B5. Imbalance assessment
#
# * Classes are **moderately imbalanced** — about 55 / 30 / 15 % (majority:minority ≈ 3.7 : 1). This is not extreme; there are
#   ~1,480 Critical examples, enough for learning without synthetic oversampling.
# * The imbalance is **structured**: the 503 air shipments are all On-Time (max delay 3.4 h), so on the sea subset the mix is
#   ~52.6 / 31.6 / 15.8 %. Air shipments add no information about the At-Risk/Critical boundary.
# * The class mix drifts by departure week (B2), so any split must be time-based and class weights should be estimated on the
#   training window only.
# * The problem is **ordinal** (On-Time < At-Risk < Critical, derived from one continuous variable). Errors are not
#   symmetric: calling a Critical shipment On-Time is costlier than calling it At-Risk.
#
# Recommended handling for modelling (not applied here): class-weighted loss or balanced sampling within the training window;
# evaluation with macro-F1, per-class recall (especially Critical), and an ordinal-aware metric (e.g., quadratic weighted kappa
# or a cost matrix); consider modelling `actual_delay_hours` (regression / quantile) and thresholding as a complementary approach,
# because it uses the full ordinal information.
