# %% [markdown]
# # 03 · Shipment-level characteristics vs delay risk
#
# **Question:** which attributes known at booking/departure (priority, carrier, mode, cargo, TEU, ports, route, departure
# time, planned transit duration) distinguish On-Time, At-Risk and Critical shipments — and are those differences
# genuine or explained by other attributes (confounding)?
#
# Air shipments are always On-Time (notebook 02), so effects are measured on **sea shipments** unless stated.

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
net = io.load("port_network")
meta = io.load("port_metadata")
s = s.merge(net[["route_id", "route_type", "transshipment_port_id", "route_distance_km", "planned_transit_hours"]], on="route_id", validate="m:1")
s["planned_duration_h"] = (s.expected_arrival_ts - s.departure_ts).dt.total_seconds() / 3600
tz = meta.set_index("port_id").local_timezone
s["dep_hour_local"] = [ts.tz_convert(tz[p]).hour for ts, p in zip(s.departure_ts, s.origin_port_id)]
s["dep_dow"] = s.departure_ts.dt.day_name().str[:3]
s["delayed"] = (s.delay_status != "On-Time").astype(int)
sea = s[s.shipping_mode == "sea"].copy()
print(s.shape, sea.shape)

# %% [markdown]
# ## C1. Shipping mode — a deterministic split

# %%
print(s.groupby("shipping_mode").actual_delay_hours.describe().round(2))
air = s[s.shipping_mode == "air"]
print("\nair planned duration / route planned hours:", (air.planned_duration_h / air.planned_transit_hours).describe().round(3).to_dict())
print("air shipments by route_type:", air.route_type.value_counts().to_dict())

# %% [markdown]
# All 503 air shipments are On-Time (max delay 3.4 h), and their planned duration is ~7–80 % of the sea route's planned
# hours. `shipping_mode` is therefore a perfect On-Time predictor for air. A model needs no learning for these rows; they
# can be handled by a rule or kept in training with the caveat that they inflate apparent accuracy. All further analysis
# uses sea shipments.

# %% [markdown]
# ## C2. Categorical attributes: association strength
# Chi-square of independence with bias-corrected Cramér's V. All expected cell counts ≥ 5 unless noted.

# %%
sea["dep_hour_bucket"] = pd.cut(sea.dep_hour_local, [-1, 5, 11, 17, 23], labels=["00–05", "06–11", "12–17", "18–23"])
cats = ["carrier", "cargo_type", "shipment_priority", "origin_port_id", "destination_port_id", "route_id", "route_type", "dep_dow", "dep_hour_bucket"]
cat_tab = stats.screen_categorical(sea, cats).sort_values("cramers_v", ascending=False)
cat_tab["magnitude"] = cat_tab.cramers_v.map(lambda v: stats.magnitude_label(v, "v"))
P.save_table(cat_tab, "03_categorical_association_sea")
cat_tab

# %% [markdown]
# `route_id` has the largest V but 90 levels, so part of its V is capacity to fit noise; it decomposes into origin,
# destination and route type (notebook 06). Departure hour shows **no association** (V = 0.01, p = 0.20). Day-of-week is
# statistically detectable but negligible (V = 0.05); with only ~6 weeks of data, each weekday has ~6 dates, so this most
# likely reflects which dates had disruptions rather than a weekly cycle.

# %% [markdown]
# ## C3. Numeric attributes: TEU, planned duration, distance, departure hour

# %%
nums = ["container_teu", "planned_duration_h", "route_distance_km", "planned_transit_hours", "dep_hour_local"]
num_tab = stats.screen_numeric(sea, nums).sort_values("epsilon_sq", ascending=False)
P.save_table(num_tab, "03_numeric_association_sea")
num_tab.round(4)

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sns.violinplot(data=sea, x="delay_status", y="container_teu", hue="delay_status", legend=False, order=CLASSES, palette=CLASS_COLORS, ax=axes[0], cut=0, inner="quartile", linewidth=1)
axes[0].set_title("Container size by class")
axes[0].set_ylabel("container_teu (TEU)")
sns.boxplot(data=sea, x="delay_status", y="planned_duration_h", hue="delay_status", legend=False, order=CLASSES, palette=CLASS_COLORS, ax=axes[1], fliersize=1, linewidth=1)
axes[1].set_title("Planned transit duration by class")
axes[1].set_ylabel("expected − departure (h)")
b = pd.qcut(sea.planned_duration_h, 8)
rate = sea.groupby(b, observed=True).agg(p=("delayed", "mean"), n=("delayed", "size"), x=("planned_duration_h", "median"))
ci = np.array([stats.wilson_ci(p * n, n) for p, n in zip(rate.p, rate.n)])
axes[2].errorbar(rate.x, rate.p * 100, yerr=[(rate.p - ci[:, 0]) * 100, (ci[:, 1] - rate.p) * 100], fmt="o-", color="#2a78d6", capsize=3)
axes[2].set_xlabel("planned transit duration (h, octile medians)")
axes[2].set_ylabel("% delayed > 12 h (95% CI)")
axes[2].set_title("Delay rate vs planned duration")
for a in axes[:2]:
    a.set_xlabel("")
P.save(fig, "03_teu_duration")

# %% [markdown]
# * **Container TEU**: medians are identical across classes and ε² ≈ 0 — no evidence that shipment size relates to delay risk.
# * **Planned duration / distance**: a weak, non-monotonic association (ε² small). Longer voyages are not simply riskier;
#   the shape is driven by which origins/route types sit in each duration band (see C5 and notebook 06).
# * **Departure hour (local)**: no association.

# %% [markdown]
# ## C4. Priority, cargo type and carrier — ordered effects

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, c in zip(axes, ["shipment_priority", "cargo_type", "carrier"]):
    r = stats.class_rates(sea, c).sort_values("pct_delayed(>12h)")
    ax.barh(r.index.astype(str), r["pct_Critical"], color=CLASS_COLORS["Critical"], label="Critical")
    ax.barh(r.index.astype(str), r["pct_At-Risk"], left=r["pct_Critical"], color=CLASS_COLORS["At-Risk"], label="At-Risk")
    ax.errorbar(r["pct_delayed(>12h)"], range(len(r)), xerr=[r["pct_delayed(>12h)"] - r.ci95_low, r.ci95_high - r["pct_delayed(>12h)"]], fmt="none", ecolor="#0b0b0b", capsize=3, lw=1)
    ax.set_title(c)
    ax.set_xlabel("% of sea shipments (bar = At-Risk + Critical; whisker = 95% CI)")
axes[0].legend(loc="lower right")
P.save(fig, "03_priority_cargo_carrier")

# %% [markdown]
# ## C5. Confounding check: is the carrier effect just a route-mix effect?
#
# Carriers serve different route mixes. If high-delay carriers simply sail more often from high-delay origins, the raw carrier
# gap would overstate a carrier effect. **Indirect standardisation**: for each carrier, compute the expected delay rate given
# its own route mix (using all-carrier route delay rates) and compare with the observed rate. The same is done for priority
# and cargo type.

# %%
def standardised(df, group, stratum="route_id", outcome="delayed"):
    base = df.groupby(stratum)[outcome].mean()
    exp = df[stratum].map(base)
    out = df.assign(_exp=exp).groupby(group, observed=True).agg(n=(outcome, "size"), observed=(outcome, "mean"), expected_from_route_mix=("_exp", "mean"))
    out["obs_minus_exp_pp"] = (out.observed - out.expected_from_route_mix) * 100
    out["obs/exp"] = out.observed / out.expected_from_route_mix
    return (out.assign(observed=out.observed * 100, expected_from_route_mix=out.expected_from_route_mix * 100)).round(2)

for g in ["carrier", "shipment_priority", "cargo_type"]:
    print(f"\n── {g}")
    display(standardised(sea, g))

# %%
# Is the carrier/priority/cargo assignment itself route-dependent?
print("Cramér's V between carrier and route_id :", round(stats.cramers_v(pd.crosstab(sea.carrier, sea.route_id)), 3))
print("Cramér's V between carrier and origin   :", round(stats.cramers_v(pd.crosstab(sea.carrier, sea.origin_port_id)), 3))
print("Cramér's V between priority and carrier :", round(stats.cramers_v(pd.crosstab(sea.shipment_priority, sea.carrier)), 3))
print("Cramér's V between cargo and carrier    :", round(stats.cramers_v(pd.crosstab(sea.cargo_type, sea.carrier)), 3))

# %% [markdown]
# **Result:** carrier, cargo type and priority are statistically independent of route and origin (Cramér's V ≈ 0), so the
# route mix explains none of their gaps — observed and route-mix-expected rates differ by up to 15 pp (Coral Bay +15 pp,
# Oceanic −10 pp; high priority −10 pp; hazardous +11 pp). These attributes carry information that is *additional* to route.

# %% [markdown]
# ## C6. Effects within route type (stratified)

# %%
fig, ax = plt.subplots(figsize=(10, 4))
tab = sea.groupby(["carrier", "route_type"]).delayed.mean().unstack() * 100
tab = tab.loc[tab.mean(1).sort_values().index]
tab.plot.barh(ax=ax, color=["#2a78d6", "#eb6834"], width=0.75)
ax.set_xlabel("% delayed > 12 h (sea)")
ax.set_title("Carrier ranking is preserved within both direct and transshipment routes")
ax.legend(title="route_type")
P.save(fig, "03_carrier_by_route_type")
tab.round(1)

# %% [markdown]
# ## C7. Departure timing

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
h = sea.groupby("dep_hour_local").delayed.agg(["mean", "size"])
axes[0].plot(h.index, h["mean"] * 100, marker="o", color="#2a78d6")
axes[0].axhline(sea.delayed.mean() * 100, color="#52514e", ls="--", lw=1)
axes[0].set_xlabel("Departure hour (origin local time)")
axes[0].set_ylabel("% delayed > 12 h")
axes[0].set_ylim(0, 100)
axes[0].set_title("No time-of-day pattern in delay risk")
daily = sea.groupby(sea.departure_ts.dt.date).agg(rate=("delayed", "mean"), crit=("delay_status", lambda x: (x == "Critical").mean()), n=("delayed", "size"))
axes[1].plot(daily.index, daily.rate * 100, color="#2a78d6", label="% delayed > 12 h")
axes[1].plot(daily.index, daily.crit * 100, color="#e34948", label="% Critical")
axes[1].set_ylabel("% of the day's sea departures")
axes[1].set_xlabel("Departure date (UTC)")
axes[1].set_title("Delay risk varies strongly by departure date")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=30)
P.save(fig, "03_departure_timing")
print(f"daily share delayed: min {daily.rate.min()*100:.1f}%, max {daily.rate.max()*100:.1f}%, sd {daily.rate.std()*100:.1f} pp (days with ≥50 departures: {(daily.n >= 50).sum()})")

# %% [markdown]
# Clock time does not matter, but **calendar date** matters a lot: the daily delay rate swings widely. Something time-varying
# and shared across shipments drives risk — port conditions, weather and disruptions (notebooks 04–08). This is also the
# main argument for a time-based train/test split.

# %% [markdown]
# ## C8. Summary — shipment-level attributes
#
# | Attribute | Evidence (sea) | Assessment |
# |---|---|---|
# | shipping_mode | air = 100 % On-Time | deterministic; handle by rule or flag |
# | origin_port_id | V ≈ 0.16; delayed share 31 % (MYPKG) → 68 % (NLRTM) | strongest static attribute; likely proxies port conditions |
# | carrier | V ≈ 0.12; 37 % → 63 %; independent of route (V≈0), ranking holds within direct and transshipment | genuine carrier-level difference *in this data* |
# | shipment_priority | V ≈ 0.08; high 37 % vs low 54 % | monotone, modest |
# | cargo_type | V ≈ 0.07; hazardous / reefer above general | modest |
# | destination_port_id | V ≈ 0.08 | weaker than origin |
# | route_type | V ≈ 0.18; transshipment riskier for every carrier | strong, structural |
| planned duration / distance | ε² ≈ 0.01; Critical median 305 h vs 225 h | weak; largely via route type |
# | container_teu | ε² ≈ 0 | no evidence of association |
# | departure hour / weekday | hour V = 0.01 (n.s.); weekday V = 0.05 | no useful association |
# | departure date | daily delay rate swings strongly | points to time-varying operational drivers |
#
# Associations are observational. A carrier effect that survives route-mix adjustment may still reflect unmeasured
# differences (vessel quality, schedules, booking behaviour), so it should not be read as causal.
