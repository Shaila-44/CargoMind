# %% [markdown]
# # 04 · Port operations and delay risk
#
# **Questions:** How do port operating conditions behave (per port, over time)? How do they relate to each other
# (equipment → throughput, labour → congestion)? And which conditions **at the origin / destination, observed before
# departure**, are associated with higher delay risk?
#
# Shipment-linked port features are taken from `features.build_shipment_features()`: the port's telemetry at the last hour
# ≤ `departure_ts − 1 h`, plus trailing-24 h aggregates. No telemetry after the cutoff is used (see notebook 08/09).

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

from cargomind_eda import io, stats, features, plotting as P
from cargomind_eda.config import CLASSES, CLASS_COLORS

P.setup()
T = io.load_all()
tel, dis, meta = T["port_hourly_telemetry"], T["disruption_events"], T["port_metadata"]
METRICS = ["throughput_teu", "port_utilization_pct", "container_dwell_hours", "vessel_time_in_port_hours",
           "vessel_arrivals", "vessel_departures", "vessels_at_anchor", "customs_hold_hours", "equipment_availability_pct"]

# %% [markdown]
# ## D1. Port operating profiles

# %%
prof = tel.groupby("port_id")[METRICS].median().join(meta.set_index("port_id")["capacity_teu_per_hour"])
prof["pct_hours_util_100"] = tel.groupby("port_id").port_utilization_pct.apply(lambda x: (x == 100).mean() * 100)
prof["pct_hours_labour_not_normal"] = tel.groupby("port_id").labour_status.apply(lambda x: (x != "normal").mean() * 100)
prof["p95_vessels_at_anchor"] = tel.groupby("port_id").vessels_at_anchor.quantile(0.95)
P.save_table(prof, "04_port_profiles")
prof.round(1).sort_values("container_dwell_hours")

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
order = prof.sort_values("container_dwell_hours").index
for ax, c in zip(axes.flat, ["port_utilization_pct", "container_dwell_hours", "vessel_time_in_port_hours", "vessels_at_anchor", "customs_hold_hours", "equipment_availability_pct"]):
    sns.boxplot(data=tel, x="port_id", y=c, order=order, ax=ax, color="#b7d3f6", fliersize=1, linewidth=1)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(c)
fig.suptitle("Hourly port conditions by port (Jan 1 – Feb 11 2025)", y=1.01, fontweight="semibold")
P.save(fig, "04_port_condition_distributions")

# %% [markdown]
# Ports differ structurally (NLRTM, LKCMB and AEJEA have long dwell; NLRTM sits at 100 % utilization 41 % of the time;
# MYPKG and INMAA rarely saturate). Because of these level differences, **pooled** correlations mix between-port and
# within-port variation; relationships below are therefore also computed **within port** (each metric centred on its port mean).

# %% [markdown]
# ## D2. How port metrics relate to each other

# %%
within = tel.copy()
within[METRICS] = within.groupby("port_id")[METRICS].transform(lambda x: x - x.mean())
pooled_r = tel[METRICS].corr("spearman")
within_r = within[METRICS].corr("spearman")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
short = ["throughput", "utilization", "dwell", "vessel_time", "arrivals", "departures", "at_anchor", "customs", "equipment"]
for ax, r, t in [(axes[0], pooled_r, "Pooled across ports"), (axes[1], within_r, "Within port (port-mean centred)")]:
    sns.heatmap(r, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7},
                xticklabels=short, yticklabels=short, cbar_kws={"label": "Spearman ρ"})
    ax.set_title(f"Port metric correlations — {t}")
P.save(fig, "04_metric_correlations")

# %% [markdown]
# ### Equipment availability vs throughput, labour vs congestion

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
eq_r = tel.groupby("port_id").apply(lambda g: ss.spearmanr(g.equipment_availability_pct, g.port_utilization_pct)[0])
hb = axes[0].hexbin(within.equipment_availability_pct, within.port_utilization_pct, gridsize=40, cmap="Blues", mincnt=1, bins="log")
axes[0].set_xlabel("equipment availability (pp vs port mean)")
axes[0].set_ylabel("utilization (pp vs port mean)")
axes[0].set_title(f"Equipment vs utilization (within port)\nmedian per-port ρ = {eq_r.median():.2f}")
fig.colorbar(hb, ax=axes[0], label="port-hours (log)")
lab_order = ["normal", "reduced", "strike"]
for ax, c in zip(axes[1:], ["vessels_at_anchor", "container_dwell_hours"]):
    sns.boxplot(data=within, x="labour_status", y=c, order=lab_order, ax=ax, color="#b7d3f6", fliersize=1, linewidth=1)
    ax.set_ylabel(f"{c} (vs port mean)")
    ax.set_title(f"Labour status vs {c}")
P.save(fig, "04_equipment_labour")
print("per-port Spearman(equipment, utilization):\n", eq_r.round(2).to_string())

# %%
lab = []
for c in ["vessels_at_anchor", "container_dwell_hours", "port_utilization_pct", "vessel_time_in_port_hours", "customs_hold_hours", "equipment_availability_pct"]:
    g = [within.loc[within.labour_status == l, c] for l in lab_order]
    h, p = ss.kruskal(*g)
    lab.append({"metric (within-port)": c, **{f"mean_{l}": x.mean() for l, x in zip(lab_order, g)}, "kruskal_p": p, "epsilon_sq": stats.epsilon_squared(h, len(within))})
pd.DataFrame(lab).set_index("metric (within-port)").round(3)

# %% [markdown]
# ## D3. Port conditions over time, with disruption events
# Anchor queue (the clearest congestion signal) for every port, with disruption intervals shaded.

# %%
fig, axes = plt.subplots(5, 2, figsize=(15, 13), sharex=True)
ev_col = {"weather": "#2a78d6", "strike": "#e34948", "congestion": "#eda100", "equipment_failure": "#6250d6", "cyber": "#1baf7a"}
for ax, p in zip(axes.flat, prof.sort_values("p95_vessels_at_anchor", ascending=False).index):
    g = tel[tel.port_id == p].set_index("ts_hour")
    ax.plot(g.index, g.vessels_at_anchor.rolling(6, min_periods=1).mean(), color="#0b0b0b", lw=1)
    for _, e in dis[dis.port_id == p].iterrows():
        ax.axvspan(e.event_start, e.event_end, color=ev_col[e.event_type], alpha=0.25, lw=0)
    ax.set_title(p, loc="left", fontsize=10)
    ax.set_ylabel("vessels at anchor\n(6 h mean)", fontsize=8)
handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in ev_col.values()]
fig.legend(handles, ev_col.keys(), loc="upper center", ncol=5, title="disruption event type", bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Anchor queues build during and after disruption events", y=1.05, fontweight="semibold")
for ax in axes[-1]:
    ax.tick_params(axis="x", rotation=30)
P.save(fig, "04_anchor_timeseries_with_disruptions")

# %%
# Quantify: port conditions inside vs outside disruption intervals (within port)
inside = np.zeros(len(tel), bool)
for _, e in dis.iterrows():
    inside |= ((tel.port_id == e.port_id) & (tel.ts_hour >= e.event_start) & (tel.ts_hour < e.event_end)).values
within["in_disruption"] = inside
cmp = within.groupby("in_disruption")[["vessels_at_anchor", "container_dwell_hours", "port_utilization_pct", "equipment_availability_pct", "customs_hold_hours"]].mean().T
cmp.columns = ["outside events", "inside events"]
cmp["cliffs_delta_inside_vs_outside"] = [stats.cliffs_delta(within.loc[inside, c], within.loc[~inside, c]) for c in cmp.index]
print(f"port-hours inside a disruption: {inside.sum()} ({inside.mean()*100:.1f}%)")
cmp.round(3)

# %% [markdown]
# **Reading:** during events, ports lose throughput (utilization falls — utilization here is *realised* throughput / nominal
# capacity, not demand), equipment availability drops and vessels queue at anchor; dwell rises with a lag after the event.
# This matters for interpretation below: **low utilization can indicate a disrupted port**, not a quiet one.

# %% [markdown]
# ## D4. Origin and destination conditions before departure vs delay class

# %%
f = features.build_shipment_features(lag_hours=1, tables=T)
sea = f[f.shipping_mode == "sea"].copy()
port_feats = [
    "o_port_utilization_pct", "o_util_mean24", "o_util_max24", "o_util_change24", "o_vessels_at_anchor", "o_anchor_mean24", "o_anchor_max24",
    "o_container_dwell_hours", "o_dwell_mean24", "o_vessel_time_in_port_hours", "o_customs_hold_hours", "o_customs_mean24",
    "o_equipment_availability_pct", "o_equip_min24", "o_labour_disrupted_share24", "o_arrivals_sum24", "o_throughput_teu",
    "d_port_utilization_pct", "d_util_mean24", "d_vessels_at_anchor", "d_anchor_mean24", "d_container_dwell_hours", "d_customs_hold_hours",
    "d_equipment_availability_pct", "d_labour_disrupted_share24",
    "h_util_mean24", "h_anchor_mean24", "h_container_dwell_hours", "h_labour_disrupted_share24",
]
pt = stats.screen_numeric(sea, port_feats)
pt["magnitude_Crit_vs_OnTime"] = pt["cliff_Critical_vs_On-Time"].map(stats.magnitude_label)
pt = pt.sort_values("epsilon_sq", ascending=False)
P.save_table(pt, "04_port_features_vs_class")
pt[["n", "epsilon_sq", "median_On-Time", "median_At-Risk", "median_Critical", "cliff_At-Risk_vs_On-Time", "cliff_Critical_vs_On-Time", "cliff_Critical_vs_At-Risk", "spearman_rho_delay_h", "magnitude_Crit_vs_OnTime"]].round(3)

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, c, lab in zip(axes.flat,
                      ["o_anchor_mean24", "o_container_dwell_hours", "o_util_mean24", "o_customs_mean24", "o_equip_min24", "d_anchor_mean24"],
                      ["origin vessels at anchor, 24 h mean", "origin container dwell (h)", "origin utilization, 24 h mean (%)",
                       "origin customs hold, 24 h mean (h)", "origin min equipment availability, 24 h (%)", "destination vessels at anchor, 24 h mean"]):
    sns.boxplot(data=sea, x="delay_status", y=c, hue="delay_status", legend=False, order=CLASSES, palette=CLASS_COLORS, ax=ax, fliersize=1, linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel(lab)
    ax.set_title(f"{c}  (ε²={pt.loc[c, 'epsilon_sq']:.3f})")
fig.suptitle("Port conditions observed before departure, by delay class (sea shipments)", y=1.01, fontweight="semibold")
P.save(fig, "04_port_conditions_by_class")

# %% [markdown]
# ### Dose–response: delay class mix by origin anchorage congestion

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
sea["anchor_band"] = pd.cut(sea.o_anchor_mean24, [-0.01, 0.5, 1, 2, 4, 8, 100], labels=["≤0.5", "0.5–1", "1–2", "2–4", "4–8", ">8"])
P.stacked_class_bar(sea, "anchor_band", ax=axes[0], order=["≤0.5", "0.5–1", "1–2", "2–4", "4–8", ">8"], title="Class mix by origin anchor queue (24 h mean, vessels)")
sea["dwell_band"] = pd.qcut(sea.o_container_dwell_hours, 6)
P.stacked_class_bar(sea, "dwell_band", ax=axes[1], order=sorted(sea.dwell_band.dropna().unique()), title="Class mix by origin container dwell (sextiles, h)")
P.save(fig, "04_dose_response_anchor_dwell")

# %% [markdown]
# ### Is the relationship consistent across ports?
# Per-origin-port Spearman ρ between each condition and delay hours. A relation that holds within most ports is not just a
# between-port difference.

# %%
rows = []
for p, g in sea.groupby("origin_port_id"):
    row = {"origin_port_id": p, "n": len(g)}
    for c in ["o_anchor_mean24", "o_container_dwell_hours", "o_util_mean24", "o_customs_mean24", "o_equip_min24", "o_vessel_time_in_port_hours"]:
        row[c] = ss.spearmanr(g[c], g.actual_delay_hours)[0]
    rows.append(row)
per_port = pd.DataFrame(rows).set_index("origin_port_id")
P.save_table(per_port, "04_per_port_spearman")
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.heatmap(per_port.drop(columns="n"), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6, ax=ax, cbar_kws={"label": "Spearman ρ with delay hours"})
ax.set_title("Within-origin association of pre-departure conditions with delay")
P.save(fig, "04_per_port_spearman")
per_port.round(2)

# %% [markdown]
# ### Customs hold vs shipment delay

# %%
print("Spearman(o_customs_hold_hours, delay):", round(ss.spearmanr(sea.o_customs_hold_hours, sea.actual_delay_hours)[0], 3))
print("Spearman(d_customs_hold_hours, delay):", round(ss.spearmanr(sea.d_customs_hold_hours, sea.actual_delay_hours)[0], 3))
print("Within-port Spearman(customs_hold, anchor):", round(ss.spearmanr(within.customs_hold_hours, within.vessels_at_anchor)[0], 3),
      "| (customs_hold, dwell):", round(ss.spearmanr(within.customs_hold_hours, within.container_dwell_hours)[0], 3))

# %% [markdown]
# ## D5. Findings — port operations
#
# * **Anchorage congestion at the origin is the strongest port-operations signal.** The 24 h mean vessels-at-anchor has
#   median 0.42 (On-Time) → 0.67 (At-Risk) → 1.92 (Critical); ε² = 0.13, Cliff's δ Critical vs On-Time = 0.54 (large),
#   At-Risk vs On-Time = 0.31. The within-origin Spearman ρ with delay hours is positive at all 10 origins (0.09 DEHAM to 0.49 NLRTM).
# * **Container dwell and vessel time in port at origin** come next (dwell median 78 → 88 → 101 h; δ = 0.44). Both rise during
#   and after disruption events (dwell δ inside vs outside events = 0.40).
# * **Utilization is not a simple congestion measure.** It is realised throughput / nominal capacity, capped at 100 %. It
#   *falls* when capacity is lost (within-port mean −24 pp during events, −38 pp during strikes). The current-hour value is
#   therefore *lower* for Critical shipments (66 % vs 75 %) while the 24 h maximum is higher (96 % vs 90 %). The relationship to
#   risk is non-monotonic. Feature engineering should pair utilization with queue length (backlog) rather than read it as
#   "busy = risky".
# * **Customs hold time** has no meaningful association with delay (ρ = 0.01 origin, −0.04 destination) and is unrelated to
#   anchor queue or dwell within port. That is a negative result. Customs hold is longer under *reduced* labour (+6.9 h vs port mean).
#   Note: 500 customs values were filled during cleaning (notebook 01).
# * **Equipment availability** shows essentially **no** within-port correlation with utilization/throughput (per-port ρ between
#   −0.05 and 0.13). It dips only slightly during disruption events (δ = −0.15) and its 24 h minimum carries negligible signal
#   for delay (ε² = 0.005). The expected "equipment → throughput" link is not visible in this data.
# * **Labour status** (reduced / strike) coincides with longer anchor queues (+2.9 vessels during strikes), longer dwell
#   (+12 h) and much lower utilization within port.
# * **Destination conditions at departure time** are weakly informative (ε² ≤ 0.008). The vessel reaches the destination
#   days later, so destination state now is a poor proxy for its state on arrival (which is future information, notebook 09).
# * **Transshipment hub utilization** (h_util_mean24, ε² = 0.02) carries some signal for the 37 % of shipments with a hub.
#
# All associations are observational. Port congestion and delays may share common causes (disruptions, weather), see notebook 08.
