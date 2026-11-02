# %% [markdown]
# # 07 · Disruptions, port traffic and public holidays
#
# **Questions:** What disruptions occur, how severe and how long are they, and how do they affect port operations? Do
# disruptions known at departure, recent traffic deviations, or nearby public holidays distinguish risk classes?
#
# Timing matters for every variable here:
# * a disruption that has **already started** at the prediction cutoff is knowable; its **end time and total duration are
#   not** (they are only known once it ends);
# * daily traffic for day *D* is complete only after *D* ends, so the last usable day is *D − 1*;
# * public holidays are published in advance, so holiday features are safe even for future dates.

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
dis, tel, traffic, hol, meta = T["disruption_events"], T["port_hourly_telemetry"], T["port_traffic_history"], T["public_holidays"], T["port_metadata"]
dis["duration_h"] = (dis.event_end - dis.event_start).dt.total_seconds() / 3600
f = features.build_shipment_features(lag_hours=1, tables=T)
sea = f[f.shipping_mode == "sea"].copy()

# %% [markdown]
# ## G1. Disruption events: type, severity, duration, capacity impact

# %%
display(pd.crosstab(dis.event_type, dis.severity, margins=True))
display(dis.groupby("event_type").agg(n=("event_id", "size"), median_duration_h=("duration_h", "median"), max_duration_h=("duration_h", "max"),
                                     median_capacity_pct=("affected_capacity_pct", "median")).round(1))
print("events whose end time was reconstructed during cleaning (notebook 01): EVT00007, EVT00022, EVT00052")
print("event_end at or after the last telemetry hour (right-censored):", (dis.event_end >= tel.ts_hour.max()).sum())
print("affected_capacity_pct = 85.0 exactly (cap?):", (dis.affected_capacity_pct == 85).sum())

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.stripplot(data=dis, x="event_type", y="duration_h", hue="severity", palette={"low": "#86b6ef", "medium": "#2a78d6", "high": "#104281"}, ax=axes[0], size=6, jitter=0.2)
axes[0].set_ylabel("duration (h)")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=20)
axes[0].set_title("Event duration by type")
sns.boxplot(data=dis, x="severity", y="affected_capacity_pct", ax=axes[1], color="#b7d3f6", linewidth=1)
sns.stripplot(data=dis, x="severity", y="affected_capacity_pct", ax=axes[1], color="#2a78d6", size=4)
axes[1].set_ylabel("affected capacity (%)")
axes[1].set_title("Capacity impact by severity")
ports_order = dis.port_id.value_counts().index
for i, p in enumerate(ports_order):
    for _, e in dis[dis.port_id == p].iterrows():
        axes[2].plot([e.event_start, e.event_end], [i, i], lw=6, solid_capstyle="butt",
                     color={"low": "#86b6ef", "medium": "#2a78d6", "high": "#104281"}[e.severity])
axes[2].set_yticks(range(len(ports_order)), ports_order)
axes[2].set_title("Event timeline (colour = severity)")
axes[2].tick_params(axis="x", rotation=30)
P.save(fig, "07_disruption_overview")

# %% [markdown]
# Severity is consistent with `affected_capacity_pct` (low ≈ 17–23 %, medium ≈ 32–50 %, high ≈ 50–85 %). The value 85.0
# appears repeatedly, which suggests the impact measure is capped at 85 %. That is consistent with clipping, but cannot be
# confirmed from the data. Weather events are short (median ~15 h); strikes and congestion last longer.

# %% [markdown]
# ## G2. Disruptions known at departure vs delay class

# %%
for c in ["dis_o_active_any", "dis_d_active_any", "dis_h_active_any"]:
    print(f"{c}: share of sea shipments = {sea[c].mean()*100:.1f}%")
sea["origin_event"] = np.select(
    [sea.dis_o_active_any == 0, sea.dis_o_active_max_sev == 1, sea.dis_o_active_max_sev == 2, sea.dis_o_active_max_sev == 3],
    ["none", "low", "medium", "high"], default="none")
r = stats.class_rates(sea, "origin_event").reindex(["none", "low", "medium", "high"])
display(r)
print(stats.chi2_vs_target(sea, "origin_event"))

# %%
types = ["weather", "strike", "congestion", "equipment_failure", "cyber"]
sea["origin_event_type"] = "none"
for t in types:
    sea.loc[sea[f"dis_o_active_{t}"] > 0, "origin_event_type"] = t
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
P.stacked_class_bar(sea, "origin_event", ax=axes[0], order=["none", "low", "medium", "high"], title="Origin disruption active at departure (by severity)")
P.stacked_class_bar(sea, "origin_event_type", ax=axes[1], title="…by event type")
sea["dest_event"] = np.where(sea.dis_d_active_any > 0, "active", "none")
P.stacked_class_bar(sea, "dest_event", ax=axes[2], order=["none", "active"], title="Destination disruption active at departure")
P.save(fig, "07_disruption_at_departure_by_class")
display(stats.class_rates(sea, "origin_event_type"))

# %% [markdown]
# ### Disruption duration: what is knowable at departure?
# For shipments departing during an active origin event: hours **already elapsed** (knowable) vs hours **remaining**
# (requires `event_end` — not knowable while the event is ongoing).

# %%
act = sea[sea.dis_o_active_any == 1]
print(f"shipments departing during an active origin event: {len(act)}")
for c in ["dis_o_active_hours_elapsed", "LEAK_dis_o_active_hours_remaining", "dis_o_active_capacity_pct"]:
    print(f"  Spearman({c}, delay) = {ss.spearmanr(act[c], act.actual_delay_hours)[0]:+.3f}")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, c, lab in [(axes[0], "dis_o_active_capacity_pct", "affected capacity of active origin event (%)"),
                   (axes[1], "dis_o_active_hours_elapsed", "hours since origin event started")]:
    ax.scatter(act[c], act.actual_delay_hours, s=6, alpha=0.4, c=act.delay_status.map(CLASS_COLORS))
    ax.set_xlabel(lab)
    ax.set_ylabel("actual_delay_hours (h)")
axes[0].set_title("Delay vs severity of the event in progress")
axes[1].set_title("Delay vs elapsed time of the event in progress")
P.save(fig, "07_active_event_severity_elapsed")

# %% [markdown]
# ### Known at departure vs starting during the voyage (leakage contrast)
# Disruptions that **start after departure** but before the planned arrival are future information. They are shown only to
# measure how much risk comes from events the model cannot see. They must not be used as features.

# %%
sea["future_origin_or_dest_event"] = ((sea.LEAK_dis_o_future_n + sea.LEAK_dis_d_future_n) > 0).map({True: "≥1 future event", False: "none"})
sea["known_origin_event"] = sea.dis_o_active_any.map({1.0: "active at departure", 0.0: "none"})
display(pd.crosstab([sea.known_origin_event, sea.future_origin_or_dest_event], sea.delay_status, normalize="index").mul(100).round(1)
        .join(pd.crosstab([sea.known_origin_event, sea.future_origin_or_dest_event], sea.delay_status).sum(1).rename("n")))

# %% [markdown]
# ## G3. Port traffic history (daily)

# %%
# Consistency with hourly telemetry: daily vessel arrivals vs vessel_calls, mean vessel_time_in_port vs avg_vessel_stay_hours
td = tel.assign(day=tel.ts_hour.dt.tz_localize(None).dt.normalize()).groupby(["port_id", "day"]).agg(
    arrivals=("vessel_arrivals", "sum"), vessel_time=("vessel_time_in_port_hours", "mean"), anchor=("vessels_at_anchor", "mean"),
    util=("port_utilization_pct", "mean"), hours=("ts_hour", "size")).reset_index()
tr = traffic.merge(td, left_on=["port_id", "period_start"], right_on=["port_id", "day"], how="left", validate="1:1")
full = tr[tr.hours == 24]
print("days with full 24 h telemetry:", len(full), "of", len(tr))
pairs = [("vessel_calls", "arrivals"), ("avg_vessel_stay_hours", "vessel_time"), ("traffic_deviation_pct", "anchor"), ("traffic_deviation_pct", "util"), ("import_volume_t", "vessel_calls")]
print("vessel_calls == daily sum of telemetry vessel_arrivals on every full day:", bool((full.vessel_calls == full.arrivals).all()))
print("max |avg_vessel_stay_hours − daily mean vessel_time_in_port_hours|:", round((full.avg_vessel_stay_hours - full.vessel_time).abs().max(), 3), "h (rounding)")
pd.DataFrame([{"traffic": a, "telemetry (daily)": b, "pooled ρ": ss.spearmanr(full[a], full[b])[0],
               "median within-port ρ": full.groupby("port_id").apply(lambda g: ss.spearmanr(g[a], g[b])[0]).median()} for a, b in pairs]).round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
for p, g in traffic.groupby("port_id"):
    axes[0].plot(g.period_start, g.traffic_deviation_pct, lw=1, alpha=0.8, label=p)
axes[0].axhline(0, color="#52514e", lw=0.8)
axes[0].set_ylabel("traffic deviation vs normal (%)")
axes[0].set_title("Daily traffic deviation by port")
axes[0].tick_params(axis="x", rotation=30)
axes[0].legend(ncol=5, fontsize=7, loc="upper left")
g = traffic[traffic.port_id == "CNSHA"]
axes[1].plot(g.period_start, g.vessel_calls, color="#2a78d6", label="vessel_calls")
ax2 = axes[1]
hd = hol[hol.country_code == "CN"].holiday_date
axes[1].axvspan(hd.min(), hd.max() + pd.Timedelta("1D"), color="#eda100", alpha=0.25, label="CN holidays (Spring Festival)")
axes[1].set_ylabel("vessel calls per day")
axes[1].set_title("CNSHA daily vessel calls around Spring Festival")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=30)
P.save(fig, "07_traffic")

# %%
tr_feats = [c for c in sea.columns if c.startswith(("tr_o_", "tr_d_"))]
tt = stats.screen_numeric(sea, tr_feats)
tt = tt.reindex(tt["cliff_Critical_vs_On-Time"].abs().sort_values(ascending=False).index)
P.save_table(tt, "07_traffic_features_vs_class")
tt[["epsilon_sq", "median_On-Time", "median_At-Risk", "median_Critical", "cliff_Critical_vs_On-Time", "cliff_At-Risk_vs_On-Time", "spearman_rho_delay_h"]].round(3)

# %%
# Traffic volumes are port-scale quantities; check within origin port
for c in ["tr_o_vessel_calls", "tr_o_traffic_deviation_pct", "tr_o_avg_vessel_stay_hours", "tr_o_import_volume_t"]:
    a = sea[c] - sea.groupby("origin_port_id")[c].transform("mean")
    print(f"{c:30s} pooled ρ={ss.spearmanr(sea[c], sea.actual_delay_hours)[0]:+.3f} | within-origin ρ={ss.spearmanr(a, sea.actual_delay_hours)[0]:+.3f}")

# %% [markdown]
# ## G4. Public holidays

# %%
hd = features._holiday_dates(hol, meta)
print("holiday-days per port (national + matching subdivision):")
print(hd.groupby("port_id").size().reindex(meta.port_id).fillna(0).astype(int).to_dict())
print("calendar covers dates up to:", hol.holiday_date.max().date(), "| latest expected arrival:", sea.expected_arrival_ts.max().date())
print("sea shipments whose expected arrival falls after the calendar's last date:", (~sea.hol_calendar_covers_exp_arr).sum())
print("holiday_period_days vs number of listed dates per holiday:")
print(hol.groupby(["country_code", "holiday_name"]).agg(listed_dates=("holiday_date", "nunique"), holiday_period_days=("holiday_period_days", "first")).query("listed_dates != holiday_period_days"))

# %%
# Port operations on holidays vs other days (within port, daily means)
loc_day = []
for p, tzname in meta.set_index("port_id").local_timezone.items():
    g = tel[tel.port_id == p].copy()
    g["local_date"] = g.ts_hour.dt.tz_convert(tzname).dt.tz_localize(None).dt.normalize()
    loc_day.append(g)
tl = pd.concat(loc_day).merge(hd.rename(columns={"holiday_date": "local_date"}).assign(holiday=True), on=["port_id", "local_date"], how="left")
tl["holiday"] = tl.holiday.notna()
hol_eff = tl.groupby(["port_id", "holiday"])[["port_utilization_pct", "vessels_at_anchor", "container_dwell_hours"]].mean().unstack()
diff = pd.DataFrame({m: hol_eff[(m, True)] - hol_eff[(m, False)] for m in ["port_utilization_pct", "vessels_at_anchor", "container_dwell_hours"]})
diff["holiday_hours"] = tl[tl.holiday].groupby("port_id").size()
print("Holiday minus non-holiday mean, by port:")
diff.dropna(subset=["holiday_hours"]).round(2)

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
sea["o_hol"] = sea.hol_o_days_to_nearest.clip(upper=7).map(lambda d: f"{int(d)}" if d < 7 else "≥7")
P.stacked_class_bar(sea, "o_hol", ax=axes[0], order=["0", "1", "2", "3", "4", "5", "6", "≥7"], title="Days from departure to nearest origin holiday")
sea["d_hol"] = np.where(sea.hol_d_within_2d_of_exp_arr == 1, "holiday within 2 d", "no")
P.stacked_class_bar(sea, "d_hol", ax=axes[1], order=["no", "holiday within 2 d"], title="Destination holiday near expected arrival")
sea["h_hol"] = sea.hol_h_days_in_voyage.map(lambda v: "direct route" if pd.isna(v) else ("hub holiday in voyage" if v > 0 else "no hub holiday"))
P.stacked_class_bar(sea, "h_hol", ax=axes[2], order=["direct route", "no hub holiday", "hub holiday in voyage"], title="Hub holiday during planned voyage")
P.save(fig, "07_holidays_by_class")
for c in ["o_hol", "d_hol", "h_hol"]:
    print(c, {k: round(v, 3) for k, v in stats.chi2_vs_target(sea, c).items() if k in ("cramers_v", "p_value")})

# %%
# Is the origin-holiday effect driven by one holiday (CN Spring Festival)?
near = sea[sea.hol_o_within_2d == 1]
display(stats.class_rates(near, "origin_port_id").sort_values("n", ascending=False))
not_cn = sea[sea.origin_port_id != "CNSHA"]
print("origin holiday within 2 d, excluding CNSHA origins:", {k: round(v, 3) for k, v in stats.chi2_vs_target(not_cn, "hol_o_within_2d").items() if k in ("cramers_v", "p_value")})
display(stats.class_rates(not_cn, "hol_o_within_2d"))

# %% [markdown]
# ## G5. Findings — disruptions, traffic, holidays
#
# **Disruptions**
# * 55 events: weather 27 (mostly high severity, median 15 h), equipment failure 11, congestion 9 (median 61 h), strike 7
#   (median 56 h), cyber 1. Severity tracks `affected_capacity_pct`. Seven events sit at exactly 85.0 %, consistent with a cap.
# * **An origin disruption active at departure is the single strongest signal in the dataset** (Cramér's V = 0.33 for severity):
#   no event → 41 % delayed / 10 % Critical; medium → 71 % / 24 %; high → 96 % / 73 %. By type: active **weather** event → 98 % delayed,
#   **86 % Critical** (n = 374); strike → 85 % / 40 %; congestion → 77 % / 33 %; equipment failure → 48 % / 11 % (close to baseline).
#   Low-severity events (n = 63) show no elevation.
# * Among the 1,431 shipments departing during an origin event, the event's **affected capacity %** is strongly associated with delay
#   (ρ = +0.60), and **hours already elapsed** negatively (ρ = −0.25): leaving late in an event is less risky. Both are knowable at
#   departure. **Remaining duration** (needs `event_end`) is *not* knowable and, interestingly, is less informative (ρ = −0.12).
# * **Events that start after departure** (future information) matter too. Among shipments with no origin event at departure,
#   those whose origin/destination has an event starting during the voyage are delayed 44 % vs 33 %. A departure-time model cannot
#   see this risk, so it is a floor on achievable error unless forecasts or en-route updates are added.
# * 15 % of sea shipments depart with an active event at origin, 15 % with one at the destination, 15 % at the hub.
#
# **Port traffic history**
# * **`port_traffic_history` is derived from the hourly telemetry.** `vessel_calls` equals the daily sum of telemetry
#   `vessel_arrivals` on all 410 complete days, and `avg_vessel_stay_hours` equals the daily mean `vessel_time_in_port_hours`
#   (≤ 0.05 h rounding). It contains no independent information about these quantities; volumes (`import_volume_t`,
#   `export_volume_t`) scale with vessel calls between ports but not within port.
# * The only traffic feature with signal is `tr_o_avg_vessel_stay_hours` (δ Critical vs On-Time = 0.38), which is a lagged duplicate
#   of telemetry vessel time. `traffic_deviation_pct` is unrelated to delay (ρ = −0.04) and to telemetry congestion.
#   Recommendation: do not add traffic features on top of telemetry features.
#
# **Public holidays**
# * Holidays coincide with lower utilization at every port (−4 to −36 pp). At **CNSHA (Spring Festival)** they coincide with a
#   +4.3-vessel anchor backlog and +19 h dwell, with no logged disruption event.
# * Departing within 2 days of an origin holiday: 60 % vs 44 % delayed (V = 0.14; days-to-holiday bands V = 0.125). This is
#   **not only Spring Festival**: excluding CNSHA origins it remains at 56 % vs 45 % (V = 0.086). A **hub holiday during the planned voyage** is associated with higher risk (V = 0.13);
#   destination holiday near expected arrival is weak (V = 0.04).
# * Holiday features are calendar-based and safe at prediction time. Caveats: the calendar ends 26 Feb while 267 sea shipments
#   are expected to arrive later (their destination-holiday features are unknown, not "no holiday"); `port_closed` is only
#   populated for CN and is always `False`; multi-day periods are listed as separate dated rows (e.g. Pongal = 3 named days), so
#   `holiday_period_days` describes the period, not the row.
