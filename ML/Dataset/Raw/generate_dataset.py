"""
Synthetic, deliberately UNCLEANED supply-chain logistics dataset.

Generates 8 linked tables that follow the schema doc:
  port_metadata, port_network, port_hourly_telemetry, weather_hourly,
  port_traffic_history, disruption_events, public_holidays, shipment_records

Main tables (shipment_records, port_hourly_telemetry, weather_hourly) have
exactly 10,000 rows each. All values are simulated; delays are driven by
congestion, weather, disruptions, holidays and route structure so that a
model has real signal to learn once the data is cleaned.

Run:  python generate_dataset.py  [output_dir]
"""
import sys, os, json
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
OUT = sys.argv[1] if len(sys.argv) > 1 else "supply_chain_dataset_raw"
os.makedirs(OUT, exist_ok=True)

ISSUES = []  # (table, column, issue, rows_affected)


def log(table, column, issue, n):
    ISSUES.append((table, column, issue, int(n)))


def pick(idx, frac, exclude=None):
    """Random subset of an index (array of row positions)."""
    idx = np.asarray(idx)
    if exclude is not None:
        idx = np.setdiff1d(idx, exclude)
    k = int(round(len(idx) * frac)) if frac < 1 else int(frac)
    k = min(k, len(idx))
    return rng.choice(idx, k, replace=False) if k else np.array([], dtype=int)


# ---------------------------------------------------------------------------
# 1. Ports
# ---------------------------------------------------------------------------
PORTS = [
    # id, name, cc, subdiv, region, lat, lon, type, cap_teu_hr, tz, base_temp
    ("INMAA", "Chennai Port", "IN", "IN-TN", "South Asia", 13.0827, 80.2920, "container", 180, "Asia/Kolkata", 26),
    ("INNSA", "Jawaharlal Nehru Port (Nhava Sheva)", "IN", "IN-MH", "South Asia", 18.9490, 72.9512, "container", 650, "Asia/Kolkata", 25),
    ("SGSIN", "Port of Singapore", "SG", "", "Southeast Asia", 1.2644, 103.8400, "container", 4300, "Asia/Singapore", 27),
    ("CNSHA", "Port of Shanghai", "CN", "CN-SH", "East Asia", 31.2304, 121.4900, "container", 5600, "Asia/Shanghai", 5),
    ("AEJEA", "Jebel Ali Port", "AE", "AE-DU", "Middle East", 25.0110, 55.0610, "container", 1700, "Asia/Dubai", 20),
    ("NLRTM", "Port of Rotterdam", "NL", "NL-ZH", "Europe", 51.9500, 4.1400, "mixed", 1700, "Europe/Amsterdam", 4),
    ("USLAX", "Port of Los Angeles", "US", "US-CA", "North America", 33.7400, -118.2600, "container", 1100, "America/Los_Angeles", 14),
    ("DEHAM", "Port of Hamburg", "DE", "DE-HH", "Europe", 53.5400, 9.9800, "mixed", 960, "Europe/Berlin", 2),
    ("LKCMB", "Port of Colombo", "LK", "LK-1", "South Asia", 6.9500, 79.8400, "container", 900, "Asia/Colombo", 27),
    ("MYPKG", "Port Klang", "MY", "MY-10", "Southeast Asia", 3.0000, 101.3900, "container", 1500, "Asia/Kuala_Lumpur", 27),
]
P = len(PORTS)
pid = [p[0] for p in PORTS]
pidx = {p: i for i, p in enumerate(pid)}
cap = np.array([p[8] for p in PORTS], float)
HUBS = ["SGSIN", "AEJEA", "LKCMB", "MYPKG"]

port_meta = pd.DataFrame({
    "port_id": pid,
    "port_name": [p[1] for p in PORTS],
    "country_code": [p[2] for p in PORTS],
    "subdivision_code": [p[3] for p in PORTS],
    "region": [p[4] for p in PORTS],
    "latitude": [p[5] for p in PORTS],
    "longitude": [p[6] for p in PORTS],
    "port_type": [p[7] for p in PORTS],
    "capacity_teu_per_hour": cap,
    "local_timezone": [p[9] for p in PORTS],
    "data_source": "synthetic",
    "is_synthetic": True,
})

# ---------------------------------------------------------------------------
# 2. Time grid
# ---------------------------------------------------------------------------
H = 1000
t0 = pd.Timestamp("2025-01-01 00:00", tz="UTC")
hours = pd.date_range(t0, periods=H, freq="h")

# ---------------------------------------------------------------------------
# 3. Holidays (2025, from public calendars; verify before real use)
# ---------------------------------------------------------------------------
hol_rows = [
    ("IN", "", "2025-01-26", "Republic Day", "public", 1),
    ("IN", "IN-TN", "2025-01-14", "Pongal", "public", 3),
    ("IN", "IN-TN", "2025-01-15", "Thiruvalluvar Day", "public", 3),
    ("IN", "IN-TN", "2025-01-16", "Uzhavar Thirunal", "public", 3),
    ("IN", "IN-MH", "2025-02-19", "Chhatrapati Shivaji Maharaj Jayanti", "public", 1),
    ("IN", "", "2025-02-26", "Maha Shivaratri", "public", 1),
    ("SG", "", "2025-01-01", "New Year's Day", "public", 1),
    ("SG", "", "2025-01-29", "Chinese New Year", "public", 2),
    ("SG", "", "2025-01-30", "Chinese New Year (Day 2)", "public", 2),
    ("CN", "", "2025-01-01", "New Year's Day", "public", 1),
] + [("CN", "", d.strftime("%Y-%m-%d"), "Spring Festival", "public", 8)
     for d in pd.date_range("2025-01-28", "2025-02-04")] + [
    ("AE", "", "2025-01-01", "New Year's Day", "public", 1),
    ("NL", "", "2025-01-01", "Nieuwjaarsdag", "public", 1),
    ("US", "", "2025-01-01", "New Year's Day", "public", 1),
    ("US", "", "2025-01-20", "Martin Luther King Jr. Day", "public", 1),
    ("US", "", "2025-02-17", "Washington's Birthday", "public", 1),
    ("DE", "", "2025-01-01", "Neujahr", "public", 1),
    ("LK", "", "2025-01-13", "Duruthu Full Moon Poya Day", "public", 1),
    ("LK", "", "2025-01-14", "Tamil Thai Pongal Day", "public", 1),
    ("LK", "", "2025-02-04", "Independence Day", "public", 1),
    ("LK", "", "2025-02-12", "Navam Full Moon Poya Day", "public", 1),
    ("MY", "", "2025-01-29", "Chinese New Year", "public", 2),
    ("MY", "", "2025-01-30", "Chinese New Year (Day 2)", "public", 2),
    ("MY", "MY-10", "2025-02-11", "Thaipusam", "public", 1),
    ("MY", "MY-10", "2025-01-01", "New Year's Day", "public", 1),
]
holidays = pd.DataFrame(hol_rows, columns=["country_code", "subdivision_code", "holiday_date",
                                           "holiday_name", "holiday_type", "holiday_period_days"])
holidays["port_closed"] = np.where(holidays["holiday_name"] == "Spring Festival", False, None)
holidays["data_source"] = "public_calendar_manual_entry"
holidays["is_synthetic"] = False

# local-date holiday flag per port-hour
is_hol = np.zeros((P, H), bool)
spring = np.zeros((P, H), bool)
for i, p in enumerate(PORTS):
    local_dates = hours.tz_convert(p[9]).strftime("%Y-%m-%d")
    m = (holidays.country_code == p[2]) & (holidays.subdivision_code.isin(["", p[3]]))
    hd = set(holidays.loc[m, "holiday_date"])
    sf = set(holidays.loc[m & (holidays.holiday_name == "Spring Festival"), "holiday_date"])
    is_hol[i] = np.isin(local_dates, list(hd))
    spring[i] = np.isin(local_dates, list(sf))

# ---------------------------------------------------------------------------
# 4. Weather (simulated, Open-Meteo-like fields)
# ---------------------------------------------------------------------------
temp = np.zeros((P, H)); hum = np.zeros((P, H)); prec = np.zeros((P, H))
wspd = np.zeros((P, H)); gust = np.zeros((P, H)); wdir = np.zeros((P, H))
pres = np.zeros((P, H)); vis = np.zeros((P, H)); wcode = np.zeros((P, H), int)
storm_events = []
for i, p in enumerate(PORTS):
    local_hr = hours.tz_convert(p[9]).hour.values
    temp[i] = p[10] + 4 * np.sin(2 * np.pi * (local_hr - 9) / 24) + rng.normal(0, 1.2, H)
    base_w = rng.uniform(10, 20)
    ar = np.zeros(H)
    for h in range(1, H):
        ar[h] = 0.92 * ar[h - 1] + rng.normal(0, 1.6)
    storm = np.zeros(H)
    for _ in range(rng.poisson(2) + 1):
        s = rng.integers(0, H - 50); d = rng.integers(12, 49); peak = rng.uniform(18, 50)
        storm[s:s + d] = np.maximum(storm[s:s + d], peak * np.sin(np.linspace(0, np.pi, d)))
    wspd[i] = np.clip(base_w + ar + storm, 0.5, None)
    gust[i] = wspd[i] * rng.uniform(1.35, 1.6, H) + rng.normal(0, 2, H)
    wdir[i] = (rng.uniform(0, 360) + np.cumsum(rng.normal(0, 6, H))) % 360
    rain_p = 0.06 + 0.5 * (storm > 10)
    raining = rng.random(H) < rain_p
    prec[i] = np.where(raining, rng.gamma(1.2, 1.5 + storm / 8), 0.0)
    hum[i] = np.clip(68 + 15 * raining + rng.normal(0, 6, H) - 0.3 * (temp[i] - p[10]), 20, 100)
    pres[i] = 1013 - storm * 0.5 + rng.normal(0, 2, H)
    fog = (hum[i] > 95) & (wspd[i] < 8)
    vis[i] = np.clip(24000 - 1500 * prec[i] - 18000 * fog + rng.normal(0, 1500, H), 200, 30000)
    wcode[i] = np.select(
        [storm > 30, prec[i] > 6, prec[i] > 2, prec[i] > 0, fog, hum[i] > 85],
        [95, 65, 63, 61, 45, 3], default=rng.choice([0, 1, 2], H))
    # storm -> weather disruption event
    above = gust[i] > 55
    if above.any():
        edges = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= 3:
                pk = gust[i, a:b].max()
                storm_events.append((p[0], a, b, "weather", float(min(85, 20 + (pk - 55) * 1.8))))

cond_map = {0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
            61: "light rain", 63: "rain", 65: "heavy rain", 95: "thunderstorm"}

# ---------------------------------------------------------------------------
# 5. Disruption events
# ---------------------------------------------------------------------------
ev_spec = {  # type: (prob, dur range h, affected range %)
    "strike": (0.15, (12, 96), (30, 80)),
    "equipment_failure": (0.40, (3, 30), (10, 40)),
    "congestion": (0.35, (24, 120), (15, 45)),
    "cyber": (0.10, (6, 48), (20, 60)),
}
types = list(ev_spec)
events = []
for i, p in enumerate(PORTS):
    for _ in range(rng.poisson(3.5)):
        et = rng.choice(types, p=[ev_spec[t][0] for t in types])
        d = int(rng.integers(*ev_spec[et][1])); s = int(rng.integers(0, H - 10))
        events.append((p[0], s, min(H, s + d), et, float(rng.uniform(*ev_spec[et][2]))))
events += storm_events


def sev(a):
    return "low" if a < 25 else ("medium" if a < 50 else "high")


ev_df = pd.DataFrame(events, columns=["port_id", "s", "e", "event_type", "affected_capacity_pct"])
ev_df = ev_df.sort_values(["s", "port_id"]).reset_index(drop=True)
ev_df["event_id"] = [f"EVT{n:05d}" for n in range(1, len(ev_df) + 1)]
ev_df["severity"] = ev_df.affected_capacity_pct.map(sev)
ev_df["event_start"] = hours[ev_df.s.clip(upper=H - 1)]
ev_df["event_end"] = [t0 + pd.Timedelta(hours=int(e)) for e in ev_df.e]
desc = {
    "strike": "Dock workers' union strike; berth operations reduced",
    "equipment_failure": "Quay crane / straddle carrier breakdown",
    "congestion": "Yard congestion after vessel bunching",
    "cyber": "Terminal operating system outage",
    "weather": "High winds suspended crane operations",
}
ev_df["event_description"] = ev_df.event_type.map(desc)
ev_df["source_url"] = None
ev_df["data_source"] = "synthetic"
ev_df["is_synthetic"] = True

sev_code = {"low": 1, "medium": 2, "high": 3}
ev_cap_cut = np.zeros((P, H)); ev_sev = np.zeros((P, H), int)
strike_on = np.zeros((P, H), bool); equip_fail = np.zeros((P, H), bool)
for r in ev_df.itertuples():
    i = pidx[r.port_id]; sl = slice(r.s, r.e)
    if r.event_type != "weather":  # weather impact comes through gusts
        ev_cap_cut[i, sl] = np.maximum(ev_cap_cut[i, sl], r.affected_capacity_pct / 100)
    ev_sev[i, sl] = np.maximum(ev_sev[i, sl], sev_code[r.severity])
    if r.event_type == "strike": strike_on[i, sl] = True
    if r.event_type == "equipment_failure": equip_fail[i, sl] = True

# ---------------------------------------------------------------------------
# 6. Port hourly telemetry
# ---------------------------------------------------------------------------
util = np.zeros((P, H)); thr = np.zeros((P, H)); backlog = np.zeros((P, H))
dwell = np.zeros((P, H)); vstay = np.zeros((P, H)); arr = np.zeros((P, H), int)
dep = np.zeros((P, H), int); anchor = np.zeros((P, H), int); customs = np.zeros((P, H))
labour = np.empty((P, H), object); equip = np.zeros((P, H))
for i, p in enumerate(PORTS):
    local_hr = hours.tz_convert(p[9]).hour.values
    base_u = rng.uniform(0.62, 0.86)
    diurnal = 1 + 0.12 * np.sin(2 * np.pi * (local_hr - 6) / 24)
    ar = np.zeros(H)
    for h in range(1, H):
        ar[h] = 0.95 * ar[h - 1] + rng.normal(0, 0.02)
    demand = cap[i] * base_u * diurnal * (1 + ar)
    demand *= np.where(spring[i], 0.7, np.where(is_hol[i], 0.88, 1.0))
    wx = np.where(gust[i] > 55, 0.45, np.where(gust[i] > 45, 0.8, 1.0)) * np.where(prec[i] > 8, 0.85, 1.0)
    hol_cap = np.where(spring[i], 0.55, np.where(is_hol[i], 0.75, 1.0))
    cap_eff = cap[i] * (1 - ev_cap_cut[i]) * wx * hol_cap
    B = 0.0; base_dwell = rng.uniform(60, 110)
    lam = cap[i] / 600
    for h in range(H):
        avail = demand[h] + B
        thr[i, h] = min(avail, cap_eff[h])
        B = max(0.0, avail - cap_eff[h])
        backlog[i, h] = B
    util[i] = thr[i] / cap[i] * 100
    bh = backlog[i] / cap[i]  # backlog in hours of capacity
    dwell[i] = base_dwell + 1.6 * np.minimum(bh, 150) + rng.normal(0, 4, H)
    vstay[i] = 18 + 0.35 * np.minimum(bh, 150) + rng.gamma(2, 3, H)
    arr[i] = rng.poisson(lam * diurnal)
    dep[i] = rng.poisson(lam * diurnal * np.clip(cap_eff / np.maximum(demand, 1), 0.2, 1.0))
    anchor[i] = rng.poisson(0.4 + bh / 2.5)
    customs[i] = rng.gamma(2, 8, H) + 12 * is_hol[i]
    lab = np.where(strike_on[i], "strike", np.where(is_hol[i] | (rng.random(H) < 0.03), "reduced", "normal"))
    labour[i] = lab
    equip[i] = np.clip(93 + rng.normal(0, 2.5, H) - np.where(equip_fail[i], rng.uniform(18, 35), 0), 40, 100)
thr = thr.round(1)

# ---------------------------------------------------------------------------
# 7. Network
# ---------------------------------------------------------------------------
def haversine(a, b):
    la1, lo1, la2, lo2 = map(np.radians, [PORTS[a][5], PORTS[a][6], PORTS[b][5], PORTS[b][6]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


routes = []
n = 1
for a in range(P):
    for b in range(P):
        if a == b: continue
        dist = haversine(a, b) * rng.uniform(1.15, 1.35)  # sea distance > great circle
        hubs = [h for h in HUBS if h not in (pid[a], pid[b])]
        trans = rng.random() < 0.35
        hub = rng.choice(hubs) if trans else None
        transit = dist / rng.uniform(30, 38) + (36 if trans else 0) + 12
        routes.append((f"RT{n:04d}", pid[a], pid[b], "transshipment" if trans else "direct", hub,
                       round(dist, 1), round(transit, 1)))
        n += 1
network = pd.DataFrame(routes, columns=["route_id", "origin_port_id", "destination_port_id", "route_type",
                                        "transshipment_port_id", "route_distance_km", "planned_transit_hours"])
network["data_source"] = "synthetic"
network["is_synthetic"] = True
route_lookup = {(r.origin_port_id, r.destination_port_id): r for r in network.itertuples()}

# ---------------------------------------------------------------------------
# 8. Shipments (9,850 unique + 150 duplicates = 10,000)
# ---------------------------------------------------------------------------
NS = 9850
CARRIERS = ["Oceanic Lines", "BlueWave Shipping", "Meridian Cargo", "Pacific Star Line",
            "Coral Bay Logistics", "Northgate Maritime"]
carrier_eff = dict(zip(CARRIERS, [-4, 0, 3, 6, 10, -1]))
w = cap / cap.sum() * 0.6 + 0.4 / P
o = rng.choice(P, NS, p=w)
d = np.array([rng.choice([j for j in range(P) if j != x], p=(w[[j for j in range(P) if j != x]] /
              w[[j for j in range(P) if j != x]].sum())) for x in o])
dep_h = rng.integers(24, H, NS)
mode = np.where(rng.random(NS) < 0.05, "air", "sea")
carrier = rng.choice(CARRIERS, NS, p=[0.22, 0.2, 0.18, 0.16, 0.12, 0.12])
prio = rng.choice(["low", "medium", "high"], NS, p=[0.35, 0.45, 0.2])
cargo = rng.choice(["general", "reefer", "hazardous"], NS, p=[0.78, 0.15, 0.07])
teu = np.clip(rng.gamma(2, 6, NS), 1, 60).round(0)

# rolling features
def back_mean(arr2, h, w=24):
    return arr2[:, max(0, h - w):h].mean(axis=1)

util24 = np.zeros((P, H)); gmax24 = np.zeros((P, H))
for h in range(1, H):
    util24[:, h] = util[:, max(0, h - 24):h].mean(axis=1)
    gmax24[:, h] = gust[:, max(0, h - 24):h].max(axis=1)

sev_w = np.array([0, 8, 20, 40])
rows = []
raw_delay = np.zeros(NS)
for k in range(NS):
    oi, di, h = o[k], d[k], dep_h[k]
    rt = route_lookup[(pid[oi], pid[di])]
    hol_near = is_hol[oi, max(0, h - 24):min(H, h + 24)].any()
    hub_term = 0.0
    if rt.route_type == "transshipment":
        hi = pidx[rt.transshipment_port_id]
        hub_term = 10 + 0.4 * (util24[hi, h] - 75)
    x = (0.55 * (util24[oi, h] - 75) + 0.45 * (util24[di, h] - 75) + 1.5 * anchor[oi, h]
         + 0.9 * max(gmax24[oi, h] - 45, 0) + 0.5 * max(gust[di, h] - 45, 0)
         + sev_w[ev_sev[oi, h]] + 0.8 * sev_w[ev_sev[di, h]] + 14 * hol_near + hub_term
         + carrier_eff[carrier[k]] + {"high": -6, "medium": 0, "low": 3}[prio[k]]
         + (5 if cargo[k] == "reefer" else 0) + (8 if cargo[k] == "hazardous" else 0)
         + rng.normal(0, 7 + rt.route_distance_km / 1500))
    if mode[k] == "air":
        x *= 0.15
    raw_delay[k] = x
    transit = rt.planned_transit_hours if mode[k] == "sea" else rt.route_distance_km / 750 + 18
    rows.append((rt.route_id, round(transit)))

# calibrate so ~55% On-Time, rest split At-Risk / Critical
q55, q85 = np.quantile(raw_delay, [0.55, 0.85])
delay = (raw_delay - q55) * (36 / (q85 - q55)) + 12
delay = np.where(delay < 0, delay * 0.3, delay).round(1)  # early arrivals are modest
status = np.where(delay <= 12, "On-Time", np.where(delay <= 48, "At-Risk", "Critical"))

dep_ts = hours[dep_h]
exp_ts = dep_ts + pd.to_timedelta([r[1] for r in rows], unit="h")
act_ts = exp_ts + pd.to_timedelta(delay, unit="h")
ship = pd.DataFrame({
    "shipment_id": [f"SHP{n:06d}" for n in range(1, NS + 1)],
    "route_id": [r[0] for r in rows],
    "origin_port_id": [pid[i] for i in o],
    "destination_port_id": [pid[i] for i in d],
    "carrier": carrier, "shipping_mode": mode, "cargo_type": cargo,
    "container_teu": teu, "shipment_priority": prio,
    "departure_ts": dep_ts, "expected_arrival_ts": exp_ts, "actual_arrival_ts": act_ts.round("min"),
    "actual_delay_hours": delay, "delay_status": status,
    "data_source": "synthetic", "is_synthetic": True,
})
# shuffle order so it doesn't look generated by time
ship = ship.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 9. Assemble clean telemetry / weather / traffic frames
# ---------------------------------------------------------------------------
tel = pd.DataFrame({
    "port_id": np.repeat(pid, H),
    "ts_hour": np.tile(hours, P),
    "throughput_teu": thr.ravel(),
    "port_utilization_pct": util.ravel().round(2),
    "container_dwell_hours": dwell.ravel().round(1),
    "vessel_time_in_port_hours": vstay.ravel().round(1),
    "vessel_arrivals": arr.ravel(),
    "vessel_departures": dep.ravel(),
    "vessels_at_anchor": anchor.ravel(),
    "customs_hold_hours": customs.ravel().round(1),
    "labour_status": labour.ravel(),
    "equipment_availability_pct": equip.ravel().round(1),
    "customs_hold_is_synthetic": True,
    "labour_status_is_synthetic": True,
    "equipment_is_synthetic": True,
    "data_source": "synthetic",
})
wx = pd.DataFrame({
    "port_id": np.repeat(pid, H),
    "ts_hour": np.tile(hours, P),
    "grid_latitude": np.repeat([round(p[5] * 4) / 4 for p in PORTS], H),
    "grid_longitude": np.repeat([round(p[6] * 4) / 4 for p in PORTS], H),
    "temperature_2m_c": temp.ravel().round(1),
    "relative_humidity_2m_pct": hum.ravel().round(0),
    "precipitation_mm": prec.ravel().round(1),
    "wind_speed_10m_kmh": wspd.ravel().round(1),
    "wind_gusts_10m_kmh": gust.ravel().round(1),
    "wind_direction_10m_deg": wdir.ravel().round(0),
    "surface_pressure_hpa": pres.ravel().round(1),
    "visibility_m": vis.ravel().round(-1),
    "weather_code": wcode.ravel(),
    "weather_condition": pd.Series(wcode.ravel()).map(cond_map).values,
    "data_source": "synthetic (Open-Meteo field layout)",
})
tel["_day"] = tel.ts_hour.dt.strftime("%Y-%m-%d")
traffic = tel.groupby(["port_id", "_day"]).agg(
    vessel_calls=("vessel_arrivals", "sum"), teu=("throughput_teu", "sum"),
    avg_vessel_stay_hours=("vessel_time_in_port_hours", "mean")).reset_index()
traffic = traffic.rename(columns={"_day": "period_start"})
tel = tel.drop(columns="_day")
traffic["period_granularity"] = "daily"
traffic["container_vessel_calls"] = (traffic.vessel_calls * rng.uniform(0.6, 0.8, len(traffic))).round().astype(int)
traffic["import_volume_t"] = (traffic.teu * rng.uniform(5.5, 7.5, len(traffic))).round(0)
traffic["export_volume_t"] = (traffic.teu * rng.uniform(4.5, 6.5, len(traffic))).round(0)
base = traffic.groupby("port_id").vessel_calls.transform("mean")
traffic["traffic_deviation_pct"] = ((traffic.vessel_calls - base) / base * 100).round(1)
traffic["avg_vessel_stay_hours"] = traffic.avg_vessel_stay_hours.round(1)
traffic["data_source"] = "synthetic (PortWatch-like)"
traffic = traffic[["port_id", "period_start", "period_granularity", "vessel_calls", "container_vessel_calls",
                   "import_volume_t", "export_volume_t", "traffic_deviation_pct", "avg_vessel_stay_hours",
                   "data_source"]]

# keep a clean copy for the reference set
clean = {
    "port_metadata": port_meta.copy(), "port_network": network.copy(),
    "port_hourly_telemetry": tel.copy(), "weather_hourly": wx.copy(),
    "port_traffic_history": traffic.copy(),
    "disruption_events": ev_df[["event_id", "port_id", "event_start", "event_end", "event_type", "severity",
                                "affected_capacity_pct", "event_description", "source_url", "data_source",
                                "is_synthetic"]].copy(),
    "public_holidays": holidays.copy(), "shipment_records": ship.copy(),
}

# ===========================================================================
# MESS IT UP
# ===========================================================================
NAMES = {p[0]: p[1].split(" (")[0].replace("Port of ", "").replace(" Port", "") for p in PORTS}


def obj(df, cols):
    for c in cols:
        df[c] = df[c].astype(object)


def messy_port_ids(df, col, table, frac_lower=0.02, frac_space=0.01, frac_typo=0.005, frac_name=0.005):
    obj(df, [col])
    idx = np.arange(len(df))
    used = []
    a = pick(idx, frac_lower); df.loc[a, col] = df.loc[a, col].str.lower(); used += list(a)
    log(table, col, "lowercase port code (e.g. 'inmaa')", len(a))
    b = pick(idx, frac_space, used); df.loc[b, col] = df.loc[b, col] + " "; used += list(b)
    log(table, col, "trailing whitespace in port code", len(b))
    typos = {"INMAA": "INMMA", "SGSIN": "SGSN", "CNSHA": "CNSAH", "NLRTM": "NLRMT", "USLAX": "USLA",
             "DEHAM": "DEHMA", "AEJEA": "AEJAE", "INNSA": "INSNA", "LKCMB": "LKCBM", "MYPKG": "MYPGK"}
    c = pick(idx, frac_typo, used); df.loc[c, col] = df.loc[c, col].map(typos); used += list(c)
    log(table, col, "misspelled port code (e.g. 'INMMA')", len(c))
    e = pick(idx, frac_name, used); df.loc[e, col] = df.loc[e, col].map(NAMES)
    log(table, col, "port name instead of code (e.g. 'Chennai')", len(e))


def fmt_ts(ts_series, table, col, mix):
    """mix: dict format_name -> fraction. Remaining rows ISO 'YYYY-MM-DD HH:MM:SS'."""
    s = pd.to_datetime(pd.Series(ts_series).reset_index(drop=True), utc=True)
    out = s.dt.strftime("%Y-%m-%d %H:%M:%S").astype(object)
    idx = np.arange(len(s)); used = []
    for name, frac in mix.items():
        sel = pick(idx, frac, used); used += list(sel)
        if not len(sel): continue
        v = s.iloc[sel]
        if name == "dmy":
            out.iloc[sel] = v.dt.strftime("%d/%m/%Y %H:%M"); lab = "DD/MM/YYYY HH:MM"
        elif name == "mdy_ampm":
            out.iloc[sel] = v.dt.strftime("%m-%d-%Y %I:%M %p"); lab = "MM-DD-YYYY hh:mm AM/PM"
        elif name == "isoz":
            out.iloc[sel] = v.dt.strftime("%Y-%m-%dT%H:%M:%SZ"); lab = "ISO 8601 with T and Z"
        elif name == "local_offset":
            out.iloc[sel] = v.dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%dT%H:%M:%S+05:30")
            lab = "converted to IST with +05:30 offset (same instant)"
        elif name == "epoch":
            out.iloc[sel] = (v.astype("int64") // 10**9).astype(str); lab = "Unix epoch seconds"
        elif name == "local_naive":
            out.iloc[sel] = v.dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d %H:%M:%S")
            lab = "IST local time with NO timezone marker (silently 5.5 h off)"
        elif name == "date_only":
            out.iloc[sel] = v.dt.strftime("%Y-%m-%d"); lab = "date only, time lost"
        log(table, col, f"timestamp format: {lab}", len(sel))
    return out.values


def set_missing(df, col, frac, table, token=None, exclude=None):
    obj(df, [col])
    sel = pick(np.arange(len(df)), frac, exclude)
    df.loc[sel, col] = token if token is not None else None
    what = f"missing, written as '{token}'" if token is not None else "missing (empty)"
    log(table, col, what, len(sel))
    return sel


def variants(df, col, table, mapping, frac):
    """Replace a fraction of each canonical value with random spelling variants."""
    obj(df, [col])
    tot = 0
    for canon, vars_ in mapping.items():
        idx = np.flatnonzero(df[col].values == canon)
        sel = pick(idx, frac)
        df.loc[sel, col] = rng.choice(vars_, len(sel))
        tot += len(sel)
    log(table, col, "inconsistent spelling / casing of categories", tot)


def dup_and_gap(df, table, n_drop, n_dup, near_dupe_col=None):
    keep = np.setdiff1d(np.arange(len(df)), rng.choice(len(df), n_drop, replace=False))
    if n_drop: log(table, "(rows)", "rows missing entirely (gaps in the hourly series)", n_drop)
    df = df.iloc[keep].reset_index(drop=True)
    d = df.iloc[rng.choice(len(df), n_dup, replace=False)].copy()
    df = pd.concat([df, d], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED + 1).reset_index(drop=True)
    log(table, "(rows)", "exact duplicate rows", n_dup)
    return df


# ---- port_metadata ------------------------------------------------------
pm = port_meta.copy()
obj(pm, list(pm.columns))
pm.loc[pidx["DEHAM"], ["latitude", "longitude"]] = [9.98, 53.54]; log("port_metadata", "latitude/longitude", "lat and lon swapped (Hamburg)", 1)
pm.loc[pidx["USLAX"], "country_code"] = "us"; log("port_metadata", "country_code", "lowercase country code", 1)
pm.loc[pidx["MYPKG"], "capacity_teu_per_hour"] = None; log("port_metadata", "capacity_teu_per_hour", "missing capacity", 1)
pm.loc[pidx["LKCMB"], "local_timezone"] = "IST"; log("port_metadata", "local_timezone", "ambiguous abbreviation 'IST' instead of IANA zone", 1)
pm.loc[pidx["SGSIN"], "subdivision_code"] = "-"; log("port_metadata", "subdivision_code", "placeholder '-' instead of blank for 'no subdivision'", 1)
pm.loc[pidx["NLRTM"], "port_type"] = "Mixed "; log("port_metadata", "port_type", "casing/whitespace", 1)
pm.loc[pidx["CNSHA"], "capacity_teu_per_hour"] = "5,600"; log("port_metadata", "capacity_teu_per_hour", "number stored as text with thousands separator", 1)
dupe = pm.iloc[[pidx["INMAA"]]].copy(); dupe["port_name"] = "Chennai"; dupe["region"] = "south asia"
pm = pd.concat([pm, dupe], ignore_index=True); log("port_metadata", "(rows)", "duplicate port with different name spelling", 1)
pm["is_synthetic"] = rng.choice(["TRUE", "True", "1", "yes"], len(pm))
log("port_metadata", "is_synthetic", "boolean written as TRUE/True/1/yes", len(pm))

# ---- port_network -------------------------------------------------------
nw = network.copy()
obj(nw, ["route_distance_km", "route_type", "transshipment_port_id", "planned_transit_hours"])
s = pick(np.arange(len(nw)), 4); nw.loc[s, "route_distance_km"] = None; log("port_network", "route_distance_km", "missing", 4)
s = pick(np.arange(len(nw)), 3, s); nw.loc[s, "route_distance_km"] = [f"{float(network.route_distance_km[i]) * 0.621:.0f} mi" for i in s]
log("port_network", "route_distance_km", "value in miles stored as text ('4520 mi')", 3)
s2 = pick(np.arange(len(nw)), 1); nw.loc[s2, "route_distance_km"] = -abs(float(network.route_distance_km[s2[0]])); log("port_network", "route_distance_km", "negative distance", 1)
variants(nw, "route_type", "port_network", {"direct": ["Direct", "DIRECT", "direct "], "transshipment": ["Transshipment", "transhipment", "T/S"]}, 0.12)
dd = np.flatnonzero(network.route_type.values == "direct")
s = pick(dd, 2); nw.loc[s, "transshipment_port_id"] = "SGSIN"; log("port_network", "transshipment_port_id", "hub port filled on a direct route", 2)
s = pick(np.arange(len(nw)), 3); nw.loc[s, "planned_transit_hours"] = [f"{float(network.planned_transit_hours[i]) / 24:.1f} days" for i in s]
log("port_network", "planned_transit_hours", "value in days stored as text", 3)
nw = pd.concat([nw, nw.iloc[[5, 40]]], ignore_index=True); log("port_network", "(rows)", "exact duplicate rows", 2)

# ---- telemetry ----------------------------------------------------------
t = dup_and_gap(tel.copy(), "port_hourly_telemetry", 120, 120)
t["ts_hour"] = fmt_ts(t.ts_hour, "port_hourly_telemetry", "ts_hour",
                      {"dmy": 0.06, "isoz": 0.08, "epoch": 0.02, "local_naive": 0.02})
messy_port_ids(t, "port_id", "port_hourly_telemetry", 0.02, 0.01, 0.003, 0.003)
N = len(t); I = np.arange(N)
obj(t, ["throughput_teu", "port_utilization_pct", "container_dwell_hours", "vessel_arrivals",
        "equipment_availability_pct", "vessels_at_anchor", "customs_hold_hours", "vessel_time_in_port_hours"])
set_missing(t, "throughput_teu", 0.02, "port_hourly_telemetry")
s = pick(I, 0.003); t.loc[s, "throughput_teu"] = -t.loc[s, "throughput_teu"].astype(float).abs(); log("port_hourly_telemetry", "throughput_teu", "negative throughput", len(s))
s = pick(I, 0.005); t.loc[s, "port_utilization_pct"] = rng.uniform(120, 180, len(s)).round(1); log("port_hourly_telemetry", "port_utilization_pct", "impossible utilization > 100% (sensor error)", len(s))
s = pick(I, 0.01, s); t.loc[s, "port_utilization_pct"] = (t.loc[s, "port_utilization_pct"].astype(float) / 100).round(3); log("port_hourly_telemetry", "port_utilization_pct", "stored as a fraction (0.82) instead of percent", len(s))
s = pick(I, 0.02); t.loc[s, "container_dwell_hours"] = [f"{float(v)}h" for v in t.loc[s, "container_dwell_hours"]]; log("port_hourly_telemetry", "container_dwell_hours", "number with unit suffix ('84.2h')", len(s))
s2 = pick(I, 0.005, s); t.loc[s2, "container_dwell_hours"] = -999; log("port_hourly_telemetry", "container_dwell_hours", "sentinel value -999", len(s2))
set_missing(t, "container_dwell_hours", 0.03, "port_hourly_telemetry", exclude=np.r_[s, s2])
s = pick(I, 0.01); t.loc[s, "vessel_arrivals"] = [f"{v}.0" for v in t.loc[s, "vessel_arrivals"]]; log("port_hourly_telemetry", "vessel_arrivals", "integer stored as '3.0'", len(s))
set_missing(t, "vessel_arrivals", 0.01, "port_hourly_telemetry", exclude=s)
set_missing(t, "vessels_at_anchor", 0.02, "port_hourly_telemetry")
set_missing(t, "customs_hold_hours", 0.05, "port_hourly_telemetry", token="N/A")
s = pick(I, 0.003); t.loc[s, "vessel_time_in_port_hours"] = rng.uniform(900, 2000, len(s)).round(1); log("port_hourly_telemetry", "vessel_time_in_port_hours", "extreme outlier (> 900 h)", len(s))
variants(t, "labour_status", "port_hourly_telemetry",
         {"normal": ["Normal", "NORMAL", "norm", "normal "], "reduced": ["Reduced", "REDUCED", "partial"],
          "strike": ["STRIKE", "Strike", "on strike"]}, 0.06)
set_missing(t, "labour_status", 0.02, "port_hourly_telemetry")
s = pick(I, 0.02); t.loc[s, "equipment_availability_pct"] = [f"{float(v)}%" for v in t.loc[s, "equipment_availability_pct"]]; log("port_hourly_telemetry", "equipment_availability_pct", "number with '%' sign", len(s))
s2 = pick(I, 0.003, s); t.loc[s2, "equipment_availability_pct"] = rng.uniform(101, 140, len(s2)).round(1); log("port_hourly_telemetry", "equipment_availability_pct", "value > 100%", len(s2))
t["customs_hold_is_synthetic"] = rng.choice(["TRUE", "True", "1"], N)
log("port_hourly_telemetry", "*_is_synthetic", "boolean written as TRUE/True/1", N)

# ---- weather ------------------------------------------------------------
w_ = dup_and_gap(wx.copy(), "weather_hourly", 100, 100)
# Fahrenheit block for Colombo (before ids get messy)
obj(w_, ["temperature_2m_c", "relative_humidity_2m_pct", "precipitation_mm", "visibility_m",
         "wind_speed_10m_kmh", "wind_gusts_10m_kmh", "weather_condition", "surface_pressure_hpa"])
cmb = np.flatnonzero((w_.port_id == "LKCMB").values & (pd.to_datetime(w_.ts_hour) < pd.Timestamp("2025-01-08", tz="UTC")).values)
w_.loc[cmb, "temperature_2m_c"] = (w_.loc[cmb, "temperature_2m_c"].astype(float) * 9 / 5 + 32).round(1)
log("weather_hourly", "temperature_2m_c", "Fahrenheit values in a Celsius column (Colombo, 1-7 Jan)", len(cmb))
noviz = np.flatnonzero(w_.port_id.isin(["CNSHA", "USLAX", "LKCMB"]).values)
w_.loc[noviz, "visibility_m"] = None; log("weather_hourly", "visibility_m", "entire ports with no visibility data (source gap)", len(noviz))
w_["ts_hour"] = fmt_ts(w_.ts_hour, "weather_hourly", "ts_hour", {"isoz": 0.6, "epoch": 0.03, "dmy": 0.02})
messy_port_ids(w_, "port_id", "weather_hourly", 0.01, 0.01, 0.002, 0.01)
N = len(w_); I = np.arange(N)
for col in ["temperature_2m_c", "precipitation_mm", "surface_pressure_hpa"]:
    s = pick(I, 0.01, cmb if col == "temperature_2m_c" else None); w_.loc[s, col] = -9999
    log("weather_hourly", col, "sentinel value -9999", len(s))
s = pick(I, 0.002); w_.loc[s, "precipitation_mm"] = -w_.loc[s, "precipitation_mm"].astype(float).abs() - 0.1; log("weather_hourly", "precipitation_mm", "negative rainfall", len(s))
s = pick(I, 0.003); w_.loc[s, "relative_humidity_2m_pct"] = rng.uniform(101, 130, len(s)).round(0); log("weather_hourly", "relative_humidity_2m_pct", "humidity > 100%", len(s))
s = pick(I, 0.005)
tmp = w_.loc[s, "wind_speed_10m_kmh"].values.copy()
w_.loc[s, "wind_speed_10m_kmh"] = w_.loc[s, "wind_gusts_10m_kmh"].values; w_.loc[s, "wind_gusts_10m_kmh"] = tmp
log("weather_hourly", "wind_speed / wind_gusts", "columns swapped (speed > gust)", len(s))
variants(w_, "weather_condition", "weather_hourly",
         {"clear": ["Clear", "CLEAR", "sunny"], "light rain": ["Light Rain", "drizzle"], "rain": ["Rain", "RAIN", "rainy"],
          "thunderstorm": ["Thunderstorm", "storm", "T-Storm"], "overcast": ["Overcast", "cloudy"], "fog": ["Fog", "mist"]}, 0.08)
s = pick(I, 0.01); w_.loc[s, "weather_condition"] = rng.choice(["clear", "rain", "thunderstorm"], len(s)); log("weather_hourly", "weather_condition", "label contradicts weather_code", len(s))
set_missing(w_, "wind_gusts_10m_kmh", 0.015, "weather_hourly")
set_missing(w_, "relative_humidity_2m_pct", 0.01, "weather_hourly", token="NaN")

# ---- traffic history ----------------------------------------------------
tr = traffic.copy()
obj(tr, ["vessel_calls", "traffic_deviation_pct", "avg_vessel_stay_hours", "import_volume_t", "period_start", "data_source"])
monthly = traffic.groupby("port_id").agg(vessel_calls=("vessel_calls", "sum"),
                                         container_vessel_calls=("container_vessel_calls", "sum"),
                                         import_volume_t=("import_volume_t", "sum"),
                                         export_volume_t=("export_volume_t", "sum"),
                                         avg_vessel_stay_hours=("avg_vessel_stay_hours", "mean")).reset_index()
monthly["period_start"] = "2025-01-01"; monthly["period_granularity"] = "monthly"
monthly["traffic_deviation_pct"] = None; monthly["avg_vessel_stay_hours"] = monthly.avg_vessel_stay_hours.round(1)
monthly["data_source"] = "UNCTAD-like (synthetic)"
tr = pd.concat([tr, monthly[tr.columns]], ignore_index=True)
log("port_traffic_history", "period_granularity", "monthly totals mixed in with daily rows (do not sum together)", len(monthly))
I = np.arange(len(tr))
s = pick(I, 0.05); tr.loc[s, "traffic_deviation_pct"] = (tr.loc[s, "traffic_deviation_pct"].astype(float) / 100).round(3); log("port_traffic_history", "traffic_deviation_pct", "fraction instead of percent", len(s))
set_missing(tr, "avg_vessel_stay_hours", 0.04, "port_traffic_history")
set_missing(tr, "import_volume_t", 0.03, "port_traffic_history", token="-")
s = pick(I, 0.05); tr.loc[s, "period_start"] = pd.to_datetime(tr.loc[s, "period_start"]).dt.strftime("%d-%b-%Y"); log("port_traffic_history", "period_start", "date as '05-Jan-2025'", len(s))
s = pick(I, 0.1); tr.loc[s, "data_source"] = tr.loc[s, "data_source"].astype(str).str.upper(); log("port_traffic_history", "data_source", "inconsistent casing", len(s))
messy_port_ids(tr, "port_id", "port_traffic_history", 0.02, 0.0, 0.0, 0.0)
tr = pd.concat([tr, tr.iloc[pick(np.arange(len(tr)), 8)]], ignore_index=True); log("port_traffic_history", "(rows)", "exact duplicate rows", 8)

# ---- disruption events --------------------------------------------------
ev = clean["disruption_events"].copy()
obj(ev, list(ev.columns))
I = np.arange(len(ev))
ev["event_start"] = fmt_ts(ev.event_start, "disruption_events", "event_start", {"dmy": 0.15, "mdy_ampm": 0.1})
ev["event_end"] = fmt_ts(clean["disruption_events"].event_end, "disruption_events", "event_end", {"dmy": 0.1})
s = pick(I, 3); ev.loc[s, "event_end"] = None; log("disruption_events", "event_end", "missing end time", 3)
s2 = pick(I, 2, s); ev.loc[s2, ["event_start", "event_end"]] = ev.loc[s2, ["event_end", "event_start"]].values; log("disruption_events", "event_start/event_end", "start and end swapped (end before start)", 2)
variants(ev, "severity", "disruption_events", {"low": ["Low", "LOW", "minor"], "medium": ["Medium", "Med", "moderate"], "high": ["High", "HIGH", "Hi", "severe"]}, 0.3)
variants(ev, "event_type", "disruption_events", {"strike": ["Strike", "labour strike"], "equipment_failure": ["Equipment Failure", "equip failure", "crane breakdown"],
                                                 "weather": ["Weather", "storm"], "congestion": ["Congestion", "yard congestion"]}, 0.25)
s = pick(I, 2); ev.loc[s, "affected_capacity_pct"] = [125.0, 0.45][:len(s)]; log("disruption_events", "affected_capacity_pct", "out of range (125) / fraction (0.45)", len(s))
ev["affected_capacity_pct"] = [round(float(v), 1) if v is not None else v for v in ev.affected_capacity_pct]
messy_port_ids(ev, "port_id", "disruption_events", 0.05, 0.03, 0.0, 0.03)
ev = pd.concat([ev, ev.iloc[[3]]], ignore_index=True); log("disruption_events", "(rows)", "exact duplicate rows", 1)

# ---- holidays -----------------------------------------------------------
hl = holidays.copy()
obj(hl, list(hl.columns))
I = np.arange(len(hl))
s = pick(I, 5); hl.loc[s, "holiday_date"] = pd.to_datetime(hl.loc[s, "holiday_date"]).dt.strftime("%d/%m/%Y"); log("public_holidays", "holiday_date", "date as DD/MM/YYYY", 5)
s = pick(I, 3, s); hl.loc[s, "holiday_date"] = pd.to_datetime(hl.loc[s, "holiday_date"]).dt.strftime("%B %d, %Y"); log("public_holidays", "holiday_date", "date as 'January 29, 2025'", 3)
e = np.flatnonzero(hl.subdivision_code.values == ""); s = pick(e, 6); hl.loc[s, "subdivision_code"] = rng.choice(["NATIONAL", "N/A", "ALL"], len(s)); log("public_holidays", "subdivision_code", "placeholder text (NATIONAL / N/A / ALL) instead of blank for national holidays", 6)
s = pick(I, 4); hl.loc[s, "country_code"] = hl.loc[s, "country_code"].str.lower(); log("public_holidays", "country_code", "lowercase country code", 4)
hl = pd.concat([hl, hl.iloc[[1]]], ignore_index=True); hl.loc[len(hl) - 1, "holiday_name"] = "Thai Pongal"
log("public_holidays", "(rows)", "same holiday listed twice under different names", 1)

# ---- shipments ----------------------------------------------------------
sh = ship.copy()
sh = pd.concat([sh, sh.iloc[rng.choice(NS, 150, replace=False)]], ignore_index=True)
sh = sh.sample(frac=1, random_state=SEED + 2).reset_index(drop=True)
log("shipment_records", "(rows)", "exact duplicate rows (same shipment_id)", 150)
N = len(sh); I = np.arange(N)
dep_true = pd.to_datetime(sh.departure_ts); exp_true = pd.to_datetime(sh.expected_arrival_ts); act_true = pd.to_datetime(sh.actual_arrival_ts)
sh["departure_ts"] = fmt_ts(dep_true, "shipment_records", "departure_ts",
                            {"dmy": 0.10, "mdy_ampm": 0.05, "isoz": 0.05, "local_offset": 0.05, "epoch": 0.03})
sh["expected_arrival_ts"] = fmt_ts(exp_true, "shipment_records", "expected_arrival_ts", {"dmy": 0.05, "date_only": 0.02})
sh["actual_arrival_ts"] = fmt_ts(act_true, "shipment_records", "actual_arrival_ts", {"dmy": 0.08, "isoz": 0.05})
obj(sh, list(sh.columns))
s = pick(I, 0.005); sh.loc[s, ["departure_ts", "expected_arrival_ts"]] = sh.loc[s, ["expected_arrival_ts", "departure_ts"]].values
log("shipment_records", "departure_ts/expected_arrival_ts", "values swapped (arrival before departure)", len(s))
set_missing(sh, "expected_arrival_ts", 0.02, "shipment_records", exclude=s)
miss_act = set_missing(sh, "actual_arrival_ts", 0.03, "shipment_records")
for c in ["origin_port_id", "destination_port_id"]:
    messy_port_ids(sh, c, "shipment_records")
set_missing(sh, "route_id", 0.01, "shipment_records")
variants(sh, "carrier", "shipment_records", {"Oceanic Lines": ["OCEANIC LINES", "Oceanic Line", "oceanic lines"],
         "BlueWave Shipping": ["Bluewave Shipping", "BWS", "Blue Wave Shipping"], "Meridian Cargo": ["MERIDIAN CARGO", "Meridian Cargo Ltd."],
         "Pacific Star Line": ["Pacific Star", "PSL", "pacific star line "], "Coral Bay Logistics": ["Coral Bay", "CORAL BAY LOGISTICS"],
         "Northgate Maritime": ["NorthGate Maritime", "Northgate"]}, 0.08)
set_missing(sh, "carrier", 0.02, "shipment_records")
variants(sh, "shipping_mode", "shipment_records", {"sea": ["Sea", "SEA", "sea ", "Ocean", "ocean freight"], "air": ["Air", "AIR", "air freight"]}, 0.15)
set_missing(sh, "shipping_mode", 0.01, "shipment_records")
variants(sh, "cargo_type", "shipment_records", {"general": ["General", "GENERAL", "gen", "dry"], "reefer": ["Reefer ", "refrigerated", "REEFER"],
         "hazardous": ["HAZMAT", "dangerous goods", "Hazardous"]}, 0.1)
set_missing(sh, "cargo_type", 0.03, "shipment_records")
variants(sh, "shipment_priority", "shipment_records", {"high": ["High", "H", "HIGH", "urgent"], "medium": ["Medium", "Med", "M"], "low": ["Low", "L", "LOW"]}, 0.12)
set_missing(sh, "shipment_priority", 0.02, "shipment_records")
s = pick(I, 0.003); sh.loc[s, "container_teu"] = 9999; log("shipment_records", "container_teu", "placeholder 9999", len(s))
s2 = pick(I, 0.003, s); sh.loc[s2, "container_teu"] = -sh.loc[s2, "container_teu"].astype(float); log("shipment_records", "container_teu", "negative TEU", len(s2))
s3 = pick(I, 0.005, np.r_[s, s2]); sh.loc[s3, "container_teu"] = [f"{int(float(v))} TEU" for v in sh.loc[s3, "container_teu"]]; log("shipment_records", "container_teu", "number with unit text ('12 TEU')", len(s3))
set_missing(sh, "container_teu", 0.02, "shipment_records", exclude=np.r_[s, s2, s3])
# delay column issues
s = pick(I, 0.01, miss_act); sh.loc[s, "actual_delay_hours"] = (sh.loc[s, "actual_delay_hours"].astype(float) / 24).round(2)
log("shipment_records", "actual_delay_hours", "recorded in days instead of hours (disagrees with timestamps)", len(s))
s2 = pick(I, 0.005, np.r_[s, miss_act]); sh.loc[s2, "actual_delay_hours"] = -sh.loc[s2, "actual_delay_hours"].astype(float)
log("shipment_records", "actual_delay_hours", "sign flipped (disagrees with timestamps)", len(s2))
s3 = pick(I, 0.01, np.r_[s, s2, miss_act]); sh.loc[s3, "actual_delay_hours"] = [f"{float(v)} hrs" for v in sh.loc[s3, "actual_delay_hours"]]
log("shipment_records", "actual_delay_hours", "number with unit text ('36.5 hrs')", len(s3))
sh.loc[miss_act, "actual_delay_hours"] = None
s4 = pick(I, 0.01, np.r_[s, s2, s3, miss_act]); sh.loc[s4, "actual_delay_hours"] = None; log("shipment_records", "actual_delay_hours", "missing although actual_arrival_ts is present (recompute)", len(s4))
# label issues
variants(sh, "delay_status", "shipment_records", {"On-Time": ["On Time", "on-time", "ONTIME", "ontime"],
         "At-Risk": ["At Risk", "at-risk", "AtRisk", "AT RISK"], "Critical": ["CRITICAL", "Critical Delay", "critical"]}, 0.1)
s = pick(I, 0.01); sh.loc[s, "delay_status"] = "Late"; log("shipment_records", "delay_status", "ambiguous label 'Late' (could be At-Risk or Critical)", len(s))
s2 = pick(I, 0.01, s); sh.loc[s2, "delay_status"] = rng.choice(["On-Time", "Critical"], len(s2)); log("shipment_records", "delay_status", "label contradicts actual_delay_hours", len(s2))
sh.loc[miss_act, "delay_status"] = None; log("shipment_records", "delay_status", "missing (shipment not yet arrived / unrecorded)", len(miss_act))
sh["is_synthetic"] = rng.choice(["TRUE", "True", "1", "yes"], N)
log("shipment_records", "is_synthetic", "boolean written as TRUE/True/1/yes", N)

# ---------------------------------------------------------------------------
# WRITE
# ---------------------------------------------------------------------------
raw = {
    "port_metadata": pm, "port_network": nw, "port_hourly_telemetry": t, "weather_hourly": w_,
    "port_traffic_history": tr, "disruption_events": ev, "public_holidays": hl, "shipment_records": sh,
}
for name, df in raw.items():
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)

ref = OUT + "_clean_reference"
os.makedirs(ref, exist_ok=True)
for name, df in clean.items():
    df.to_csv(os.path.join(ref, f"{name}.csv"), index=False)

pd.DataFrame(ISSUES, columns=["table", "column", "issue", "rows_affected"]).to_csv(
    os.path.join(OUT, "_data_issues_log.csv"), index=False)

summary = {k: len(v) for k, v in raw.items()}
summary["label_balance_clean"] = ship.delay_status.value_counts(normalize=True).round(3).to_dict()
print(json.dumps(summary, indent=2))
