"""Sample prediction: one validation-period window (read-only) through the inference interface.

    .venv/bin/python -m dl.inference.demo [--model transformer_t_w] [--port SGSIN] [--hour 811]

`--hour` is the last input hour index t (the window covers t-11..t; the forecast is for t+1).
"""
import argparse
import json

import pandas as pd

from dl.inference import forecast
from cargomind_dl.config import DATASET


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="transformer_t_w")
    ap.add_argument("--port", default="SGSIN")
    ap.add_argument("--hour", type=int, default=811)
    a = ap.parse_args()
    panel = pd.read_csv(DATASET / "hourly_panel.csv")
    p = panel[panel["port_id"] == a.port]
    rows = p[p["hour_index"].between(a.hour - 11, a.hour)]
    cols = ["ts_hour", "dwell", "utilization", "labour_strike", "wind_gust", "disruption_active", "disruption_hours_elapsed",
            "vessels_at_anchor"]
    window = rows[cols].assign(dwell_reconstructed=(rows["prov_dwell"] == 2).astype(int).values).reset_index(drop=True)
    print(window.to_string(index=False))
    # production call (transformer_t_w with automatic c_w_lstm fallback), then the EWMA benchmark for reference
    for model in (a.model, "ewma"):
        print(json.dumps(forecast(window, a.port, model=model).to_dict(), indent=2))
    actual = p.loc[p["hour_index"] == a.hour + 1, "dwell"]
    print(f"actual dwell at t+1: {actual.iloc[0] if len(actual) else 'n/a'} h")


if __name__ == "__main__":
    main()
