"""CargoMind Streamlit application (scaffold only; dashboards are not implemented yet).

Run from the repository root after installing streamlit_app/requirements.txt:

    .venv/bin/streamlit run streamlit_app/app.py

The app consumes two independent workflows through thin services and never re-implements model logic:
  * services/ml_risk.py        -> ML workflow (ML/): shipment delay-risk classification
  * services/dl_forecasting.py -> DL workflow (DL/): next-hour port dwell-time forecasting (dl.inference)
"""
import streamlit as st

st.set_page_config(page_title="CargoMind", layout="wide")
st.title("CargoMind")
st.caption("Research prototype trained on synthetic data. Not validated for real port operations.")
st.info("Scaffold only: dashboard pages will be added under streamlit_app/pages/.")
