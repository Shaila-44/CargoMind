"""CargoMind Streamlit application: navigation entry point.

Run from the repository root after installing streamlit_app/requirements.txt:

    .venv/bin/streamlit run streamlit_app/app.py

The pages call the thin services in streamlit_app/services/ and never re-implement model logic:
  * services/ml_risk.py        -> ML workflow (ML/): shipment delay class (XGBoost) and operating-conditions flag (Isolation Forest)
  * services/dl_forecasting.py -> DL workflow (DL/): next-hour port dwell-time forecast (Transformer T_w, C_w fallback)
  * services/port_ops.py       -> observed synthetic port history replayed at a chosen hour (no model, no live feed)

Light / dark mode: menu (top right) > Settings > Theme. The custom panels and charts follow the active theme.
"""
import streamlit as st

st.set_page_config(page_title="CargoMind", layout="wide")

st.navigation([
    st.Page("overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("pages/1_ML_Intelligence.py", title="ML Intelligence", icon=":material/insights:"),
    st.Page("pages/2_Port_Forecast.py", title="Port Forecast", icon=":material/timeline:"),
    st.Page("pages/3_Methodology.py", title="Methodology", icon=":material/menu_book:"),
]).run()
