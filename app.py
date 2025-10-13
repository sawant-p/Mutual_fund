import streamlit as st
from prediction_module import run_prediction_app
from mftool_analysis import run_mftool_app

# --- Main App Configuration ---
st.set_page_config(
    page_title="Integrated Mutual Fund Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar for Main Navigation ---
st.sidebar.title("App Navigation")

main_option = st.sidebar.selectbox(
    "Select Application Mode",
    [
        "ML Fund Prediction Dashboard",
        "Mutual Fund Market Analysis (MFTool)"
    ],
    key="main_app_mode"
)

st.title("💰 Integrated Financial Assistant")
st.markdown("A comprehensive tool combining **Machine Learning predictions** and **live market data analysis**.")
st.markdown("---")

# --- Routing Logic ---

if main_option == "ML Fund Prediction Dashboard":
    # Runs the prediction_module.py logic
    run_prediction_app()

elif main_option == "Mutual Fund Market Analysis (MFTool)":
    # Runs the mftool_analysis.py logic
    run_mftool_app()