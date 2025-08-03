# streamlit_app.py

import streamlit as st
import pandas as pd

from wyckoff import run_backtest, get_available_tickers

st.set_page_config(
    page_title="Wyckoff Backtest",
    layout="wide"
)

# Sidebar
st.sidebar.header("📊 Backtest Settings")
tickers = get_available_tickers()
selected = st.sidebar.selectbox("Choose a stock:", tickers)

start_date = st.sidebar.date_input(
    "Start date", value=pd.to_datetime("2020-01-01")
)
end_date = st.sidebar.date_input(
    "End date", value=pd.to_datetime("2025-06-30")
)

if st.sidebar.button("▶️ Run Backtest"):
    with st.spinner(f"Running backtest on {selected}…"):
        equity_df, metrics = run_backtest(
            selected,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

    # Equity curve plot
    st.subheader(f"Wyckoff Strategy vs Buy-and-Hold — {selected}")
    st.line_chart(equity_df)

    # Metrics table
    st.subheader("Aggregate Results")
    df_metrics = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    st.table(df_metrics)
