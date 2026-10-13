"""Demand Forecast Explorer: Streamlit web app.

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import warnings

import requests
import streamlit as st

from forecast_explorer import backtest as bt
from forecast_explorer.app_helpers import (
    SHORT_LABELS,
    error_by_horizon,
    forecast_table,
    history_window,
    summary_for_display,
)
from forecast_explorer.charts import forecast_figure, horizon_error_figure
from forecast_explorer.data import SERIES, load_series
from forecast_explorer.models import MAX_HORIZON, MODELS, run_forecast

REPO_URL = "https://github.com/miguel-flores26/demand-forecast-explorer"

# statsmodels sometimes reports optimizer convergence warnings during SARIMA fits. They are
# investigated and documented in the README; in the app they would only be log noise.
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed")

st.set_page_config(page_title="Demand Forecast Explorer", layout="wide")


# ---------- cached data and computation ----------


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_series(series_id: str):
    """Fresh data at most once a day; fall back to the saved copy if FRED is unreachable."""
    try:
        return load_series(series_id, refresh=True)
    except requests.RequestException:
        return load_series(series_id)


@st.cache_data(show_spinner=False)
def get_forecasts(series_id: str, data_version: str, horizon: int, level: float,
                  models: tuple[str, ...]):
    series = get_series(series_id)
    return {key: run_forecast(key, series, horizon, level) for key in models}


@st.cache_data(show_spinner=False)
def get_backtest(series_id: str, data_version: str, horizon: int, level: float,
                 models: tuple[str, ...]):
    series = get_series(series_id)
    results = bt.rolling_origin_backtest(series, horizon=horizon, level=level,
                                         models=list(models))
    return results, bt.summarize(results)


# ---------- sidebar controls ----------

st.sidebar.header("Settings")
series_id = st.sidebar.selectbox(
    "Retail category", options=list(SERIES), format_func=SERIES.get
)
chosen = st.sidebar.multiselect(
    "Forecasting methods",
    options=list(MODELS),
    default=list(MODELS),
    format_func=lambda k: SHORT_LABELS[k],
)
horizon = st.sidebar.slider("Months to forecast", 3, MAX_HORIZON, 12)
level = st.sidebar.radio(
    "Prediction interval", [0.80, 0.95], format_func=lambda x: f"{x:.0%}", horizontal=True
)
years_shown = st.sidebar.slider("Years of history to show", 3, 15, 6)

# ---------- header ----------

st.title("Demand Forecast Explorer")
st.markdown(
    "How much demand should you plan for, how uncertain is it, and does a more complex "
    "model actually beat a simple one? Pick a U.S. retail category to compare three "
    "forecasting methods and see how each would have performed in the past."
)

if not chosen:
    st.info("Select at least one forecasting method in the sidebar.")
    st.stop()

# Keep the registry order regardless of the order the user clicked them in.
models = tuple(k for k in MODELS if k in chosen)

try:
    series = get_series(series_id)
except Exception as exc:  # network or data problem: show it rather than crash
    st.error(f"Couldn't load data for this category right now. Details: {exc}")
    st.stop()

data_version = str(series.index[-1].date())
st.caption(
    f"**{SERIES[series_id]}**, monthly U.S. sales, not seasonally adjusted. "
    f"Latest data: {series.index[-1]:%B %Y} "
    f"(${series.iloc[-1] / 1000:,.1f} billion). Source: U.S. Census Bureau via FRED."
)

# ---------- forecast ----------

st.subheader("Forecast")
with st.spinner("Fitting models..."):
    forecasts = get_forecasts(series_id, data_version, horizon, level, models)

st.plotly_chart(
    forecast_figure(history_window(series, years_shown), forecasts, level),
    width="stretch",
)

table = forecast_table(forecasts, level)
with st.expander("Forecast table"):
    st.dataframe(table.round(0), width="stretch")
    st.download_button(
        "Download as CSV",
        table.round(1).to_csv().encode(),
        file_name=f"{series_id}_forecast.csv",
        mime="text/csv",
    )

# ---------- backtest ----------

st.subheader("Backtest: would these methods have been right?")
st.markdown(
    "Each method is refit from 12 past starting points, 3 months apart, using only the data "
    f"available at the time, and its {horizon}-month forecasts are compared with what "
    "actually happened. The seasonal naive baseline (\"same month last year\") is always "
    "included for comparison."
)

backtest_models = tuple(k for k in MODELS if k in models or k == bt.BASELINE)
try:
    with st.spinner("Running backtest (the first run for a category can take up to a minute)..."):
        results, summary = get_backtest(series_id, data_version, horizon, level, backtest_models)
except ValueError as exc:
    st.warning(f"Backtest not available: {exc}")
else:
    origins = sorted(results["origin"].unique())
    st.markdown("**Findings**")
    for line in bt.verdict(summary, level):
        st.markdown(f"- {line}")
    st.caption(
        f"Starting points from {origins[0]:%b %Y} to {origins[-1]:%b %Y}. "
        "A method must cut average error by at least 5% to count as beating the baseline. "
        "Overlapping test windows mean these results are a careful comparison, "
        "not a statistical significance test."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Accuracy summary**")
        st.dataframe(summary_for_display(summary, level), width="stretch")
    with col2:
        st.markdown("**How error grows with forecast distance**")
        st.plotly_chart(horizon_error_figure(error_by_horizon(results)), width="stretch")

# ---------- about ----------

with st.expander("About the data, methods, and limitations"):
    st.markdown(
        f"""
**Data.** U.S. Census Bureau Monthly Retail Trade Survey, retrieved from FRED. Values are
monthly sales in millions of dollars, *not* seasonally adjusted, because modeling seasonality
is part of the task.

**Methods.**
- *Seasonal naive*: each month equals the same month one year earlier. The baseline.
- *ETS*: exponential smoothing with a damped trend and yearly seasonality, fitted on log(sales).
- *SARIMA*: seasonal ARIMA(0,1,1)(0,1,1)[12], fitted on log(sales).

**Limitations.**
- Forecasts assume past patterns continue; they cannot anticipate shocks.
- These are national totals. A single business's sales can behave very differently.
- Dollar sales mix price and volume, so growth partly reflects inflation.
- This is a portfolio project, not a production forecasting system or financial advice.

Code, tests, and full documentation: [{REPO_URL}]({REPO_URL})
"""
    )
