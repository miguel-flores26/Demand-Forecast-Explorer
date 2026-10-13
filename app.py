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
from forecast_explorer.models import MAX_HORIZON, MIN_HISTORY, MODELS, run_forecast
from forecast_explorer.upload import (
    MAX_INTERPOLATED_MONTHS,
    UploadError,
    example_csv,
    guess_columns,
    prepare_series,
    read_csv_bytes,
)

REPO_URL = "https://github.com/miguel-flores26/demand-forecast-explorer"
EXAMPLE_SERIES = "MRTSSM4451USN"  # grocery stores, used for the downloadable example file
BACKTEST_STEP = 3
CONVERGENCE_MSG = "Maximum Likelihood optimization failed"

st.set_page_config(page_title="Demand Forecast Explorer", layout="wide")


# ---------- cached data and computation ----------


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_series(series_id: str):
    """Fresh data at most once a day; fall back to the saved copy if FRED is unreachable."""
    try:
        return load_series(series_id, refresh=True)
    except requests.RequestException:
        return load_series(series_id)


@st.cache_data(show_spinner=False, max_entries=50)
def get_forecasts(series, horizon: int, level: float, models: tuple[str, ...]):
    with warnings.catch_warnings():
        # SARIMA convergence warnings were investigated (see README); hide them from logs.
        warnings.filterwarnings("ignore", message=CONVERGENCE_MSG)
        return {key: run_forecast(key, series, horizon, level) for key in models}


@st.cache_data(show_spinner=False, max_entries=50)
def get_backtest(series, horizon: int, level: float, models: tuple[str, ...], n_origins: int):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=CONVERGENCE_MSG)
        results = bt.rolling_origin_backtest(
            series, horizon=horizon, n_origins=n_origins, step=BACKTEST_STEP,
            level=level, models=list(models),
        )
    return results, bt.summarize(results)


# ---------- header ----------

st.title("Demand Forecast Explorer")
st.markdown(
    "How much demand should you plan for, how uncertain is it, and does a more complex "
    "model actually beat a simple one? Pick a U.S. retail category or upload your own "
    "monthly sales, then compare three forecasting methods and see how each would have "
    "performed in the past."
)

# ---------- sidebar: data source ----------

st.sidebar.header("Data")
source = st.sidebar.radio(
    "Data source", ["Built-in retail data", "Upload your own CSV"], label_visibility="collapsed"
)

if source == "Built-in retail data":
    series_id = st.sidebar.selectbox(
        "Retail category", options=list(SERIES), format_func=SERIES.get
    )
    try:
        series = get_series(series_id)
    except Exception as exc:  # network or data problem: show it rather than crash
        st.error(f"Couldn't load data for this category right now. Details: {exc}")
        st.stop()
    y_title, err_title, unit = "Monthly sales ($ millions)", "Average error ($ millions)", "$M"
    file_stub = series_id
    caption = (
        f"**{SERIES[series_id]}**, monthly U.S. sales, not seasonally adjusted. "
        f"Latest data: {series.index[-1]:%B %Y} "
        f"(${series.iloc[-1] / 1000:,.1f} billion). Source: U.S. Census Bureau via FRED."
    )
    notes: list[str] = []

else:
    uploaded = st.sidebar.file_uploader("CSV file", type=["csv", "txt"])
    try:
        st.sidebar.download_button(
            "Download an example file",
            example_csv(get_series(EXAMPLE_SERIES)),
            file_name="example_grocery_sales.csv",
            mime="text/csv",
            help="Real monthly U.S. grocery store sales ($ millions), in the expected format.",
        )
    except Exception:
        pass  # the example is a convenience; skip it if FRED is unreachable

    if uploaded is None:
        st.info(
            "**Upload a CSV with one column of dates and one column of values.**\n\n"
            "- Monthly, weekly, or daily data all work (weekly and daily are added up into "
            "months).\n"
            f"- At least {MIN_HISTORY} months of history, so the models can learn the "
            "yearly pattern.\n"
            "- All monthly totals must be positive.\n"
            "- Dates like `2024-01`, `Jan 2024`, or `1/15/2024` and values like `$1,200` "
            "are fine.\n\n"
            "Not sure about the format? Download the example file in the sidebar."
        )
        st.stop()

    try:
        raw = read_csv_bytes(uploaded.getvalue())
    except UploadError as exc:
        st.error(str(exc))
        st.stop()

    guessed_date, guessed_value = guess_columns(raw)
    columns = list(raw.columns)
    date_col = st.sidebar.selectbox(
        "Date column", columns,
        index=columns.index(guessed_date) if guessed_date in columns else 0,
    )
    value_default = columns.index(guessed_value) if guessed_value in columns else 1
    value_col = st.sidebar.selectbox("Value column", columns, index=value_default)

    try:
        series, notes = prepare_series(raw, date_col, value_col)
    except UploadError as exc:
        st.error(str(exc))
        with st.expander("First rows of your file"):
            st.dataframe(raw.head(10), width="stretch")
        st.stop()

    y_title, err_title, unit = f"Monthly {value_col}", "Average error", None
    file_stub = "uploaded"
    caption = (
        f"**Your data: '{value_col}'**, {len(series)} months from "
        f"{series.index[0]:%B %Y} to {series.index[-1]:%B %Y}. Your file is used only to "
        "compute this page. It isn't written to disk, though results are kept in memory "
        "briefly to speed things up."
    )

# ---------- sidebar: settings ----------

st.sidebar.header("Settings")
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

st.caption(caption)
if notes:
    st.info("**Adjustments made to your data:**\n" + "\n".join(f"- {n}" for n in notes))

if not chosen:
    st.info("Select at least one forecasting method in the sidebar.")
    st.stop()

# Keep the registry order regardless of the order the user clicked them in.
models = tuple(k for k in MODELS if k in chosen)

# ---------- forecast ----------

st.subheader("Forecast")
with st.spinner("Fitting models..."):
    forecasts = get_forecasts(series, horizon, level, models)

st.plotly_chart(
    forecast_figure(history_window(series, years_shown), forecasts, level, y_title=y_title),
    width="stretch",
)

table = forecast_table(forecasts, level)
with st.expander("Forecast table"):
    st.dataframe(table.round(0), width="stretch")
    st.download_button(
        "Download as CSV",
        table.round(1).to_csv().encode(),
        file_name=f"{file_stub}_forecast.csv",
        mime="text/csv",
    )

# ---------- backtest ----------

st.subheader("Backtest: would these methods have been right?")
n_origins = bt.max_origins(len(series), horizon, step=BACKTEST_STEP)

if n_origins == 0:
    st.warning(
        f"Not enough history for a backtest with a {horizon}-month horizon: it needs at "
        f"least {MIN_HISTORY + horizon} months, and this data has {len(series)}. "
        "Try forecasting fewer months ahead."
    )
else:
    st.markdown(
        f"Each method is refit from {n_origins} past starting point(s), {BACKTEST_STEP} "
        "months apart, using only the data available at the time, and its "
        f"{horizon}-month forecasts are compared with what actually happened. The seasonal "
        'naive baseline ("same month last year") is always included for comparison.'
    )
    if n_origins < 12:
        st.caption(
            f"With only {n_origins} starting point(s), these results rest on less evidence "
            "than usual; treat them as a rough guide."
        )

    backtest_models = tuple(k for k in MODELS if k in models or k == bt.BASELINE)
    with st.spinner("Running backtest (the first run can take up to a minute)..."):
        results, summary = get_backtest(series, horizon, level, backtest_models, n_origins)

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
        st.dataframe(summary_for_display(summary, level, unit=unit), width="stretch")
    with col2:
        st.markdown("**How error grows with forecast distance**")
        st.plotly_chart(
            horizon_error_figure(error_by_horizon(results), y_title=err_title),
            width="stretch",
        )

# ---------- about ----------

with st.expander("About the data, methods, and limitations"):
    st.markdown(
        f"""
**Built-in data.** U.S. Census Bureau Monthly Retail Trade Survey, retrieved from FRED.
Values are monthly sales in millions of dollars, *not* seasonally adjusted, because modeling
seasonality is part of the task.

**Uploaded data.** Daily and weekly values are summed into months; incomplete first and last
months are dropped; up to {MAX_INTERPOLATED_MONTHS} missing months are filled by
straight-line interpolation. Every adjustment is listed above the forecast.

**Methods.**
- *Seasonal naive*: each month equals the same month one year earlier. The baseline.
- *ETS*: exponential smoothing with a damped trend and yearly seasonality, fitted on log values.
- *SARIMA*: seasonal ARIMA(0,1,1)(0,1,1)[12], fitted on log values.

**Limitations.**
- Forecasts assume past patterns continue; they cannot anticipate shocks.
- The built-in series are national totals. A single business's sales can behave very
  differently.
- Dollar sales mix price and volume, so growth partly reflects inflation.
- This is a portfolio project, not a production forecasting system or financial advice.

Code, tests, and full documentation: [{REPO_URL}]({REPO_URL})
"""
    )
