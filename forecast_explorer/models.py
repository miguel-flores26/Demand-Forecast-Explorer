"""Forecasting methods.

Every method takes the same inputs and returns the same output shape, so the app and the
backtest can treat them interchangeably:

    inputs:  a monthly pandas Series (month-start index, positive values),
             a horizon (number of months ahead), and an interval level (e.g. 0.80)
    output:  a DataFrame indexed by the forecast months, with columns
             "forecast", "lower", "upper"

Methods
-------
seasonal_naive
    "Same month last year." The baseline every other method has to beat.
ets
    Exponential smoothing with a damped additive trend and additive seasonality,
    fitted to log(sales).
sarima
    Seasonal ARIMA(0,1,1)(0,1,1)[12], the classic "airline" model, fitted to log(sales).

Why log scale for ETS and SARIMA: retail seasonality tends to grow with the level of sales
(a December spike is bigger in dollars when overall sales are bigger). On the log scale that
pattern becomes additive, and back-transforming keeps forecasts and intervals positive.
One consequence: exponentiating a log-scale forecast gives the forecast *median*, not the
mean. For data like these the two are close, but they are not identical.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

SEASON_LENGTH = 12  # monthly data, yearly seasonality
MIN_HISTORY = 36  # months; seasonal models need several full years to estimate the pattern
MAX_HORIZON = 24  # months


def _validate(series: pd.Series, horizon: int, level: float) -> None:
    """Raise ValueError with a readable message if inputs can't be forecast sensibly."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex.")
    if series.index.freqstr != "MS":
        raise ValueError("Series must be monthly with month-start dates (freq='MS').")
    if series.isna().any():
        raise ValueError("Series contains missing values.")
    if len(series) < MIN_HISTORY:
        raise ValueError(
            f"Need at least {MIN_HISTORY} months of history; got {len(series)}."
        )
    if (series <= 0).any():
        raise ValueError("All values must be positive (models are fitted on the log scale).")
    if not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"Horizon must be a whole number from 1 to {MAX_HORIZON}.")
    if not 0 < level < 1:
        raise ValueError("Interval level must be between 0 and 1, e.g. 0.80.")


def _future_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    start = series.index[-1] + pd.offsets.MonthBegin(1)
    return pd.date_range(start, periods=horizon, freq="MS", name="date")


def _frame(index: pd.DatetimeIndex, forecast, lower, upper) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast": np.asarray(forecast, dtype=float),
            "lower": np.asarray(lower, dtype=float),
            "upper": np.asarray(upper, dtype=float),
        },
        index=index,
    )


def seasonal_naive(series: pd.Series, horizon: int, level: float = 0.80) -> pd.DataFrame:
    """Forecast each future month as the value from the same month one year earlier.

    Prediction intervals follow the standard seasonal-naive formula (Hyndman &
    Athanasopoulos, *Forecasting: Principles and Practice*): the residual standard deviation,
    scaled by sqrt(k + 1), where k is the number of complete years ahead. So the intervals
    widen once per year of horizon, not every month.
    """
    _validate(series, horizon, level)
    m = SEASON_LENGTH
    y = series.to_numpy(dtype=float)
    n = len(y)

    steps = np.arange(1, horizon + 1)
    forecast = y[n - m + (steps - 1) % m]

    residuals = y[m:] - y[:-m]  # this year's value minus same month last year
    sigma = np.sqrt(np.mean(residuals**2))
    k = (steps - 1) // m
    z = NormalDist().inv_cdf(0.5 + level / 2)
    half_width = z * sigma * np.sqrt(k + 1)

    return _frame(_future_index(series, horizon), forecast, forecast - half_width,
                  forecast + half_width)


def ets(series: pd.Series, horizon: int, level: float = 0.80) -> pd.DataFrame:
    """Exponential smoothing: ETS(A, Ad, A) on log(sales)."""
    _validate(series, horizon, level)
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    n = len(series)
    model = ETSModel(
        np.log(series),
        error="add",
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=SEASON_LENGTH,
    )
    result = model.fit(disp=False)
    pred = result.get_prediction(start=n, end=n + horizon - 1)
    frame = pred.summary_frame(alpha=1 - level)

    return _frame(
        _future_index(series, horizon),
        np.exp(frame["mean"]),
        np.exp(frame["pi_lower"]),
        np.exp(frame["pi_upper"]),
    )


def sarima(series: pd.Series, horizon: int, level: float = 0.80) -> pd.DataFrame:
    """Seasonal ARIMA(0,1,1)(0,1,1)[12] on log(sales)."""
    _validate(series, horizon, level)
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        np.log(series),
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, SEASON_LENGTH),
    )
    result = model.fit(disp=False)
    fc = result.get_forecast(steps=horizon)
    intervals = fc.conf_int(alpha=1 - level)

    return _frame(
        _future_index(series, horizon),
        np.exp(fc.predicted_mean),
        np.exp(intervals.iloc[:, 0]),
        np.exp(intervals.iloc[:, 1]),
    )


# Registry used by the app and the backtest: key -> (display label, function)
MODELS = {
    "seasonal_naive": ("Seasonal naive (same month last year)", seasonal_naive),
    "ets": ("Exponential smoothing (ETS)", ets),
    "sarima": ("Seasonal ARIMA", sarima),
}


def run_forecast(
    model_key: str, series: pd.Series, horizon: int, level: float = 0.80
) -> pd.DataFrame:
    """Run one forecasting method by its registry key."""
    if model_key not in MODELS:
        raise KeyError(f"Unknown model {model_key!r}. Choose from: {sorted(MODELS)}")
    _, fn = MODELS[model_key]
    return fn(series, horizon, level)
