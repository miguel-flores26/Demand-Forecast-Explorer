"""Tests for the seasonal naive baseline and the shared input checks.

These need only numpy and pandas.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_explorer import models


def make_series(n_months=60, start="2018-01-01", noise=0.0, seed=0):
    """A synthetic monthly series: upward trend times a yearly pattern, optionally noisy."""
    idx = pd.date_range(start, periods=n_months, freq="MS")
    t = np.arange(n_months)
    seasonal = 1 + 0.2 * np.sin(2 * np.pi * t / 12)
    values = (1000 + 5 * t) * seasonal
    if noise:
        values = values * (1 + np.random.default_rng(seed).normal(0, noise, n_months))
    return pd.Series(values, index=idx)


# ---------- seasonal naive ----------


def test_seasonal_naive_repeats_last_year():
    s = make_series()
    fc = models.seasonal_naive(s, horizon=12)
    np.testing.assert_allclose(fc["forecast"].to_numpy(), s.iloc[-12:].to_numpy())


def test_seasonal_naive_wraps_for_horizon_beyond_a_year():
    s = make_series()
    fc = models.seasonal_naive(s, horizon=24)
    # Months 13-24 repeat the same last-year values again.
    np.testing.assert_allclose(fc["forecast"].iloc[12:].to_numpy(), s.iloc[-12:].to_numpy())


def test_seasonal_naive_intervals_widen_once_per_year():
    s = make_series(noise=0.02)
    fc = models.seasonal_naive(s, horizon=24)
    width = (fc["upper"] - fc["lower"]).to_numpy()
    assert np.allclose(width[:12], width[0])  # constant within year 1
    np.testing.assert_allclose(width[12], width[0] * np.sqrt(2))  # sqrt(k+1), k=1


def test_seasonal_naive_wider_level_gives_wider_interval():
    s = make_series(noise=0.02)
    w80 = models.seasonal_naive(s, 6, level=0.80).eval("upper - lower")
    w95 = models.seasonal_naive(s, 6, level=0.95).eval("upper - lower")
    assert (w95 > w80).all()


# ---------- output shape (shared contract) ----------


def test_output_index_starts_month_after_history():
    s = make_series()
    fc = models.seasonal_naive(s, horizon=6)
    assert list(fc.columns) == ["forecast", "lower", "upper"]
    assert fc.index[0] == s.index[-1] + pd.offsets.MonthBegin(1)
    assert fc.index.freqstr == "MS"
    assert len(fc) == 6


# ---------- input validation ----------


def test_rejects_short_history():
    with pytest.raises(ValueError, match="at least 36 months"):
        models.seasonal_naive(make_series(n_months=30), 6)


def test_rejects_non_positive_values():
    s = make_series()
    s.iloc[5] = 0
    with pytest.raises(ValueError, match="positive"):
        models.seasonal_naive(s, 6)


def test_rejects_missing_values():
    s = make_series()
    s.iloc[5] = np.nan
    with pytest.raises(ValueError, match="missing"):
        models.seasonal_naive(s, 6)


def test_rejects_non_monthly_index():
    s = make_series()
    s.index = pd.date_range("2018-01-01", periods=len(s), freq="D")
    with pytest.raises(ValueError, match="monthly"):
        models.seasonal_naive(s, 6)


@pytest.mark.parametrize("horizon", [0, 25, 2.5])
def test_rejects_bad_horizon(horizon):
    with pytest.raises(ValueError, match="Horizon"):
        models.seasonal_naive(make_series(), horizon)


@pytest.mark.parametrize("level", [0, 1, 80])
def test_rejects_bad_level(level):
    with pytest.raises(ValueError, match="level"):
        models.seasonal_naive(make_series(), 6, level=level)


def test_run_forecast_unknown_model():
    with pytest.raises(KeyError, match="Unknown model"):
        models.run_forecast("prophet", make_series(), 6)


def test_run_forecast_dispatches():
    s = make_series()
    pd.testing.assert_frame_equal(
        models.run_forecast("seasonal_naive", s, 6), models.seasonal_naive(s, 6)
    )
