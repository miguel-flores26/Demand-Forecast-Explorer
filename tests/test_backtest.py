"""Tests for the rolling-origin backtest.

These use only the seasonal naive method, whose forecasts are easy to predict by hand, so
the tests can check the backtest's bookkeeping exactly: which data each origin sees, how
forecasts line up with actuals, and how the metrics are computed.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_explorer import backtest as bt
from forecast_explorer.models import MIN_HISTORY
from tests.test_models import make_series

NAIVE = ["seasonal_naive"]


def test_result_shape_and_columns():
    s = make_series(n_months=96)
    res = bt.rolling_origin_backtest(s, horizon=12, n_origins=4, step=3, models=NAIVE)
    assert list(res.columns) == bt.RESULT_COLUMNS
    assert len(res) == 4 * 12


def test_origins_are_spaced_and_end_at_latest_possible_month():
    s = make_series(n_months=96)
    res = bt.rolling_origin_backtest(s, horizon=12, n_origins=4, step=3, models=NAIVE)
    origins = sorted(res["origin"].unique())
    assert origins[-1] == s.index[-13]  # leaves exactly 12 months to compare against
    gaps = np.diff([pd.Timestamp(o).to_period("M").ordinal for o in origins])
    assert (gaps == 3).all()


def test_actuals_line_up_with_dates():
    s = make_series(n_months=96, noise=0.02)
    res = bt.rolling_origin_backtest(s, horizon=6, n_origins=3, step=2, models=NAIVE)
    np.testing.assert_allclose(res["actual"].to_numpy(), s.loc[res["date"]].to_numpy())


def test_forecast_dates_follow_their_origin():
    s = make_series(n_months=96)
    res = bt.rolling_origin_backtest(s, horizon=6, n_origins=3, step=2, models=NAIVE)
    for _, row in res.iterrows():
        assert row["date"] == row["origin"] + pd.DateOffset(months=int(row["h"]))


def test_no_peeking_at_future_data():
    # Seasonal naive forecasts each month from the same month a year earlier. If the
    # backtest trains only on data up to the origin, every forecast must equal a value
    # from on or before the origin (for horizons up to 12 months).
    s = make_series(n_months=96, noise=0.02)
    res = bt.rolling_origin_backtest(s, horizon=12, n_origins=4, step=3, models=NAIVE)
    one_year_earlier = res["date"] - pd.DateOffset(years=1)
    assert (one_year_earlier <= res["origin"]).all()
    np.testing.assert_allclose(res["forecast"].to_numpy(), s.loc[one_year_earlier].to_numpy())


def test_too_little_history_raises_helpful_error():
    s = make_series(n_months=MIN_HISTORY + 12)
    with pytest.raises(ValueError, match="Not enough history"):
        bt.rolling_origin_backtest(s, horizon=12, n_origins=4, step=3, models=NAIVE)


def test_minimum_history_exactly_enough():
    n_origins, step, horizon = 4, 3, 12
    s = make_series(n_months=MIN_HISTORY + (n_origins - 1) * step + horizon)
    res = bt.rolling_origin_backtest(s, horizon, n_origins, step, models=NAIVE)
    assert res["origin"].nunique() == n_origins


@pytest.mark.parametrize(
    "n_months, expected",
    [
        (MIN_HISTORY + 12 - 1, 0),  # one month short of a single origin
        (MIN_HISTORY + 12, 1),
        (MIN_HISTORY + 12 + 3 * 4, 5),
        (400, 12),  # capped at the number wanted
    ],
)
def test_max_origins(n_months, expected):
    assert bt.max_origins(n_months, horizon=12, step=3) == expected


def test_max_origins_is_always_runnable():
    s = make_series(n_months=MIN_HISTORY + 12 + 7)
    k = bt.max_origins(len(s), horizon=12, step=3)
    res = bt.rolling_origin_backtest(s, horizon=12, n_origins=k, step=3, models=NAIVE)
    assert res["origin"].nunique() == k


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="Unknown model"):
        bt.rolling_origin_backtest(make_series(n_months=96), models=["magic"])


# ---------- summarize ----------


def _results(model, actual, forecast, lower, upper):
    n = len(actual)
    return pd.DataFrame(
        {
            "model": model,
            "origin": pd.Timestamp("2020-01-01"),
            "h": np.arange(1, n + 1),
            "date": pd.date_range("2020-02-01", periods=n, freq="MS"),
            "actual": actual,
            "forecast": forecast,
            "lower": lower,
            "upper": upper,
        }
    )


def test_summarize_metrics_by_hand():
    res = _results(
        "seasonal_naive",
        actual=[100.0, 200.0],
        forecast=[110.0, 180.0],  # errors +10, -20
        lower=[90.0, 190.0],  # second actual (200) is inside [190, 210]
        upper=[105.0, 210.0],  # first actual (100) is inside [90, 105]
    )
    row = bt.summarize(res).loc["seasonal_naive"]
    assert row["mae"] == pytest.approx(15.0)
    assert row["mape"] == pytest.approx((0.10 + 0.10) / 2)
    assert row["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert row["coverage"] == pytest.approx(1.0)
    assert row["n_forecasts"] == 2


def test_summarize_compares_to_baseline():
    base = _results("seasonal_naive", [100.0], [120.0], [0.0], [999.0])  # MAE 20
    other = _results("ets", [100.0], [110.0], [0.0], [999.0])  # MAE 10
    summary = bt.summarize(pd.concat([base, other]))
    assert list(summary.index) == ["seasonal_naive", "ets"]  # registry order
    assert summary.loc["ets", "mae_vs_baseline"] == pytest.approx(0.5)
    assert summary.loc["seasonal_naive", "mae_vs_baseline"] == pytest.approx(0.0)


# ---------- verdict ----------


def _summary(gain, coverage):
    return pd.DataFrame(
        {"coverage": [0.80, coverage], "mae_vs_baseline": [0.0, gain]},
        index=["seasonal_naive", "ets"],
    )


@pytest.mark.parametrize(
    "gain, phrase",
    [(0.20, "beat the baseline"), (0.02, "about the same"), (-0.20, "did worse")],
)
def test_verdict_comparison_wording(gain, phrase):
    lines = bt.verdict(_summary(gain, 0.80))
    assert any(phrase in line for line in lines)


@pytest.mark.parametrize(
    "coverage, phrase",
    [(0.50, "too narrow"), (0.80, "roughly as wide"), (0.99, "wider than needed")],
)
def test_verdict_coverage_wording(coverage, phrase):
    lines = bt.verdict(_summary(0.0, coverage), level=0.80)
    ets_lines = [line for line in lines if line.startswith("Exponential smoothing (ETS):")]
    assert phrase in ets_lines[0]


def test_verdict_does_not_compare_baseline_to_itself():
    lines = bt.verdict(_summary(0.20, 0.80))
    assert not any("Seasonal naive" in line and "baseline" in line for line in lines)
