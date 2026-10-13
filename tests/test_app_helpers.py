"""Tests for the app's data-preparation helpers (no Streamlit needed)."""

import numpy as np
import pandas as pd
import pytest

from forecast_explorer import app_helpers as ah
from forecast_explorer import backtest as bt
from forecast_explorer.models import MODELS, seasonal_naive
from tests.test_models import make_series


def test_every_model_has_a_label_and_color():
    assert set(ah.SHORT_LABELS) == set(MODELS)
    assert set(ah.MODEL_COLORS) == set(MODELS)
    assert len(set(ah.MODEL_COLORS.values())) == len(MODELS)  # no shared colors


def test_history_window():
    s = make_series(n_months=120)
    assert len(ah.history_window(s, 3)) == 36
    assert ah.history_window(s, 3).index[-1] == s.index[-1]
    assert len(ah.history_window(s, 50)) == 120  # shorter series: all of it


def test_forecast_table_columns_and_index():
    s = make_series()
    table = ah.forecast_table({"seasonal_naive": seasonal_naive(s, 3)}, level=0.80)
    assert list(table.columns) == [
        "Seasonal naive forecast",
        "Seasonal naive low (80%)",
        "Seasonal naive high (80%)",
    ]
    assert table.index[0] == (s.index[-1] + pd.offsets.MonthBegin(1)).strftime("%b %Y")
    assert table.index.name == "Month"


def _summary():
    return pd.DataFrame(
        {
            "mae": [1672.0, 1131.6, 2000.0],
            "mape": [0.022, 0.015, 0.03],
            "rmse": [1.0, 1.0, 1.0],
            "coverage": [0.875, 0.882, 0.5],
            "n_forecasts": [144, 144, 144],
            "mae_vs_baseline": [0.0, 0.323, -0.196],
        },
        index=["seasonal_naive", "ets", "sarima"],
    )


def test_summary_for_display_formats():
    table = ah.summary_for_display(_summary(), level=0.80)
    assert table.loc["Seasonal naive", "Error vs. baseline"] == "(baseline)"
    assert table.loc["ETS", "Error vs. baseline"] == "32% lower"
    assert table.loc["SARIMA", "Error vs. baseline"] == "20% higher"
    assert table.loc["ETS", "Avg error ($M)"] == "1,132"
    assert table.loc["ETS", "Avg % error"] == "1.5%"
    assert table.loc["SARIMA", "80% interval coverage"] == "50%"


def test_error_by_horizon_matches_manual_mean():
    s = make_series(n_months=96, noise=0.02)
    res = bt.rolling_origin_backtest(s, horizon=6, n_origins=3, step=2,
                                     models=["seasonal_naive"])
    table = ah.error_by_horizon(res)
    assert list(table.index) == [1, 2, 3, 4, 5, 6]
    h1 = res[res["h"] == 1]
    expected = np.mean(np.abs(h1["forecast"] - h1["actual"]))
    assert table.loc[1, "seasonal_naive"] == pytest.approx(expected)
