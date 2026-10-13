"""Smoke tests for the Plotly charts: they build, with the expected traces."""

import pytest

pytest.importorskip("plotly")

from forecast_explorer import backtest as bt  # noqa: E402
from forecast_explorer.app_helpers import error_by_horizon  # noqa: E402
from forecast_explorer.charts import forecast_figure, horizon_error_figure  # noqa: E402
from forecast_explorer.models import seasonal_naive  # noqa: E402
from tests.test_models import make_series  # noqa: E402


def test_forecast_figure_traces():
    s = make_series()
    fig = forecast_figure(s, {"seasonal_naive": seasonal_naive(s, 6)}, level=0.80)
    names = [t.name for t in fig.data if t.showlegend is not False]
    assert names == ["Actual", "Seasonal naive"]
    # actual + band upper + band lower + connector + forecast line
    assert len(fig.data) == 5
    connector, forecast_line = fig.data[-2], fig.data[-1]
    assert connector.x[0] == s.index[-1]  # connector starts at the last actual
    assert len(forecast_line.x) == 6


def test_horizon_error_figure_traces():
    s = make_series(n_months=96, noise=0.02)
    res = bt.rolling_origin_backtest(s, horizon=6, n_origins=3, step=2,
                                     models=["seasonal_naive"])
    fig = horizon_error_figure(error_by_horizon(res))
    assert [t.name for t in fig.data] == ["Seasonal naive"]
    assert len(fig.data[0].x) == 6
