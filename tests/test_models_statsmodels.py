"""Tests for the ETS and SARIMA methods, which depend on statsmodels.

They check the shared output contract and that each model recovers a clear trend-and-season
pattern reasonably well. They are deliberately loose about accuracy: the goal is to catch a
broken model (wrong scale, shifted dates, inverted intervals), not to benchmark it.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from forecast_explorer import models  # noqa: E402
from tests.test_models import make_series  # noqa: E402

STATS_MODELS = ["ets", "sarima"]


@pytest.mark.parametrize("key", STATS_MODELS)
def test_output_contract(key):
    s = make_series(n_months=96, noise=0.01)
    fc = models.run_forecast(key, s, horizon=12)
    assert list(fc.columns) == ["forecast", "lower", "upper"]
    assert len(fc) == 12
    assert fc.index[0] == s.index[-1] + pd.offsets.MonthBegin(1)
    assert fc.notna().all().all()
    assert (fc["lower"] > 0).all()
    assert (fc["lower"] <= fc["forecast"]).all()
    assert (fc["forecast"] <= fc["upper"]).all()


@pytest.mark.parametrize("key", STATS_MODELS)
def test_recovers_clear_pattern_on_holdout(key):
    full = make_series(n_months=108, noise=0.01)
    train, test = full.iloc[:-12], full.iloc[-12:]
    fc = models.run_forecast(key, train, horizon=12)
    mape = np.mean(np.abs(fc["forecast"].to_numpy() - test.to_numpy()) / test.to_numpy())
    assert mape < 0.10, f"{key} MAPE {mape:.1%} on an easy synthetic series"


@pytest.mark.parametrize("key", STATS_MODELS)
def test_intervals_widen_with_horizon(key):
    s = make_series(n_months=96, noise=0.02)
    fc = models.run_forecast(key, s, horizon=24)
    width = (fc["upper"] - fc["lower"]).to_numpy()
    assert width[-1] > width[0]


@pytest.mark.parametrize("key", STATS_MODELS)
def test_validation_applies(key):
    with pytest.raises(ValueError, match="at least 36 months"):
        models.run_forecast(key, make_series(n_months=30), 6)
