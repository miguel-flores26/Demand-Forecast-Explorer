"""Rolling-origin backtest: how would each method have performed in the past?

The idea
--------
Pretend it is some earlier month (the "origin"). Fit each method using only the data up to
that month, forecast the next ``horizon`` months, and compare with what actually happened.
Then move the origin forward and repeat. Averaging the errors over many origins gives a much
fairer picture than a single train/test split, which can be lucky or unlucky.

    history:  ████████████████████████████████████████
    origin 1: ███████████████████████|forecast→|
    origin 2: ██████████████████████████|forecast→|
    origin 3: █████████████████████████████|forecast→|
               (train on data up to the origin only)    (compare with actuals)

By default the origins are the most recent ones, so the evaluation reflects recent behavior.

Metrics
-------
MAE       mean absolute error, in the series' own units (here, millions of dollars)
MAPE      mean absolute percentage error
RMSE      root mean squared error; penalizes large misses more than MAE
coverage  share of actual values that fell inside the prediction interval. For an 80%
          interval, roughly 80% is what a well-calibrated method should achieve.

Caveat: forecasts from neighboring origins overlap heavily, so the many forecast-actual
pairs are far from independent. Treat these numbers as a careful comparison, not as a
statistical significance test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_explorer.models import MIN_HISTORY, MODELS, run_forecast

BASELINE = "seasonal_naive"

# Rule of thumb for the plain-language verdict: a method must cut the baseline's MAE by at
# least this share to count as beating it. Smaller differences are reported as a tie.
MEANINGFUL_IMPROVEMENT = 0.05

RESULT_COLUMNS = ["model", "origin", "h", "date", "actual", "forecast", "lower", "upper"]


def rolling_origin_backtest(
    series: pd.Series,
    horizon: int = 12,
    n_origins: int = 12,
    step: int = 3,
    level: float = 0.80,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Run every method from several past origins and collect forecasts next to actuals.

    Origins are spaced ``step`` months apart, ending at the latest month that still leaves
    ``horizon`` months of actual data to compare against.

    Returns one row per (model, origin, months-ahead), with columns:
    model, origin (last month of training data), h (months ahead), date, actual,
    forecast, lower, upper.
    """
    keys = list(models) if models is not None else list(MODELS)
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        raise KeyError(f"Unknown model(s) {unknown}. Choose from: {sorted(MODELS)}")
    if n_origins < 1 or step < 1:
        raise ValueError("n_origins and step must be at least 1.")

    n = len(series)
    last_train_end = n - horizon
    train_ends = [last_train_end - i * step for i in range(n_origins)][::-1]
    if train_ends[0] < MIN_HISTORY:
        needed = MIN_HISTORY + (n_origins - 1) * step + horizon
        raise ValueError(
            f"Not enough history for this backtest: need at least {needed} months "
            f"({MIN_HISTORY} to train + {(n_origins - 1) * step} to roll the origin + "
            f"{horizon} to compare), got {n}. Try fewer origins or a shorter horizon."
        )

    frames = []
    for end in train_ends:
        train = series.iloc[:end]  # only data up to the origin: no peeking ahead
        actual = series.iloc[end : end + horizon]
        for key in keys:
            fc = run_forecast(key, train, horizon, level)
            frames.append(
                pd.DataFrame(
                    {
                        "model": key,
                        "origin": train.index[-1],
                        "h": np.arange(1, horizon + 1),
                        "date": fc.index,
                        "actual": actual.to_numpy(dtype=float),
                        "forecast": fc["forecast"].to_numpy(),
                        "lower": fc["lower"].to_numpy(),
                        "upper": fc["upper"].to_numpy(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)[RESULT_COLUMNS]


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model error metrics from backtest results, in the registry's model order.

    Adds ``mae_vs_baseline``: the share by which a method's MAE is lower than the baseline's
    (positive = better than baseline, negative = worse), when the baseline was included.
    """
    err = results["forecast"] - results["actual"]
    scored = results.assign(
        abs_err=err.abs(),
        sq_err=err**2,
        ape=err.abs() / results["actual"].abs(),
        covered=(results["actual"] >= results["lower"]) & (results["actual"] <= results["upper"]),
    )
    summary = scored.groupby("model").agg(
        mae=("abs_err", "mean"),
        mape=("ape", "mean"),
        rmse=("sq_err", lambda x: float(np.sqrt(x.mean()))),
        coverage=("covered", "mean"),
        n_forecasts=("abs_err", "size"),
    )
    order = [k for k in MODELS if k in summary.index]
    summary = summary.loc[order]

    if BASELINE in summary.index:
        summary["mae_vs_baseline"] = 1 - summary["mae"] / summary.loc[BASELINE, "mae"]
    return summary


def verdict(summary: pd.DataFrame, level: float = 0.80) -> list[str]:
    """Plain-language findings from a summary table, one sentence per finding."""
    lines = []
    has_baseline = "mae_vs_baseline" in summary.columns

    for key, row in summary.iterrows():
        label = MODELS[key][0]
        if key != BASELINE and has_baseline:
            gain = row["mae_vs_baseline"]
            if gain >= MEANINGFUL_IMPROVEMENT:
                lines.append(
                    f"{label} beat the baseline: its average error was {gain:.0%} lower."
                )
            elif gain <= -MEANINGFUL_IMPROVEMENT:
                lines.append(
                    f"{label} did worse than the baseline: its average error was "
                    f"{-gain:.0%} higher."
                )
            else:
                lines.append(
                    f"{label} performed about the same as the baseline "
                    f"(average error within {MEANINGFUL_IMPROVEMENT:.0%})."
                )

        cov = row["coverage"]
        if cov < level - 0.10:
            note = "too narrow (overconfident)"
        elif cov > level + 0.10:
            note = "wider than needed (underconfident)"
        else:
            note = "roughly as wide as they should be"
        lines.append(
            f"{label}: its {level:.0%} intervals contained the actual value "
            f"{cov:.0%} of the time, so they were {note}."
        )
    return lines
