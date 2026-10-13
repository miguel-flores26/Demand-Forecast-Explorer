"""Data preparation for the web app.

Everything here is plain pandas, kept out of app.py so it can be unit-tested without
running Streamlit.
"""

from __future__ import annotations

import pandas as pd

from forecast_explorer.backtest import BASELINE
from forecast_explorer.models import MODELS

# Short names for legends and tables (full names live in models.MODELS).
SHORT_LABELS = {
    "seasonal_naive": "Seasonal naive",
    "ets": "ETS",
    "sarima": "SARIMA",
}

# One fixed color per method, so a method keeps its color whatever else is selected.
# First three slots of a palette validated for colorblind-safe separation.
MODEL_COLORS = {
    "seasonal_naive": "#2a78d6",  # blue
    "ets": "#eb6834",  # orange
    "sarima": "#1baf7a",  # aqua
}
ACTUAL_COLOR = "#52514e"  # neutral gray for the historical data


def history_window(series: pd.Series, years: int) -> pd.Series:
    """The most recent ``years`` years of a monthly series (or all of it if shorter)."""
    return series.iloc[-12 * years :]


def forecast_table(forecasts: dict[str, pd.DataFrame], level: float) -> pd.DataFrame:
    """Combine per-method forecasts into one wide table for display and download.

    Columns look like "ETS forecast", "ETS low (80%)", "ETS high (80%)".
    """
    pct = f"{level:.0%}"
    parts = []
    for key, fc in forecasts.items():
        name = SHORT_LABELS[key]
        parts.append(
            fc.rename(
                columns={
                    "forecast": f"{name} forecast",
                    "lower": f"{name} low ({pct})",
                    "upper": f"{name} high ({pct})",
                }
            )
        )
    table = pd.concat(parts, axis=1)
    table.index = table.index.strftime("%b %Y")
    table.index.name = "Month"
    return table


def summary_for_display(
    summary: pd.DataFrame, level: float, unit: str | None = "$M"
) -> pd.DataFrame:
    """Turn backtest.summarize() output into a readable table of formatted strings.

    ``unit`` labels the average-error column, e.g. "$M" for the built-in series. Pass None
    for uploaded data, whose units the app doesn't know.
    """
    error_col = f"Avg error ({unit})" if unit else "Avg error"
    rows = []
    for key, row in summary.iterrows():
        if key == BASELINE or "mae_vs_baseline" not in summary.columns:
            vs = "(baseline)" if key == BASELINE else "n/a"
        else:
            gain = row["mae_vs_baseline"]
            vs = f"{gain:.0%} lower" if gain >= 0 else f"{-gain:.0%} higher"
        rows.append(
            {
                "Method": SHORT_LABELS[key],
                error_col: f"{row['mae']:,.0f}",
                "Avg % error": f"{row['mape']:.1%}",
                "Error vs. baseline": vs,
                f"{level:.0%} interval coverage": f"{row['coverage']:.0%}",
            }
        )
    return pd.DataFrame(rows).set_index("Method")


def error_by_horizon(results: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error for each months-ahead step, one column per method.

    Shows how quickly each method's accuracy fades the further ahead it forecasts.
    """
    abs_err = (results["forecast"] - results["actual"]).abs()
    table = results.assign(abs_err=abs_err).pivot_table(
        index="h", columns="model", values="abs_err", aggfunc="mean"
    )
    order = [k for k in MODELS if k in table.columns]
    return table[order]
