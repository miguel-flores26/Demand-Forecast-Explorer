"""Plotly charts for the web app."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from forecast_explorer.app_helpers import (
    ACTUAL_COLOR,
    MODEL_COLORS,
    SHORT_LABELS,
)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def forecast_figure(
    history: pd.Series,
    forecasts: dict[str, pd.DataFrame],
    level: float,
    y_title: str = "Monthly sales ($ millions)",
) -> go.Figure:
    """Recent actuals plus each method's forecast line and shaded prediction interval."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.to_numpy(),
            name="Actual",
            mode="lines",
            line={"color": ACTUAL_COLOR, "width": 2},
            hovertemplate="%{y:,.0f}<extra>Actual</extra>",
        )
    )

    last_date, last_value = history.index[-1], float(history.iloc[-1])
    for key, fc in forecasts.items():
        color, name = MODEL_COLORS[key], SHORT_LABELS[key]
        # Interval band: upper edge, then lower edge filled up to it.
        fig.add_trace(
            go.Scatter(
                x=fc.index, y=fc["upper"], mode="lines", line={"width": 0},
                showlegend=False, hoverinfo="skip", legendgroup=key,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc.index, y=fc["lower"], mode="lines", line={"width": 0},
                fill="tonexty", fillcolor=_rgba(color, 0.15),
                showlegend=False, hoverinfo="skip", legendgroup=key,
            )
        )
        # Short connector from the last actual to the first forecast, so the forecast
        # visibly continues the data. Kept separate so it doesn't show up in the hover.
        fig.add_trace(
            go.Scatter(
                x=[last_date, fc.index[0]],
                y=[last_value, fc["forecast"].iloc[0]],
                mode="lines", line={"color": color, "width": 2, "dash": "dot"},
                showlegend=False, hoverinfo="skip", legendgroup=key,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc.index,
                y=fc["forecast"],
                name=name,
                mode="lines",
                line={"color": color, "width": 2},
                legendgroup=key,
                customdata=fc[["lower", "upper"]].to_numpy(),
                hovertemplate=(
                    "%{y:,.0f} (%{customdata[0]:,.0f} to %{customdata[1]:,.0f})"
                    f"<extra>{name}</extra>"
                ),
            )
        )

    fig.update_layout(
        hovermode="x unified",
        hoverdistance=5,
        yaxis_title=y_title,
        yaxis_tickformat=",",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        height=440,
    )
    fig.add_annotation(
        text=f"Shaded bands: {level:.0%} prediction intervals",
        xref="paper", yref="paper", x=0, y=-0.12, showarrow=False,
        font={"size": 12}, xanchor="left",
    )
    return fig


def horizon_error_figure(
    err_by_h: pd.DataFrame, y_title: str = "Average error ($ millions)"
) -> go.Figure:
    """Average backtest error at each months-ahead step, one line per method."""
    fig = go.Figure()
    for key in err_by_h.columns:
        fig.add_trace(
            go.Scatter(
                x=err_by_h.index,
                y=err_by_h[key],
                name=SHORT_LABELS[key],
                mode="lines+markers",
                line={"color": MODEL_COLORS[key], "width": 2},
                marker={"size": 8},
                hovertemplate="%{y:,.0f}" + f"<extra>{SHORT_LABELS[key]}</extra>",
            )
        )
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Months ahead",
        yaxis_title=y_title,
        yaxis_tickformat=",",
        yaxis_rangemode="tozero",
        xaxis={"dtick": 1 if len(err_by_h) <= 12 else 3},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        height=360,
    )
    return fig
