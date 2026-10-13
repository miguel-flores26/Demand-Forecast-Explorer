"""Tests for CSV upload handling, one test per kind of messy file a visitor might upload."""

import numpy as np
import pandas as pd
import pytest

from forecast_explorer.models import MIN_HISTORY
from forecast_explorer.upload import (
    UploadError,
    example_csv,
    guess_columns,
    parse_numbers,
    prepare_series,
    read_csv_bytes,
)


def monthly_csv(n=48, day_of_month=1, date_fmt="%Y-%m", sep=",", value_fmt="{:.0f}",
                header=("month", "sales")):
    dates = pd.date_range("2020-01-01", periods=n, freq="MS") + pd.Timedelta(
        days=day_of_month - 1
    )
    vals = 1000 + 10 * np.arange(n) + 100 * np.sin(2 * np.pi * np.arange(n) / 12)
    lines = [sep.join(header)]
    lines += [
        f"{d.strftime(date_fmt)}{sep}{value_fmt.format(v)}"
        for d, v in zip(dates, vals, strict=True)
    ]
    return "\n".join(lines).encode()


def load(data: bytes):
    df = read_csv_bytes(data)
    date_col, value_col = guess_columns(df)
    return prepare_series(df, date_col, value_col)


# ---------- the happy path ----------


def test_clean_monthly_file():
    series, notes = load(monthly_csv())
    assert len(series) == 48
    assert series.index.freqstr == "MS"
    assert series.dtype == "float64"
    assert notes == []


@pytest.mark.parametrize("fmt", ["%Y-%m", "%Y-%m-%d", "%b %Y", "%m/%d/%Y", "%B %Y"])
def test_common_date_formats(fmt):
    series, _ = load(monthly_csv(date_fmt=fmt))
    assert len(series) == 48
    assert series.index[0] == pd.Timestamp("2020-01-01")


def test_semicolon_delimiter():
    series, _ = load(monthly_csv(sep=";"))
    assert len(series) == 48


def test_money_formatted_values():
    # Values containing commas are quoted in a comma-separated file, as Excel writes them.
    series, _ = load(monthly_csv(value_fmt='"${:,.2f}"'))
    assert series.iloc[0] == pytest.approx(1000.0)


def test_underscore_header_is_not_treated_as_delimiter():
    series, _ = load(monthly_csv(header=("sale_month", "unit_sales")))
    assert series.name == "unit_sales"


def test_byte_order_mark_and_latin1():
    bom = b"\xef\xbb\xbf" + monthly_csv()
    assert len(load(bom)[0]) == 48
    latin = monthly_csv(header=("año", "ventas"))
    latin = latin.decode().encode("latin-1")
    series, _ = load(latin)
    assert series.name == "ventas"


def test_mid_month_dates_are_normalized_to_month_start():
    series, _ = load(monthly_csv(day_of_month=15, date_fmt="%Y-%m-%d"))
    assert series.index[0] == pd.Timestamp("2020-01-01")


def test_unsorted_rows():
    lines = monthly_csv().decode().splitlines()
    shuffled = [lines[0]] + lines[1:][::-1]
    series, _ = load("\n".join(shuffled).encode())
    assert series.index.is_monotonic_increasing


# ---------- number parsing ----------


def test_parse_numbers_variants():
    raw = pd.Series(["$1,200", "1 200", "(300)", "€45.5", "abc", ""])
    out = parse_numbers(raw)
    assert out.iloc[:4].tolist() == [1200.0, 1200.0, -300.0, 45.5]
    assert out.iloc[4:].isna().all()


# ---------- column guessing ----------


def test_guesses_columns_regardless_of_order_and_extra_columns():
    text = "store,total_sales,date\n" + "\n".join(
        f"A,{1000 + i},{d:%Y-%m}" for i, d in
        enumerate(pd.date_range("2020-01-01", periods=48, freq="MS"))
    )
    df = read_csv_bytes(text.encode())
    assert guess_columns(df) == ("date", "total_sales")


def test_year_column_is_not_mistaken_for_dates():
    text = "year,month,sales\n" + "\n".join(
        f"{d.year},{d:%Y-%m},{1000 + i}" for i, d in
        enumerate(pd.date_range("2020-01-01", periods=48, freq="MS"))
    )
    df = read_csv_bytes(text.encode())
    assert guess_columns(df)[0] == "month"


def test_guess_returns_none_when_nothing_fits():
    df = read_csv_bytes(b"a,b\nx,y\nz,w\n")
    assert guess_columns(df) == (None, None)


# ---------- daily and weekly data ----------


def test_daily_data_is_summed_and_partial_months_dropped():
    days = pd.date_range("2020-01-10", "2024-03-15", freq="D")  # partial Jan 2020, Mar 2024
    text = "day,units\n" + "\n".join(f"{d:%Y-%m-%d},10" for d in days)
    series, notes = load(text.encode())
    assert series.index[0] == pd.Timestamp("2020-02-01")
    assert series.index[-1] == pd.Timestamp("2024-02-01")
    assert series.loc["2020-02-01"] == pytest.approx(290.0)  # 29 days in Feb 2020
    assert any("daily" in n for n in notes)
    assert any("incomplete" in n for n in notes)


def test_weekly_data_is_summed():
    weeks = pd.date_range("2020-01-06", periods=52 * 4, freq="W-MON")
    text = "week,orders\n" + "\n".join(f"{d:%Y-%m-%d},7" for d in weeks)
    series, notes = load(text.encode())
    assert len(series) >= MIN_HISTORY
    assert any("weekly" in n for n in notes)
    assert set(series.round()).issubset({28.0, 35.0})  # 4 or 5 weeks per month


def test_quarterly_data_is_rejected():
    qs = pd.date_range("2010-01-01", periods=40, freq="QS")
    text = "quarter,sales\n" + "\n".join(f"{d:%Y-%m-%d},100" for d in qs)
    with pytest.raises(UploadError, match="quarterly or yearly"):
        load(text.encode())


# ---------- gaps, duplicates, bad rows ----------


def drop_lines(data: bytes, line_numbers):
    lines = data.decode().splitlines()
    kept = (line for i, line in enumerate(lines) if i not in line_numbers)
    return "\n".join(kept).encode()


def test_small_gaps_are_interpolated_with_note():
    series, notes = load(drop_lines(monthly_csv(), {10, 20}))
    assert len(series) == 48
    assert series.notna().all()
    assert any("Filled 2 missing" in n for n in notes)


def test_large_gaps_are_rejected():
    with pytest.raises(UploadError, match="missing"):
        load(drop_lines(monthly_csv(), {5, 10, 15, 20, 25}))


def test_duplicate_months_are_summed_with_note():
    lines = monthly_csv().decode().splitlines()
    lines.append(lines[1].split(",")[0] + ",500")  # second row for the first month
    series, notes = load("\n".join(lines).encode())
    assert series.iloc[0] == pytest.approx(1000.0 + 500.0)
    assert any("more than one row" in n for n in notes)


def test_a_few_bad_rows_are_skipped_with_note():
    lines = monthly_csv(n=60).decode().splitlines()
    lines.append("not a date,123")
    lines.append("2030-01,n/a")
    series, notes = load("\n".join(lines).encode())
    assert len(series) == 60
    assert any("Skipped 2" in n for n in notes)


def test_mostly_bad_rows_are_rejected():
    df = read_csv_bytes(monthly_csv())
    df["sales"] = "unknown"
    with pytest.raises(UploadError, match="couldn't be read"):
        prepare_series(df, "month", "sales")


# ---------- values and length ----------


def test_zero_or_negative_values_rejected():
    lines = monthly_csv().decode().splitlines()
    lines[5] = lines[5].split(",")[0] + ",0"
    with pytest.raises(UploadError, match="zero or less"):
        load("\n".join(lines).encode())


def test_too_short_rejected():
    with pytest.raises(UploadError, match=f"at least {MIN_HISTORY}"):
        load(monthly_csv(n=24))


# ---------- reading errors ----------


@pytest.mark.parametrize(
    "data, phrase",
    [
        (b"", "empty"),
        (b"   \n  ", "empty"),
        (b"just_one_column\n1\n2\n", "at least two columns"),
        (b"a,b\n", "no data"),
    ],
)
def test_unreadable_files(data, phrase):
    with pytest.raises(UploadError, match=phrase):
        read_csv_bytes(data)


def test_same_column_twice_rejected():
    df = read_csv_bytes(monthly_csv())
    with pytest.raises(UploadError, match="two different columns"):
        prepare_series(df, "sales", "sales")


def test_example_csv_round_trips():
    s, _ = load(monthly_csv())
    again, notes = load(example_csv(s))
    assert len(again) == len(s)
    assert notes == []
