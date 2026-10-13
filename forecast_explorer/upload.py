"""Turn a user's uploaded CSV into a clean monthly series the models can use.

Real-world files are messy, so this module is defensive. It:

1. reads the file, detecting the delimiter and handling common text encodings;
2. guesses which column holds dates and which holds values (the user can override);
3. cleans values like "$1,200" or "(300)" into numbers;
4. works out whether the data is daily, weekly, or monthly, and adds daily or weekly
   data up into months (dropping incomplete first/last months so they don't look like dips);
5. fills a small number of missing months by straight-line interpolation, and refuses
   files with larger gaps;
6. checks there is enough history and that all values are positive.

Every adjustment is recorded as a plain-language note so the app can tell the user exactly
what was done to their data. Problems that can't be fixed safely raise UploadError with a
message written for the user, not for a developer.
"""

from __future__ import annotations

import csv
import io
import re

import numpy as np
import pandas as pd

from forecast_explorer.models import MIN_HISTORY

MAX_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_ROWS = 50_000
MAX_INTERPOLATED_MONTHS = 3
PARSE_THRESHOLD = 0.9  # share of non-blank cells that must parse for a column to qualify
MAX_BAD_ROWS_SHARE = 0.10  # drop up to 10% unparseable rows, with a note; more is an error


class UploadError(ValueError):
    """A problem with an uploaded file, described in user-facing language."""


# ---------- reading ----------


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read uploaded bytes into a DataFrame of strings, detecting the delimiter."""
    if not data or not data.strip():
        raise UploadError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise UploadError(f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")

    text = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    # Detect the delimiter, but only among real CSV delimiters. Left unrestricted, the
    # detector can pick letters or underscores from the header (e.g. split "unit_sales").
    try:
        sep = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, skip_blank_lines=True)
    except Exception as exc:  # pandas raises several different parser errors
        raise UploadError(
            "Couldn't read this file as a CSV. Check that it's a plain comma-separated "
            "file with a header row."
        ) from exc

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    if df.shape[1] < 2:
        raise UploadError(
            "The file needs at least two columns: one with dates and one with values."
        )
    if len(df) > MAX_ROWS:
        raise UploadError(f"The file has more than {MAX_ROWS:,} rows.")
    if df.empty:
        raise UploadError("The file has a header row but no data.")
    return df


# ---------- parsing cells ----------

_NUMBER_JUNK = re.compile(r"[$€£¥,\s]")


def parse_numbers(values: pd.Series) -> pd.Series:
    """Convert text like "$1,200", "1 200", or "(300)" to floats; unparseable -> NaN."""
    s = values.astype(str).str.strip()
    negative = s.str.match(r"^\(.*\)$")
    s = s.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    s = s.str.replace(_NUMBER_JUNK, "", regex=True)
    nums = pd.to_numeric(s, errors="coerce")
    return nums.where(~negative, -nums)


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse date text in common formats ("2024-01", "Jan 2024", "1/15/2024"); bad -> NaT."""
    return pd.to_datetime(values.astype(str).str.strip(), errors="coerce", format="mixed")


def _parse_share(parsed: pd.Series, raw: pd.Series) -> float:
    filled = raw.notna() & (raw.astype(str).str.strip() != "")
    if filled.sum() == 0:
        return 0.0
    return float(parsed[filled].notna().mean())


def guess_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Best guess at (date column, value column). Either may be None if nothing qualifies."""
    date_col = None
    for col in df.columns:
        # Pure numbers like 2023 or 45000 parse as dates too; don't treat them as dates.
        looks_numeric = _parse_share(parse_numbers(df[col]), df[col]) >= PARSE_THRESHOLD
        if not looks_numeric and _parse_share(parse_dates(df[col]), df[col]) >= PARSE_THRESHOLD:
            date_col = col
            break

    value_col = None
    for col in df.columns:
        if col != date_col and _parse_share(parse_numbers(df[col]), df[col]) >= PARSE_THRESHOLD:
            value_col = col
            break
    return date_col, value_col


# ---------- building the monthly series ----------


def _detect_frequency(dates: pd.Series) -> str:
    gaps = dates.drop_duplicates().sort_values().diff().dropna().dt.days
    if gaps.empty:
        raise UploadError("The file needs more than one distinct date.")
    median_gap = gaps.median()
    if median_gap <= 1.5:
        return "daily"
    if median_gap <= 8:
        return "weekly"
    if 27 <= median_gap <= 32:
        return "monthly"
    raise UploadError(
        "This looks like quarterly or yearly data (dates about "
        f"{median_gap:.0f} days apart). The app needs monthly, weekly, or daily data."
    )


def prepare_series(
    df: pd.DataFrame, date_col: str, value_col: str
) -> tuple[pd.Series, list[str]]:
    """Clean the chosen columns into a gap-free monthly series.

    Returns the series (month-start index, float values) and a list of notes describing
    every change made. Raises UploadError if the data can't be used safely.
    """
    if date_col == value_col:
        raise UploadError("Choose two different columns for dates and values.")

    notes: list[str] = []
    dates = parse_dates(df[date_col])
    values = parse_numbers(df[value_col])

    bad = dates.isna() | values.isna()
    if bad.mean() > MAX_BAD_ROWS_SHARE:
        raise UploadError(
            f"{bad.sum()} of {len(df)} rows have a date or value that couldn't be read. "
            f"Check that '{date_col}' holds dates and '{value_col}' holds numbers."
        )
    if bad.any():
        notes.append(f"Skipped {bad.sum()} row(s) with an unreadable date or value.")
    dates, values = dates[~bad], values[~bad]

    freq = _detect_frequency(dates)
    frame = pd.DataFrame({"date": dates, "value": values})
    frame["month"] = frame["date"].dt.to_period("M")

    if freq in ("daily", "weekly"):
        notes.append(f"Your data looks {freq}; values were added up into monthly totals.")
        first_day, last_day = frame["date"].min(), frame["date"].max()
        # A month only counts as complete if the data spans it from start to end.
        span_slack = pd.Timedelta(days=0 if freq == "daily" else 6)
        drop = []
        if first_day - span_slack > first_day.to_period("M").start_time:
            drop.append(first_day.to_period("M"))
        if last_day + span_slack < last_day.to_period("M").end_time.normalize():
            drop.append(last_day.to_period("M"))
        if drop:
            frame = frame[~frame["month"].isin(drop)]
            names = ", ".join(p.strftime("%b %Y") for p in drop)
            notes.append(f"Dropped incomplete month(s) at the edges of your data: {names}.")
    else:
        dupes = frame["month"].duplicated(keep=False)
        if dupes.any():
            n_months = frame.loc[dupes, "month"].nunique()
            notes.append(
                f"{n_months} month(s) had more than one row; their values were added together."
            )

    monthly = frame.groupby("month")["value"].sum()
    monthly.index = monthly.index.to_timestamp()  # month-start dates
    full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    series = monthly.reindex(full_index)

    missing = series.index[series.isna()]
    if len(missing) > MAX_INTERPOLATED_MONTHS:
        examples = ", ".join(d.strftime("%b %Y") for d in missing[:5])
        raise UploadError(
            f"{len(missing)} months are missing from your data (e.g. {examples}). "
            f"The app can fill in at most {MAX_INTERPOLATED_MONTHS}."
        )
    if len(missing):
        series = series.interpolate(method="linear")
        names = ", ".join(d.strftime("%b %Y") for d in missing)
        notes.append(
            f"Filled {len(missing)} missing month(s) by straight-line interpolation: {names}."
        )

    if (series <= 0).any():
        n_bad = int((series <= 0).sum())
        raise UploadError(
            f"{n_bad} month(s) have a total of zero or less. The models work on the log "
            "scale, so every month needs a positive value."
        )
    if len(series) < MIN_HISTORY:
        raise UploadError(
            f"The data covers {len(series)} months; at least {MIN_HISTORY} are needed so "
            "the models can learn the yearly pattern."
        )

    series = series.astype(float)
    series.name = value_col
    series.index = pd.DatetimeIndex(series.index, freq="MS")
    return series, notes


def example_csv(series: pd.Series) -> bytes:
    """A small example file in the expected format, built from a real series."""
    out = pd.DataFrame({"month": series.index.strftime("%Y-%m"), "sales": np.round(series, 0)})
    return out.to_csv(index=False).encode()
