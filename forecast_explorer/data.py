"""Load monthly U.S. retail sales series.

Source: U.S. Census Bureau, Monthly Retail Trade Survey (MRTS), retrieved through FRED
(Federal Reserve Bank of St. Louis). FRED serves each series as a CSV download that
does not need an API key, which keeps the deployed app free of secrets.

We use the NOT seasonally adjusted versions (IDs ending in "USN"). Seasonal adjustment
removes the seasonal pattern, and modelling that pattern is part of what this app's
forecasting methods are supposed to do.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# Built-in series: FRED ID -> display label.
# All are monthly, in millions of dollars, not seasonally adjusted.
SERIES: dict[str, str] = {
    "MRTSSM44X72USN": "Total retail and food services",
    "MRTSSM4451USN": "Grocery stores",
    "MRTSSM448USN": "Clothing and clothing accessory stores",
    "MRTSSM452USN": "General merchandise stores",
    "MRTSSM444USN": "Building materials and garden supply dealers",
    "MRTSSM4541USN": "Electronic shopping and mail-order houses",
    "MRTSSM722USN": "Food services and drinking places",
}

DEFAULT_CACHE_DIR = Path("data/raw")


def build_url(series_id: str) -> str:
    """Return the FRED CSV download URL for a series."""
    return FRED_CSV_URL.format(series_id=series_id)


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """Turn the text of a FRED CSV download into a clean monthly series.

    FRED CSVs have two columns: a date column and a value column named after the series.
    The date column has been called both "DATE" (older files) and "observation_date"
    (newer files), so we use column position rather than the name. Missing values
    appear as "." or as blank cells.

    Returns a float Series indexed by month-start dates, sorted oldest to newest.
    Raises ValueError if the file doesn't look like a monthly FRED series.
    """
    df = pd.read_csv(StringIO(text))

    if df.shape[1] != 2:
        raise ValueError(
            f"Expected 2 columns (date, value) for {series_id}, got {df.shape[1]}: "
            f"{list(df.columns)}"
        )

    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")  # "." and blanks become NaN

    if dates.isna().any():
        bad = df.iloc[:, 0][dates.isna()].head(3).tolist()
        raise ValueError(f"Unparseable dates in {series_id}, e.g. {bad}")

    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=series_id)
    series = series.sort_index().dropna()

    if series.empty:
        raise ValueError(f"No numeric values found for {series_id}")

    # Every date should be the first of a month, with no months missing in between.
    expected = pd.date_range(series.index[0], series.index[-1], freq="MS")
    if len(expected) != len(series) or not (series.index == expected).all():
        missing = expected.difference(series.index)
        raise ValueError(
            f"{series_id} is not a gap-free monthly series; "
            f"{len(missing)} missing month(s), e.g. {[d.date() for d in missing[:3]]}"
        )

    series.index = expected  # attaches the monthly frequency ("MS") to the index
    return series


def fetch_csv_text(series_id: str, timeout: float = 30.0) -> str:
    """Download the raw CSV text for one series from FRED."""
    response = requests.get(build_url(series_id), timeout=timeout)
    response.raise_for_status()
    return response.text


def load_series(
    series_id: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.Series:
    """Load a built-in series, using a local cached copy when one exists.

    The raw CSV is cached in ``cache_dir`` so repeated runs don't re-download it.
    Pass ``refresh=True`` to force a fresh download.
    """
    if series_id not in SERIES:
        raise KeyError(f"Unknown series {series_id!r}. Choose from: {sorted(SERIES)}")

    cache_path = Path(cache_dir) / f"{series_id}.csv"

    if cache_path.exists() and not refresh:
        text = cache_path.read_text()
    else:
        text = fetch_csv_text(series_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)

    return parse_fred_csv(text, series_id)
