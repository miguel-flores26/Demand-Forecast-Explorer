"""Tests for the data loader.

These tests never touch the internet. They feed the parser small, hand-written CSV text
shaped like FRED's downloads, and replace the download function with a fake when testing
the caching logic. That keeps the tests fast and means they can't fail because a website
is down.
"""

import pandas as pd
import pytest

from forecast_explorer import data

NEW_STYLE = """observation_date,MRTSSM4451USN
2024-01-01,100.0
2024-02-01,110.5
2024-03-01,120.25
"""

OLD_STYLE = """DATE,MRTSSM4451USN
2024-01-01,100.0
2024-02-01,110.5
2024-03-01,120.25
"""


@pytest.mark.parametrize("text", [NEW_STYLE, OLD_STYLE])
def test_parse_handles_both_header_styles(text):
    s = data.parse_fred_csv(text, "MRTSSM4451USN")
    assert list(s) == [100.0, 110.5, 120.25]
    assert s.name == "MRTSSM4451USN"


def test_parse_returns_monthly_index():
    s = data.parse_fred_csv(NEW_STYLE, "X")
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.freqstr == "MS"
    assert s.index[0] == pd.Timestamp("2024-01-01")


def test_parse_returns_floats_even_for_whole_numbers():
    # Real FRED retail files contain whole numbers only; found when the loader returned
    # int64 on live data.
    text = "observation_date,X\n2024-01-01,75774\n2024-02-01,75343\n"
    s = data.parse_fred_csv(text, "X")
    assert s.dtype == "float64"


def test_parse_sorts_out_of_order_rows():
    text = "observation_date,X\n2024-03-01,3\n2024-01-01,1\n2024-02-01,2\n"
    assert list(data.parse_fred_csv(text, "X")) == [1.0, 2.0, 3.0]


def test_parse_drops_trailing_missing_marker():
    # FRED marks not-yet-released values with "."
    text = "observation_date,X\n2024-01-01,1\n2024-02-01,2\n2024-03-01,.\n"
    s = data.parse_fred_csv(text, "X")
    assert len(s) == 2
    assert s.index[-1] == pd.Timestamp("2024-02-01")


def test_parse_rejects_gap_in_middle():
    text = "observation_date,X\n2024-01-01,1\n2024-02-01,.\n2024-03-01,3\n"
    with pytest.raises(ValueError, match="gap-free"):
        data.parse_fred_csv(text, "X")


def test_parse_rejects_wrong_column_count():
    text = "observation_date,X,Y\n2024-01-01,1,2\n"
    with pytest.raises(ValueError, match="Expected 2 columns"):
        data.parse_fred_csv(text, "X")


def test_parse_rejects_bad_dates():
    text = "observation_date,X\nnot-a-date,1\n"
    with pytest.raises(ValueError, match="Unparseable dates"):
        data.parse_fred_csv(text, "X")


def test_parse_rejects_all_missing():
    text = "observation_date,X\n2024-01-01,.\n2024-02-01,.\n"
    with pytest.raises(ValueError, match="No numeric values"):
        data.parse_fred_csv(text, "X")


def test_builtin_series_are_not_seasonally_adjusted():
    # Deliberate choice: the models need to see the seasonal pattern.
    assert all(sid.endswith("USN") for sid in data.SERIES)


def test_build_url():
    assert data.build_url("ABC").endswith("fredgraph.csv?id=ABC")


def test_load_unknown_series_raises():
    with pytest.raises(KeyError, match="Unknown series"):
        data.load_series("NOT_A_SERIES")


def test_load_downloads_once_then_uses_cache(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(series_id, timeout=30.0):
        calls.append(series_id)
        return NEW_STYLE

    monkeypatch.setattr(data, "fetch_csv_text", fake_fetch)

    first = data.load_series("MRTSSM4451USN", cache_dir=tmp_path)
    second = data.load_series("MRTSSM4451USN", cache_dir=tmp_path)

    assert calls == ["MRTSSM4451USN"]  # downloaded only the first time
    assert (tmp_path / "MRTSSM4451USN.csv").exists()
    pd.testing.assert_series_equal(first, second)


def test_load_refresh_forces_download(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        data, "fetch_csv_text", lambda sid, timeout=30.0: calls.append(sid) or NEW_STYLE
    )
    data.load_series("MRTSSM4451USN", cache_dir=tmp_path)
    data.load_series("MRTSSM4451USN", cache_dir=tmp_path, refresh=True)
    assert len(calls) == 2
