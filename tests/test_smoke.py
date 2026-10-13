"""Smoke test: confirms the package imports and the test setup works.

This is a placeholder so the automated test run has something to check from the
first commit. Real tests arrive with the data loader on the next branch.
"""

import forecast_explorer


def test_package_imports():
    assert forecast_explorer.__version__ == "0.1.0"
