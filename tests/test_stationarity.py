"""Tests for stationarity diagnostics and differencing suggestions."""

import numpy as np
import pytest

from ts_forecast.diagnostics import adf_test, stationarity_report


def _random_walk(n=400, seed=42):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n))


def test_adf_rejects_unit_root_on_random_walk():
    result = adf_test(_random_walk())
    assert result["p_value"] > 0.05
    assert result["stationary"] is False
    assert result["statistic"] > -3.0
    assert set(result) >= {
        "statistic",
        "p_value",
        "usedlag",
        "nobs",
        "critical_values",
        "stationary",
    }


def test_adf_accepts_stationary_noise():
    rng = np.random.default_rng(3)
    result = adf_test(rng.normal(size=150))
    assert result["p_value"] < 0.05
    assert result["stationary"] is True


def test_adf_accepts_differenced_random_walk():
    result = adf_test(np.diff(_random_walk()))
    assert result["p_value"] < 0.05
    assert result["stationary"] is True


def test_adf_trend_regression_accepts_trend_stationary_series():
    rng = np.random.default_rng(5)
    trend = np.arange(150.0) + rng.normal(size=150)
    with_constant = adf_test(trend)
    with_trend = adf_test(trend, regression="ct")
    assert with_constant["stationary"] is False
    assert with_trend["stationary"] is True


def test_adf_fixed_maxlag_is_respected():
    result = adf_test(_random_walk(), maxlag=4, autolag=None)
    assert result["usedlag"] == 4


def test_adf_validation_errors():
    with pytest.raises(ValueError, match="at least ten observations"):
        adf_test([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="finite"):
        adf_test([1.0, 2.0, np.nan] + list(np.arange(7.0)))
    with pytest.raises(ValueError, match="regression"):
        adf_test(np.arange(15.0), regression="x")
    with pytest.raises(ValueError, match="maxlag"):
        adf_test(np.arange(15.0), maxlag=-1)
    with pytest.raises(ValueError, match="autolag"):
        adf_test(np.arange(15.0), autolag="X")


def test_stationarity_report_flags_noise_as_stationary():
    rng = np.random.default_rng(9)
    report = stationarity_report(rng.normal(size=150))
    assert report["verdict"] == "stationary"
    assert report["suggested_diffs"] == 0
    assert report["tests"][0]["diffs"] == 0
    assert report["tests"][0]["stationary"] is True


def test_stationarity_report_suggests_one_difference_for_random_walk():
    report = stationarity_report(_random_walk(seed=1))
    assert report["verdict"] == "non-stationary"
    assert report["suggested_diffs"] >= 1
    assert report["tests"][0]["stationary"] is False
    assert any(test["stationary"] for test in report["tests"])


def test_stationarity_report_trajectory_starts_nonstationary_and_recovers():
    report = stationarity_report(_random_walk(seed=2), max_diffs=3)
    p_values = [test["p_value"] for test in report["tests"]]
    assert p_values[0] > 0.05
    assert any(p < 0.05 for p in p_values)


def test_stationarity_report_respects_max_diffs():
    report = stationarity_report(_random_walk(seed=4), max_diffs=1)
    assert len(report["tests"]) == 2
    assert report["tests"][-1]["diffs"] == 1


def test_stationarity_report_validation_errors():
    with pytest.raises(ValueError, match="max_diffs"):
        stationarity_report(np.arange(20.0), max_diffs=0)
    with pytest.raises(ValueError, match="alpha"):
        stationarity_report(np.arange(20.0), alpha=1.0)
    with pytest.raises(ValueError, match="at least ten observations"):
        stationarity_report([1.0, 2.0, 3.0])
