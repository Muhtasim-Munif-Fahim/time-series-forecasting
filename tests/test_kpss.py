"""Tests for the KPSS stationarity diagnostic."""

import numpy as np
import pytest

from ts_forecast.diagnostics import adf_test, kpss_test


def _random_walk(n=300, seed=11):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n))


def test_kpss_rejects_unit_root_on_random_walk():
    result = kpss_test(_random_walk())
    assert result["p_value"] < 0.05
    assert result["stationary"] is False
    assert result["statistic"] > 1.0
    assert set(result) >= {
        "statistic",
        "p_value",
        "usedlag",
        "critical_values",
        "stationary",
    }


def test_kpss_accepts_stationary_noise():
    rng = np.random.default_rng(3)
    result = kpss_test(rng.normal(size=200))
    assert result["p_value"] >= 0.05
    assert result["stationary"] is True


def test_kpss_accepts_differenced_random_walk():
    result = kpss_test(np.diff(_random_walk(seed=7)))
    assert result["p_value"] >= 0.05
    assert result["stationary"] is True


def test_kpss_trend_regression_accepts_trend_stationary_series():
    rng = np.random.default_rng(5)
    trend = np.arange(200.0) + rng.normal(size=200)
    with_constant = kpss_test(trend)
    with_trend = kpss_test(trend, regression="ct")
    assert with_constant["stationary"] is False
    assert with_trend["stationary"] is True


def test_kpss_and_adf_agree_on_nonstationary_series():
    walk = _random_walk(seed=2)
    assert adf_test(walk)["stationary"] is False
    assert kpss_test(walk)["stationary"] is False


def test_kpss_and_adf_agree_on_stationary_differenced_series():
    differenced = np.diff(_random_walk(seed=4))
    assert adf_test(differenced)["stationary"] is True
    assert kpss_test(differenced)["stationary"] is True


def test_kpss_fixed_nlags_is_respected():
    result = kpss_test(_random_walk(), nlags=5)
    assert result["usedlag"] == 5


def test_kpss_validation_errors():
    with pytest.raises(ValueError, match="at least ten observations"):
        kpss_test([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="finite"):
        kpss_test([1.0, 2.0, np.nan] + list(np.arange(8.0)))
    with pytest.raises(ValueError, match="regression"):
        kpss_test(np.arange(15.0), regression="x")
    with pytest.raises(ValueError, match="nlags"):
        kpss_test(np.arange(15.0), nlags=-1)
    with pytest.raises(ValueError, match="nlags"):
        kpss_test(np.arange(15.0), nlags="x")
    with pytest.raises(ValueError, match="alpha"):
        kpss_test(np.arange(15.0), alpha=1.0)
