"""Tests for the Ljung-Box residual autocorrelation test."""

import numpy as np
import pytest

from ts_forecast.evaluation import ljung_box_test


def _white_noise(n=300, seed=13):
    return np.random.default_rng(seed).normal(size=n)


def _ar1(n=300, phi=0.9, seed=17):
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=n)
    series = np.zeros(n)
    for i in range(1, n):
        series[i] = phi * series[i - 1] + innovations[i]
    return series


def test_ljung_box_accepts_white_noise():
    result = ljung_box_test(_white_noise())
    assert result["autocorrelated"] is False
    assert result["significant_lags"] == []
    assert all(test["p_value"] > 0.05 for test in result["tests"])
    assert set(result) >= {"autocorrelated", "alpha", "significant_lags", "tests"}
    assert {test["lag"] for test in result["tests"]} == set(range(1, 11))


def test_ljung_box_rejects_autocorrelated_residuals():
    result = ljung_box_test(_ar1())
    assert result["autocorrelated"] is True
    assert result["significant_lags"] != []
    assert result["tests"][0]["p_value"] < 0.05
    assert result["tests"][0]["lb_stat"] > 0


def test_ljung_box_respects_explicit_lag_sequence():
    result = ljung_box_test(_white_noise(), lags=[5, 10])
    assert [test["lag"] for test in result["tests"]] == [5, 10]


def test_ljung_box_respects_integer_lag_count():
    result = ljung_box_test(_ar1(), lags=3)
    assert [test["lag"] for test in result["tests"]] == [1, 2, 3]
    assert result["autocorrelated"] is True


def test_ljung_box_alpha_changes_verdict_on_marginal_series():
    marginal = _ar1(n=80, phi=0.1, seed=31)
    strict = ljung_box_test(marginal, lags=[1], alpha=0.01)
    relaxed = ljung_box_test(marginal, lags=[1], alpha=0.3)
    assert strict["autocorrelated"] is False
    assert relaxed["autocorrelated"] is True


def test_ljung_box_on_model_residuals():
    rng = np.random.default_rng(23)
    observed = np.cumsum(rng.normal(size=120))
    predicted = observed[:-1]
    residuals = observed[1:] - predicted
    result = ljung_box_test(residuals, lags=5)
    assert result["autocorrelated"] is False


def test_ljung_box_validation_errors():
    with pytest.raises(ValueError, match="empty"):
        ljung_box_test([])
    with pytest.raises(ValueError, match="finite"):
        ljung_box_test([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(ValueError, match="at least two residuals"):
        ljung_box_test([1.0])
    with pytest.raises(ValueError, match="variance"):
        ljung_box_test([2.0, 2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="alpha"):
        ljung_box_test(_white_noise(), alpha=1.0)
    with pytest.raises(ValueError, match="lags"):
        ljung_box_test(_white_noise(), lags=0)
    with pytest.raises(ValueError, match="lags"):
        ljung_box_test(_white_noise(), lags=True)
    with pytest.raises(ValueError, match="lags"):
        ljung_box_test(_white_noise(), lags=[])
    with pytest.raises(ValueError, match="below the number"):
        ljung_box_test(np.arange(10.0), lags=10)
