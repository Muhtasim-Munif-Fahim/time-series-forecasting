"""Tests for the Mann-Kendall trend test."""

import numpy as np
import pytest
from scipy.stats import kendalltau

from ts_forecast.diagnostics import mann_kendall_test


def test_strictly_increasing_series_is_reported_as_increasing():
    result = mann_kendall_test(np.arange(100.0))
    assert result["trend"] == "increasing"
    assert result["tau"] > 0.9
    assert result["p_value"] < 0.05
    assert set(result) >= {"tau", "p_value", "s", "var_s", "trend", "alpha"}


def test_strictly_decreasing_series_is_reported_as_decreasing():
    result = mann_kendall_test(np.arange(100.0)[::-1])
    assert result["trend"] == "decreasing"
    assert result["tau"] < -0.9
    assert result["p_value"] < 0.05


def test_white_noise_has_no_significant_trend():
    rng = np.random.default_rng(21)
    result = mann_kendall_test(rng.normal(size=200))
    assert result["trend"] == "no trend"
    assert result["p_value"] >= 0.05


def test_tau_matches_scipy_reference():
    rng = np.random.default_rng(8)
    values = rng.normal(size=120) + np.linspace(0.0, 2.0, 120)
    result = mann_kendall_test(values)
    reference = kendalltau(np.arange(values.size), values).statistic
    np.testing.assert_allclose(result["tau"], reference, atol=1e-12)


def test_noisy_trend_is_detected():
    rng = np.random.default_rng(4)
    values = np.cumsum(rng.normal(size=250)) + np.linspace(0.0, 5.0, 250)
    result = mann_kendall_test(values)
    assert result["trend"] in {"increasing", "decreasing"}
    assert result["p_value"] < 0.05


def test_ties_are_handled():
    rng = np.random.default_rng(6)
    values = np.round(rng.uniform(0.0, 5.0, size=120))
    result = mann_kendall_test(values)
    assert result["trend"] in {"increasing", "decreasing", "no trend"}
    assert np.isfinite(result["tau"])


def test_constant_series_is_rejected():
    with pytest.raises(ValueError, match="constant"):
        mann_kendall_test(np.ones(50))


def test_validation_errors():
    with pytest.raises(ValueError, match="at least ten observations"):
        mann_kendall_test([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        mann_kendall_test([1.0, np.nan] + list(np.arange(9.0)))
    with pytest.raises(ValueError, match="alpha"):
        mann_kendall_test(np.arange(20.0), alpha=0.0)
