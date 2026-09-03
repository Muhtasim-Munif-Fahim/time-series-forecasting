"""Tests for bootstrap_forecast_intervals."""

from __future__ import annotations

import numpy as np
import pytest

from ts_forecast.evaluation import bootstrap_forecast_intervals


def test_returns_expected_keys() -> None:
    rng = np.random.default_rng(0)
    residuals = rng.standard_normal(100)
    forecast = np.array([0.5, 1.0, 1.5])
    result = bootstrap_forecast_intervals(residuals, forecast, n_bootstrap=200, seed=42)
    assert set(result) == {"lower", "upper", "half_width", "alpha"}
    assert result["lower"].shape == forecast.shape
    assert result["upper"].shape == forecast.shape
    assert (result["upper"] > result["lower"]).all()


def test_is_deterministic_for_a_given_seed() -> None:
    rng = np.random.default_rng(0)
    residuals = rng.standard_normal(80)
    forecast = np.array([0.0, 0.5, 1.0])
    a = bootstrap_forecast_intervals(residuals, forecast, n_bootstrap=100, seed=7)
    b = bootstrap_forecast_intervals(residuals, forecast, n_bootstrap=100, seed=7)
    assert a["half_width"] == pytest.approx(b["half_width"])


def test_wider_coverage_gives_wider_band() -> None:
    rng = np.random.default_rng(0)
    residuals = rng.standard_normal(80)
    forecast = np.array([0.0])
    narrow = bootstrap_forecast_intervals(residuals, forecast, coverage=0.5, n_bootstrap=200, seed=1)
    wide = bootstrap_forecast_intervals(residuals, forecast, coverage=0.99, n_bootstrap=200, seed=1)
    assert wide["half_width"] > narrow["half_width"]


def test_block_size_must_be_positive() -> None:
    residuals = np.array([0.1, -0.2, 0.3])
    with pytest.raises(ValueError, match="block_size"):
        bootstrap_forecast_intervals(residuals, np.array([0.0]), block_size=0)


def test_rejects_invalid_coverage() -> None:
    residuals = np.array([0.1, -0.2, 0.3])
    forecast = np.array([0.0])
    with pytest.raises(ValueError, match="coverage"):
        bootstrap_forecast_intervals(residuals, forecast, coverage=0.0)
    with pytest.raises(ValueError, match="coverage"):
        bootstrap_forecast_intervals(residuals, forecast, coverage=1.0)


def test_rejects_invalid_n_bootstrap() -> None:
    residuals = np.array([0.1, -0.2, 0.3])
    forecast = np.array([0.0])
    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap_forecast_intervals(residuals, forecast, n_bootstrap=0)


def test_handles_block_size_greater_than_one() -> None:
    rng = np.random.default_rng(0)
    residuals = rng.standard_normal(40)
    forecast = np.array([0.0, 0.5])
    result = bootstrap_forecast_intervals(residuals, forecast, n_bootstrap=50, seed=2, block_size=4)
    assert (result["upper"] > result["lower"]).all()


def test_handles_non_finite_residuals() -> None:
    residuals = np.array([0.1, np.nan, -0.2, np.inf, 0.3, 0.0])
    forecast = np.array([0.0])
    result = bootstrap_forecast_intervals(residuals, forecast, n_bootstrap=50, seed=3)
    assert (result["upper"] > result["lower"]).all()


def test_rejects_when_too_few_residuals_for_block() -> None:
    residuals = np.array([0.1, -0.2])
    forecast = np.array([0.0])
    with pytest.raises(ValueError, match="not enough"):
        bootstrap_forecast_intervals(residuals, forecast, n_bootstrap=20, block_size=5)