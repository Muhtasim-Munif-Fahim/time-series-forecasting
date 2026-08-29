"""Tests for the classical seasonal decomposition."""

import numpy as np
import pytest

from ts_forecast.evaluation import seasonal_decompose


def test_additive_decomposition_recovers_components():
    rng = np.random.default_rng(13)
    n, period = 240, 12
    time = np.arange(n, dtype=float)
    trend = 0.5 * time + 10.0
    seasonal = np.tile(
        np.sin(np.arange(period) * 2.0 * np.pi / period), n // period
    )
    noise = rng.normal(scale=0.1, size=n)
    result = seasonal_decompose(trend + seasonal + noise, period)

    assert set(result) == {"trend", "seasonal", "residual"}
    assert result["trend"].shape == (n,)
    assert not np.isfinite(result["trend"][0])
    assert not np.isfinite(result["trend"][-1])

    finite = np.isfinite(result["trend"])
    np.testing.assert_allclose(result["trend"][finite], trend[finite], rtol=0.05)
    np.testing.assert_allclose(
        result["seasonal"][finite], seasonal[finite], atol=0.1
    )
    assert np.abs(result["residual"][finite]).max() < 0.4


def test_additive_components_reconstruct_the_series():
    rng = np.random.default_rng(3)
    n, period = 200, 7
    values = (
        np.linspace(1.0, 20.0, n)
        + 2.0 * np.sin(np.arange(n) * 2.0 * np.pi / period)
        + rng.normal(scale=0.05, size=n)
    )
    result = seasonal_decompose(values, period)
    finite = np.isfinite(result["trend"])
    reconstructed = (
        result["trend"] + result["seasonal"] + result["residual"]
    )
    np.testing.assert_allclose(
        reconstructed[finite], values[finite], rtol=1e-10, atol=1e-10
    )


def test_multiplicative_decomposition_recovers_components():
    rng = np.random.default_rng(5)
    n, period = 240, 12
    trend = np.linspace(2.0, 8.0, n)
    seasonal = np.tile(
        1.0 + 0.3 * np.sin(np.arange(period) * 2.0 * np.pi / period),
        n // period,
    )
    noise = 1.0 + rng.normal(scale=0.02, size=n)
    result = seasonal_decompose(trend * seasonal * noise, period, model="multiplicative")

    finite = np.isfinite(result["trend"])
    np.testing.assert_allclose(result["trend"][finite], trend[finite], rtol=0.05)
    np.testing.assert_allclose(
        result["seasonal"][finite], seasonal[finite], atol=0.05
    )
    np.testing.assert_allclose(result["residual"][finite], 1.0, atol=0.08)


def test_multiplicative_components_reconstruct_the_series():
    n, period = 180, 6
    trend = np.linspace(1.5, 9.0, n)
    seasonal = np.tile(
        1.0 + 0.4 * np.cos(np.arange(period) * 2.0 * np.pi / period),
        n // period,
    )
    result = seasonal_decompose(trend * seasonal, period, model="multiplicative")
    finite = np.isfinite(result["trend"])
    reconstructed = (
        result["trend"] * result["seasonal"] * result["residual"]
    )
    np.testing.assert_allclose(
        reconstructed[finite], (trend * seasonal)[finite], rtol=1e-8
    )


def test_even_period_uses_centered_moving_average():
    rng = np.random.default_rng(9)
    n, period = 120, 4
    values = np.linspace(0.0, 6.0, n) + rng.normal(scale=0.15, size=n)
    result = seasonal_decompose(values, period)
    finite = np.isfinite(result["trend"])
    assert finite.sum() == n - period
    np.testing.assert_allclose(result["trend"][finite], np.linspace(0.0, 6.0, n)[finite], atol=0.4)


def test_seasonal_component_is_centered_and_periodic():
    n, period = 96, 8
    values = np.tile(np.arange(period, dtype=float), n // period) * 0.1
    result = seasonal_decompose(values, period)
    finite = np.isfinite(result["seasonal"])
    assert np.mean(result["seasonal"][finite]) < 1e-10
    offset = (n - (n - period)) // 2
    first_cycle = result["seasonal"][offset : offset + period]
    np.testing.assert_allclose(
        result["seasonal"][finite],
        np.tile(first_cycle, (n - period) // period),
        atol=1e-10,
    )


def test_validation_errors():
    with pytest.raises(ValueError, match="seasonal_period"):
        seasonal_decompose(np.arange(30.0), 1)
    with pytest.raises(ValueError, match="model"):
        seasonal_decompose(np.arange(30.0), 7, model="log")
    with pytest.raises(ValueError, match="finite"):
        seasonal_decompose([1.0, np.nan] + list(np.arange(28.0)), 7)
    with pytest.raises(ValueError, match="two complete seasonal periods"):
        seasonal_decompose(np.arange(13.0), 7)
    with pytest.raises(ValueError, match="strictly positive"):
        seasonal_decompose(np.arange(-5.0, 25.0), 7, model="multiplicative")
    with pytest.raises(ValueError, match="constant"):
        seasonal_decompose(np.ones(40), 7)
