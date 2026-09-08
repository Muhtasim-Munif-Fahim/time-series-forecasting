"""Tests for the plot_decomposition visualization."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from ts_forecast.visualization import plot_decomposition


def _make_series() -> np.ndarray:
    rng = np.random.default_rng(42)
    n, period = 240, 12
    time = np.arange(n, dtype=float)
    trend = 0.5 * time + 10.0
    seasonal = np.tile(
        np.sin(np.arange(period) * 2.0 * np.pi / period), n // period
    )
    noise = rng.normal(scale=0.1, size=n)
    return trend + seasonal + noise


def test_plot_decomposition_returns_figure_and_axes() -> None:
    series = _make_series()
    fig, axes = plot_decomposition(series, 12)
    assert len(axes) == 4
    assert axes[0].get_ylabel() == "Observed"
    assert axes[1].get_ylabel() == "Trend"
    assert axes[2].get_ylabel() == "Seasonal"
    assert axes[3].get_ylabel() == "Residual"


def test_plot_decomposition_multiplicative() -> None:
    n, period = 240, 12
    trend = np.linspace(2.0, 8.0, n)
    seasonal = np.tile(
        1.0 + 0.3 * np.sin(np.arange(period) * 2.0 * np.pi / period),
        n // period,
    )
    values = trend * seasonal + 0.1
    fig, axes = plot_decomposition(values, period, model="multiplicative")
    assert axes[0].get_title().startswith("Seasonal Decomposition")


def test_plot_decomposition_custom_title() -> None:
    series = _make_series()
    fig, axes = plot_decomposition(series, 12, title="My Title")
    assert axes[0].get_title() == "My Title"
