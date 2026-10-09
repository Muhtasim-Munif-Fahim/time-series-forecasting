"""Tests for Fourier-seasonality regression forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_forecast.models import fit_fourier_regression, fourier_regression_forecast


def _frame(values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)})


def test_returns_correct_length():
    t = np.arange(40, dtype=float)
    y = 3.0 + 0.1 * t + 2.0 * np.sin(2 * np.pi * t / 7)
    forecast = fourier_regression_forecast(
        _frame(y), "value", steps=5, seasonal_period=7, n_harmonics=1
    )
    assert forecast.shape == (5,)


def test_recovers_known_sine():
    t = np.arange(70, dtype=float)
    y = 5.0 + 3.0 * np.sin(2 * np.pi * t / 7) + 1.5 * np.cos(2 * np.pi * t / 7)
    forecast = fourier_regression_forecast(
        _frame(y),
        "value",
        steps=7,
        seasonal_period=7,
        n_harmonics=1,
        include_trend=False,
    )
    t_h = np.arange(70, 77, dtype=float)
    expected = 5.0 + 3.0 * np.sin(2 * np.pi * t_h / 7) + 1.5 * np.cos(2 * np.pi * t_h / 7)
    assert forecast.tolist() == pytest.approx(expected.tolist(), abs=0.05)


def test_trend_extrapolated():
    t = np.arange(30, dtype=float)
    y = 1.0 + 0.5 * t
    forecast = fourier_regression_forecast(
        _frame(y),
        "value",
        steps=3,
        seasonal_period=7,
        n_harmonics=1,
        include_trend=True,
    )
    assert forecast[0] == pytest.approx(1.0 + 0.5 * 30, abs=0.2)
    assert forecast[1] > forecast[0]


def test_fit_metadata():
    y = np.arange(20, dtype=float)
    fitted = fit_fourier_regression(
        _frame(y), "value", seasonal_period=4, n_harmonics=1
    )
    assert fitted["n_obs"] == 20
    assert fitted["periods"] == (4.0,)
    assert fitted["sigma"] >= 0.0


def test_multi_period():
    t = np.arange(120, dtype=float)
    y = (
        np.sin(2 * np.pi * t / 7)
        + 0.5 * np.sin(2 * np.pi * t / 30)
    )
    forecast = fourier_regression_forecast(
        _frame(y),
        "value",
        steps=5,
        seasonal_period=(7, 30),
        n_harmonics=1,
        include_trend=False,
    )
    assert forecast.shape == (5,)
    assert np.all(np.isfinite(forecast))


def test_invalid_steps():
    with pytest.raises(ValueError, match="steps"):
        fourier_regression_forecast(_frame(np.arange(10.0)), "value", steps=0)


def test_invalid_harmonics():
    with pytest.raises(ValueError, match="n_harmonics"):
        fit_fourier_regression(_frame(np.arange(10.0)), "value", n_harmonics=0)
