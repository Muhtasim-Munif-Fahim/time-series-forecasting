"""Tests for Theil's U inequality coefficient."""
from __future__ import annotations

import numpy as np
import pytest

from ts_forecast.evaluation import theils_u


def test_perfect_forecast_returns_zero() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.copy()
    assert theils_u(y_true, y_pred) == 0.0


def test_naive_forecast_returns_one() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 1.0, 2.0, 3.0, 4.0])
    assert theils_u(y_true, y_pred) == pytest.approx(1.0, abs=1e-6)


def test_better_than_naive_returns_less_than_one() -> None:
    np.random.seed(0)
    n = 200
    y_true = np.cumsum(np.random.randn(n))
    y_pred = y_true + np.random.randn(n) * 0.1
    assert theils_u(y_true, y_pred) < 1.0


def test_worse_than_naive_returns_greater_than_one() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    assert theils_u(y_true, y_pred) > 1.0


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="equal length"):
        theils_u(np.array([1.0, 2.0]), np.array([1.0]))


def test_too_few_observations_raise() -> None:
    with pytest.raises(ValueError, match="at least two"):
        theils_u(np.array([1.0]), np.array([1.0]))


def test_constant_series_naive_rmse_zero_returns_zero() -> None:
    y_true = np.array([5.0, 5.0, 5.0, 5.0])
    y_pred = np.array([5.0, 5.0, 5.0, 5.0])
    assert theils_u(y_true, y_pred) == 0.0


def test_constant_series_model_worse_returns_inf() -> None:
    y_true = np.array([5.0, 5.0, 5.0, 5.0])
    y_pred = np.array([6.0, 6.0, 6.0, 6.0])
    assert theils_u(y_true, y_pred) == float("inf")


def test_nan_input_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        theils_u(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]))


def test_reproducible() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    a = theils_u(y_true, y_pred)
    b = theils_u(y_true, y_pred)
    assert a == b
