"""Tests for the Holt-Winters exponential smoothing forecast model."""

import numpy as np
import pandas as pd
import pytest

from ts_forecast.models import holt_winters_forecast


def _frame(values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)})


def test_returns_correct_length():
    frame = _frame(np.arange(1.0, 21.0))
    forecast = holt_winters_forecast(frame, "value", steps=5)
    assert isinstance(forecast, np.ndarray)
    assert forecast.size == 5


def test_constant_series_flat_forecast():
    frame = _frame(np.full(16, 5.0))
    forecast = holt_winters_forecast(
        frame, "value", steps=3, trend=None, seasonal=None
    )
    assert forecast.tolist() == pytest.approx([5.0, 5.0, 5.0], abs=0.1)


def test_additive_trend_continues_trend():
    frame = _frame(np.arange(1.0, 21.0))
    forecast = holt_winters_forecast(
        frame, "value", steps=3, trend="add", seasonal=None
    )
    assert np.all(forecast > 20.0)
    assert np.all(np.diff(forecast) > 0)


def test_no_trend_no_seasonal_reduces_to_ses():
    frame = _frame(np.arange(1.0, 11.0))
    forecast = holt_winters_forecast(
        frame, "value", steps=3, trend=None, seasonal=None
    )
    assert forecast.tolist() == pytest.approx([10.0, 10.0, 10.0], abs=0.5)


def test_seasonal_pattern_reproduced():
    pattern = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    train = np.tile(pattern, 4)
    frame = _frame(train)
    forecast = holt_winters_forecast(
        frame, "value", steps=6, trend=None,
        seasonal="add", seasonal_period=6,
    )
    assert forecast.tolist() == pytest.approx(pattern.tolist(), abs=0.5)


def test_multiplicative_trend_with_positive_data():
    frame = _frame(
        np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    )
    forecast = holt_winters_forecast(
        frame, "value", steps=2, trend="mul", seasonal=None
    )
    assert forecast.size == 2
    assert np.all(forecast > 128.0)


def test_damped_trend_less_aggressive_than_undamped():
    frame = _frame(np.arange(1.0, 21.0))
    undamped = holt_winters_forecast(
        frame, "value", steps=5, trend="add", seasonal=None,
        damped_trend=False,
    )
    damped = holt_winters_forecast(
        frame, "value", steps=5, trend="add", seasonal=None,
        damped_trend=True,
    )
    assert np.all(damped < undamped)


def test_damped_trend_with_no_trend_raises():
    frame = _frame(np.arange(1.0, 11.0))
    with pytest.raises(ValueError, match="damped_trend requires"):
        holt_winters_forecast(
            frame, "value", steps=3, trend=None, seasonal=None,
            damped_trend=True,
        )


def test_trailing_nan_dropped():
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, np.nan]})
    forecast = holt_winters_forecast(
        frame, "value", steps=2, trend=None, seasonal=None
    )
    assert forecast.size == 2
    assert np.all(np.isfinite(forecast))


def test_rejects_empty_values():
    frame = pd.DataFrame({"value": [np.nan, np.nan, np.nan]})
    with pytest.raises(ValueError, match="at least one observation"):
        holt_winters_forecast(frame, "value", steps=1)


def test_validates_steps():
    frame = _frame(np.arange(1.0, 11.0))
    with pytest.raises(ValueError, match="steps must be at least 1"):
        holt_winters_forecast(frame, "value", steps=0)


def test_validates_target_column():
    frame = _frame(np.arange(1.0, 11.0))
    with pytest.raises(KeyError, match="unknown target column"):
        holt_winters_forecast(frame, "missing", steps=1)


def test_validates_trend_mode():
    frame = _frame(np.arange(1.0, 11.0))
    with pytest.raises(ValueError, match="trend must be"):
        holt_winters_forecast(frame, "value", steps=1, trend="invalid")


def test_validates_seasonal_mode():
    frame = _frame(np.arange(1.0, 33.0))
    with pytest.raises(ValueError, match="seasonal must be"):
        holt_winters_forecast(
            frame, "value", steps=4, seasonal="invalid", seasonal_period=4
        )


def test_seasonal_requires_period():
    frame = _frame(np.arange(1.0, 33.0))
    with pytest.raises(ValueError, match="seasonal_period is required"):
        holt_winters_forecast(frame, "value", steps=4, seasonal="add")


def test_seasonal_validates_period():
    frame = _frame(np.arange(1.0, 33.0))
    with pytest.raises(ValueError, match="at least 2"):
        holt_winters_forecast(
            frame, "value", steps=4, seasonal="add", seasonal_period=1
        )


def test_validates_finiteness():
    frame = pd.DataFrame({"value": [1.0, 2.0, np.inf, 4.0]})
    with pytest.raises(ValueError, match="finite values"):
        holt_winters_forecast(frame, "value", steps=1)


def test_multiplicative_trend_rejects_nonpositive():
    frame = _frame(np.array([-1.0, 2.0, 3.0, 4.0, 5.0]))
    with pytest.raises(ValueError, match="multiplicative trend"):
        holt_winters_forecast(
            frame, "value", steps=2, trend="mul", seasonal=None
        )


def test_multiplicative_seasonal_rejects_nonpositive():
    frame = _frame(np.array([-1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0, 8.0]))
    with pytest.raises(ValueError, match="multiplicative seasonal"):
        holt_winters_forecast(
            frame, "value", steps=4, trend="add",
            seasonal="mul", seasonal_period=4,
        )


def test_rejects_too_few_points_for_seasonal():
    frame = _frame(np.arange(1.0, 5.0))
    with pytest.raises(ValueError, match="two complete seasonal"):
        holt_winters_forecast(
            frame, "value", steps=4, trend=None,
            seasonal="add", seasonal_period=4,
        )


def test_rejects_too_few_points_for_trend():
    frame = _frame([3.0])
    with pytest.raises(ValueError, match="two observations"):
        holt_winters_forecast(
            frame, "value", steps=1, trend="add", seasonal=None
        )
