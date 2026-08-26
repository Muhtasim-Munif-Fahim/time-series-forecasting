"""Tests for the Naive2 deseasonalized baseline forecaster."""

import numpy as np
import pandas as pd
import pytest

from ts_forecast.models import naive2_forecast


def _frame(values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)})


def test_without_seasonal_period_reduces_to_naive():
    frame = _frame([3.0, 7.0, 2.0])
    forecast = naive2_forecast(frame, "value", steps=3)
    assert isinstance(forecast, np.ndarray)
    assert forecast.tolist() == [2.0, 2.0, 2.0]


def test_deseasonalizes_and_reapplies_the_seasonal_pattern():
    frame = _frame([4.0, 6.0, 8.0, 10.0])
    forecast = naive2_forecast(frame, "value", steps=2, seasonal_period=2)
    assert forecast.tolist() == pytest.approx([8.0, 10.0])


def test_horizon_wraps_around_the_seasonal_pattern():
    frame = _frame([4.0, 6.0, 8.0, 10.0])
    forecast = naive2_forecast(frame, "value", steps=4, seasonal_period=2)
    assert forecast.tolist() == pytest.approx([8.0, 10.0, 8.0, 10.0])


def test_constant_series_stays_constant_under_any_period():
    frame = _frame(np.full(9, 5.0))
    forecast = naive2_forecast(frame, "value", steps=3, seasonal_period=3)
    assert forecast.tolist() == [5.0, 5.0, 5.0]


def test_trailing_missing_values_are_dropped():
    frame = pd.DataFrame({"value": [4.0, 6.0, 8.0, 10.0, np.nan]})
    forecast = naive2_forecast(frame, "value", steps=1, seasonal_period=2)
    assert forecast.tolist() == pytest.approx([8.0])


def test_validates_inputs():
    frame = _frame([4.0, 6.0, 8.0, 10.0])

    with pytest.raises(ValueError, match="steps must be at least 1"):
        naive2_forecast(frame, "value", steps=0)

    with pytest.raises(ValueError, match="at least 2 when provided"):
        naive2_forecast(frame, "value", seasonal_period=1)

    with pytest.raises(ValueError, match="complete seasonal period"):
        naive2_forecast(_frame([1.0, 2.0]), "value", seasonal_period=3)

    with pytest.raises(ValueError, match="only finite values"):
        gapped = _frame([1.0, np.inf, 3.0, 4.0])
        naive2_forecast(gapped, "value")

    with pytest.raises(ValueError, match="at least one observation"):
        naive2_forecast(_frame([np.nan]), "value")
