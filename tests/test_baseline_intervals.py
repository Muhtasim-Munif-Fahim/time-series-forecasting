"""Tests for prediction intervals around baseline forecasts."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from ts_forecast.models import baseline_prediction_interval, seasonal_naive_forecast


def _frame(values):
    return pd.DataFrame({"value": values})


def test_naive_intervals_collapse_when_errors_are_zero():
    frame = _frame([0.0, 1.0, 2.0, 3.0, 4.0])
    result = baseline_prediction_interval(frame, "value", steps=3, method="naive")

    np.testing.assert_allclose(result["forecast"], [4.0, 4.0, 4.0])
    np.testing.assert_allclose(result["lower"], [4.0, 4.0, 4.0])
    np.testing.assert_allclose(result["upper"], [4.0, 4.0, 4.0])


def test_naive_intervals_widen_with_horizon():
    frame = _frame([0.0, 1.0, 0.0, 1.0, 0.0])
    result = baseline_prediction_interval(frame, "value", steps=4, method="naive")

    residuals = np.array([1.0, -1.0, 1.0, -1.0])
    sigma = residuals.std(ddof=1)
    half = norm.ppf(0.95) * sigma * np.sqrt(np.arange(1, 5))
    np.testing.assert_allclose(result["lower"], -half)
    np.testing.assert_allclose(result["upper"], half)
    widths = result["upper"] - result["lower"]
    assert widths[2] > widths[0]


def test_naive_interval_respects_coverage_level():
    frame = _frame([1.0, 3.0, 2.0, 4.0, 3.0])
    narrow = baseline_prediction_interval(frame, "value", steps=2, coverage=0.5)
    wide = baseline_prediction_interval(frame, "value", steps=2, coverage=0.9)
    assert (wide["upper"] - wide["lower"] > narrow["upper"] - narrow["lower"]).all()


def test_seasonal_naive_interval_matches_repeated_season():
    frame = _frame([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    result = baseline_prediction_interval(
        frame, "value", steps=5, method="seasonal_naive", seasonal_period=2
    )
    expected = seasonal_naive_forecast(frame, "value", steps=5, seasonal_period=2)

    np.testing.assert_allclose(result["forecast"], expected)
    np.testing.assert_allclose(result["lower"], expected)
    np.testing.assert_allclose(result["upper"], expected)


def test_seasonal_naive_interval_scales_within_season():
    frame = _frame([1.0, 2.0, 3.0, 1.5, 2.5, 4.0, 1.0, 3.0, 3.0])
    result = baseline_prediction_interval(
        frame, "value", steps=5, method="seasonal_naive", seasonal_period=3
    )
    widths = result["upper"] - result["lower"]

    np.testing.assert_allclose(widths[:3], widths[0], rtol=1e-10)
    np.testing.assert_allclose(widths[3:], widths[3], rtol=1e-10)
    assert widths[3] > widths[0]


def test_drift_interval_extrapolates_the_linear_slope():
    frame = _frame([0.0, 2.0, 4.0, 6.0, 8.0])
    result = baseline_prediction_interval(frame, "value", steps=3, method="drift")

    np.testing.assert_allclose(result["forecast"], [10.0, 12.0, 14.0])
    np.testing.assert_allclose(result["lower"], [10.0, 12.0, 14.0])
    np.testing.assert_allclose(result["upper"], [10.0, 12.0, 14.0])


def test_drift_intervals_widen_with_horizon_and_history():
    frame = _frame([0.0, 1.0, 3.0, 4.0, 8.0, 9.0])
    result = baseline_prediction_interval(frame, "value", steps=3, method="drift")

    widths = result["upper"] - result["lower"]
    assert widths[2] > widths[1] > widths[0]


def test_validation_errors():
    frame = _frame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    with pytest.raises(ValueError, match="steps"):
        baseline_prediction_interval(frame, "value", steps=0)
    with pytest.raises(ValueError, match="method"):
        baseline_prediction_interval(frame, "value", steps=1, method="sma")
    with pytest.raises(ValueError, match="coverage"):
        baseline_prediction_interval(frame, "value", steps=1, coverage=1.0)
    with pytest.raises(KeyError, match="unknown target"):
        baseline_prediction_interval(frame, "missing", steps=1)
    with pytest.raises(ValueError, match="seasonal_period"):
        baseline_prediction_interval(
            frame, "value", steps=1, method="seasonal_naive"
        )
    with pytest.raises(ValueError, match="seasonal_period"):
        baseline_prediction_interval(
            frame, "value", steps=1, method="seasonal_naive", seasonal_period=1
        )
    with pytest.raises(ValueError, match="two complete seasonal periods"):
        baseline_prediction_interval(
            frame, "value", steps=1, method="seasonal_naive", seasonal_period=7
        )
    with pytest.raises(ValueError, match="only used with method"):
        baseline_prediction_interval(
            frame, "value", steps=1, method="naive", seasonal_period=7
        )
    with pytest.raises(ValueError, match="finite"):
        baseline_prediction_interval(
            _frame([1.0, np.inf, 3.0, 4.0, 5.0, 6.0]), "value", steps=1
        )
