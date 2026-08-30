"""Tests for robust time-series outlier detection."""

import numpy as np
import pytest

from ts_forecast.preprocessing import detect_outliers


def test_seasonal_detection_flags_only_the_injected_spike():
    n, period = 48, 4
    trend = 0.3 * np.arange(n, dtype=float)
    pattern = np.zeros(period)
    pattern[1] = 1.0
    pattern[3] = -1.0
    seasonal = np.tile(pattern, n // period)
    values = trend + seasonal
    values[20] += 20.0

    result = detect_outliers(values, seasonal_period=period)

    assert set(result) == {"outlier", "score"}
    assert result["outlier"].sum() == 1
    assert result["outlier"][20]


def test_seasonal_detection_leaves_clean_series_unflagged():
    n, period = 60, 5
    rng = np.random.default_rng(7)
    values = (
        0.4 * np.arange(n, dtype=float)
        + np.tile(np.sin(np.arange(period) * 2 * np.pi / period), n // period)
        + rng.normal(scale=0.02, size=n)
    )

    result = detect_outliers(values, seasonal_period=period)

    assert result["outlier"].sum() == 0


def test_rolling_detection_follows_a_linear_trend():
    n = 60
    values = 3.0 + 0.5 * np.arange(n, dtype=float)
    values[30] += 8.0

    result = detect_outliers(values, window=5)

    assert result["outlier"][30]
    assert result["outlier"].sum() == 1


def test_rolling_detection_leaves_edges_unflagged():
    values = np.arange(20, dtype=float)
    values[10] += 50.0

    result = detect_outliers(values, window=5)

    assert result["outlier"][10]
    assert not result["outlier"][0]
    assert not result["outlier"][-1]
    assert np.isnan(result["score"][0])
    assert np.isnan(result["score"][-1])


def test_rolling_detection_flags_clean_series_as_empty():
    values = 3.0 + 0.5 * np.arange(40, dtype=float)

    result = detect_outliers(values, window=5)

    assert result["outlier"].sum() == 0
    assert not result["outlier"].any()


def test_threshold_controls_sensitivity():
    values = 3.0 + 0.5 * np.arange(40, dtype=float)
    values[20] += 4.0

    strict = detect_outliers(values, window=5, threshold=1.0)
    lax = detect_outliers(values, window=5, threshold=10.0)

    assert strict["outlier"][20]
    assert not lax["outlier"][20]


def test_validation_errors():
    with pytest.raises(ValueError, match="not be empty"):
        detect_outliers([])
    with pytest.raises(ValueError, match="finite"):
        detect_outliers([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="threshold"):
        detect_outliers(np.arange(10.0), threshold=0)
    with pytest.raises(ValueError, match="threshold"):
        detect_outliers(np.arange(10.0), threshold=True)
    with pytest.raises(ValueError, match="window"):
        detect_outliers(np.arange(10.0), window=2)
    with pytest.raises(ValueError, match="window"):
        detect_outliers(np.arange(10.0), window=11)
    with pytest.raises(ValueError, match="seasonal_period"):
        detect_outliers(np.arange(20.0), seasonal_period=1)
    with pytest.raises(ValueError, match="two complete periods"):
        detect_outliers(np.arange(10.0), seasonal_period=7)
