import numpy as np
import pytest

from ts_forecast.evaluation import interval_score, mean_scaled_interval_score


def test_msis_matches_interval_score_over_scale():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.5, 1.5, 2.5, 3.5])
    upper = np.array([1.5, 2.5, 3.5, 4.5])
    y_train = np.arange(1.0, 21.0)  # seasonal naive scale with m=1 is 1.0
    detail = interval_score(y_true, lower, upper, coverage=0.9)
    out = mean_scaled_interval_score(
        y_true, lower, upper, y_train, coverage=0.9, seasonal_period=1
    )
    assert out["scale"] == pytest.approx(1.0)
    assert out["msis"] == pytest.approx(detail["score"] / 1.0)
    assert out["interval_score"] == pytest.approx(detail["score"])


def test_msis_none_on_constant_train():
    y_true = np.array([1.0, 1.0, 1.0])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([2.0, 2.0, 2.0])
    y_train = np.ones(10)
    out = mean_scaled_interval_score(y_true, lower, upper, y_train, seasonal_period=1)
    assert out["msis"] is None
    assert out["scale"] == 0.0


def test_msis_rejects_bad_inputs():
    with pytest.raises(ValueError):
        mean_scaled_interval_score([1], [0], [2], [1], seasonal_period=1)
    with pytest.raises(ValueError):
        mean_scaled_interval_score([1, 2], [0, 3], [2, 1], np.arange(5.0))
