"""Tests for the forecast_accuracy convenience scorer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ts_forecast.evaluation import forecast_accuracy


class TestForecastAccuracy:
    def test_reports_standard_point_forecast_metrics(self):
        y_true = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        y_pred = np.array([11.0, 13.0, 15.0, 17.0, 19.0])

        metrics = forecast_accuracy(y_true, y_pred)

        assert set(metrics) == {
            "mae", "rmse", "mape", "me", "smape", "bias", "count", "mase",
            "rmse_scaled",
        }
        assert metrics["me"] == pytest.approx(-1.0)
        assert metrics["mae"] == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(1.0)
        assert metrics["bias"] == pytest.approx(1.0)
        assert metrics["count"] == 5

    def test_perfect_forecast_has_zero_error(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.copy()

        metrics = forecast_accuracy(y_true, y_pred)

        assert metrics["me"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["bias"] == 0.0

    def test_omits_scale_free_metrics_without_training(self):
        metrics = forecast_accuracy([1.0, 2.0], [1.5, 2.5])

        assert metrics["mase"] is None
        assert metrics["rmse_scaled"] is None

    def test_computes_scale_free_metrics_with_training(self):
        y_true = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        y_pred = np.array([11.0, 13.0, 15.0, 17.0, 19.0])
        y_train = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

        metrics = forecast_accuracy(y_true, y_pred, y_train=y_train, seasonal_period=1)

        # Seasonal-naive (lag-1) scale of the training series is 2.0.
        assert metrics["mase"] == pytest.approx(0.5)
        assert metrics["rmse_scaled"] == pytest.approx(0.5)

    def test_mase_uses_seasonal_scale(self):
        y_true = np.array([13.0, 15.0, 17.0, 19.0])
        y_pred = np.array([14.0, 16.0, 18.0, 20.0])
        y_train = np.array([1.0, 2.0, 3.0, 11.0, 12.0, 13.0, 21.0, 22.0, 23.0])

        metrics = forecast_accuracy(y_true, y_pred, y_train=y_train, seasonal_period=3)

        # lag-3 differences of the training series are all 10.
        assert metrics["mase"] == pytest.approx(0.1)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            forecast_accuracy([1.0, 2.0], [1.0])

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="observation"):
            forecast_accuracy([], [])

    def test_rejects_non_finite_values(self):
        with pytest.raises(ValueError, match="finite"):
            forecast_accuracy([1.0, np.nan], [1.0, 2.0])

    def test_rejects_bad_seasonal_period(self):
        with pytest.raises(ValueError, match="strictly between|at least 1"):
            forecast_accuracy([1.0], [1.0], seasonal_period=0)
        with pytest.raises(TypeError, match="integer"):
            forecast_accuracy([1.0], [1.0], seasonal_period=1.5)

    def test_rejects_short_training_series(self):
        with pytest.raises(ValueError, match="seasonal period"):
            forecast_accuracy([1.0], [1.0], y_train=[5.0, 6.0], seasonal_period=2)
