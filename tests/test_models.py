"""Tests for forecasting models."""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_generation import generate_time_series
from preprocessing import create_supervised_data, temporal_split
from models import naive_forecast, moving_average_forecast, compute_metrics, linear_regression_forecast


class TestBaselines:
    def test_naive_forecast(self):
        series = np.array([1.0, 2.0, 3.0, 4.0])
        preds = naive_forecast(pd.Series(series), horizon=3)
        assert len(preds) == 3
        assert np.all(preds == 4.0)

    def test_moving_average_forecast(self):
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        preds = moving_average_forecast(pd.Series(series), window=3, horizon=2)
        assert len(preds) == 2
        assert np.isclose(preds[0], 6.0)


class TestMetrics:
    def test_compute_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0


class TestMLModels:
    def test_linear_regression_predicts(self):
        df = generate_time_series(n_points=100)
        X, y = create_supervised_data(df, target="value", lags=[1, 2])
        split = int(len(X) * 0.8)
        preds, model = linear_regression_forecast(X.iloc[:split], y.iloc[:split], X.iloc[split:])
        assert len(preds) == len(X) - split
