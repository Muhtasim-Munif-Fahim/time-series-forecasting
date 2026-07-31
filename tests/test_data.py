"""Tests for data generation and preprocessing."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_generation import generate_time_series, generate_regime_series
from preprocessing import add_lag_features, add_datetime_features, temporal_split


class TestDataGeneration:
    def test_generates_expected_length(self):
        df = generate_time_series(n_points=100)
        assert len(df) == 100

    def test_has_required_columns(self):
        df = generate_time_series(n_points=50)
        assert "date" in df.columns
        assert "value" in df.columns

    def test_regime_series(self):
        df = generate_regime_series(n_points=100)
        assert len(df) == 100


class TestPreprocessing:
    def test_lag_features(self):
        df = generate_time_series(n_points=100)
        result = add_lag_features(df, "value", lags=[1, 2])
        assert "lag_1" in result.columns
        assert "lag_2" in result.columns

    def test_datetime_features(self):
        df = generate_time_series(n_points=100)
        result = add_datetime_features(df)
        assert "day_of_week" in result.columns
        assert "month" in result.columns

    def test_temporal_split(self):
        df = generate_time_series(n_points=100)
        train, test = temporal_split(df, train_ratio=0.8)
        assert len(train) == 80
        assert len(test) == 20
