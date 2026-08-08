"""Data generation utilities for testing and demos."""

import numpy as np
import pandas as pd


def generate_synthetic_series(n_points=500, trend=0.01, seasonal_amplitude=5, seasonal_period=30, noise_std=1.0, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)
    trend_component = trend * t
    seasonal_component = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
    noise = rng.normal(0, noise_std, n_points)
    values = trend_component + seasonal_component + noise
    dates = pd.date_range("2024-01-01", periods=n_points, freq="D")
    return pd.DataFrame({"value": values}, index=dates)


def add_structural_break(df, target_col, break_point, shift=20):
    result = df.copy()
    result.loc[break_point:, target_col] += shift
    return result


def generate_multivariate_series(n_series=3, n_points=500, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_points, freq="D")
    data = {}
    for i in range(n_series):
        trend = 0.005 * (i + 1)
        t = np.arange(n_points)
        data[f"series_{i}"] = trend * t + rng.normal(0, 1, n_points).cumsum() * 0.5
    df = pd.DataFrame(data, index=dates)
    df["target"] = df["series_0"] * 0.5 + df["series_1"] * 0.3 + df["series_2"] * 0.2
    return df
