"""Synthetic time series data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_time_series(
    n_points: int = 500,
    trend: float = 0.05,
    seasonality: float = 1.0,
    noise: float = 1.0,
    start_date: str = "2023-01-01",
    freq: str = "D",
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    t = np.arange(n_points)

    trend_component = trend * t
    seasonal_component = seasonality * np.sin(2 * np.pi * t / 30.0)
    noise_component = noise * rng.standard_normal(n_points)

    values = trend_component + seasonal_component + noise_component

    dates = pd.date_range(start=start_date, periods=n_points, freq=freq)
    df = pd.DataFrame({"date": dates, "value": values.round(4)})
    df["value"] = df["value"] - df["value"].min() + 10
    return df


def generate_regime_series(
    n_points: int = 500, random_state: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    values = np.concatenate([
        rng.normal(50, 5, n_points // 3),
        rng.normal(80, 8, n_points // 3),
        rng.normal(30, 4, n_points - 2 * (n_points // 3)),
    ])
    dates = pd.date_range(start="2023-01-01", periods=n_points, freq="D")
    return pd.DataFrame({"date": dates, "value": values.round(4)})
