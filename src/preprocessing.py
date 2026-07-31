"""Time series preprocessing: lag features, rolling stats, temporal splits."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_lag_features(df: pd.DataFrame, col: str, lags: list[int] | None = None) -> pd.DataFrame:
    df = df.copy()
    lags = lags or [1, 2, 3, 7, 14, 30]
    for lag in lags:
        df[f"lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows: list[int] | None = None) -> pd.DataFrame:
    df = df.copy()
    windows = windows or [7, 14, 30]
    for window in windows:
        df[f"rolling_mean_{window}"] = df[col].rolling(window).mean()
        df[f"rolling_std_{window}"] = df[col].rolling(window).std()
    return df


def add_datetime_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    df["day_of_week"] = dates.dt.dayofweek
    df["day_of_month"] = dates.dt.day
    df["month"] = dates.dt.month
    df["day_of_year"] = dates.dt.dayofyear
    df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    return df


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.8):
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def create_supervised_data(
    df: pd.DataFrame, target: str = "value", lags: list[int] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    df = add_lag_features(df, target, lags)
    df = df.dropna()
    X = df.drop(columns=[target, "date"])
    y = df[target]
    return X, y
