"""Data loading and preprocessing for time series datasets."""

import pandas as pd
import numpy as np
from pathlib import Path


def load_csv(path, date_col, target_col, freq=None):
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)
    if freq:
        df = df.asfreq(freq)
    return df


def train_test_split(df, target_col, test_size=0.2):
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def add_lag_features(df, columns, lags=(1, 2, 3, 7, 14)):
    result = df.copy()
    for col in columns:
        for lag in lags:
            result[f"{col}_lag_{lag}"] = result[col].shift(lag)
    return result


def add_rolling_features(df, columns, windows=(7, 14, 30)):
    result = df.copy()
    for col in columns:
        for w in windows:
            result[f"{col}_rolling_mean_{w}"] = result[col].rolling(w).mean()
            result[f"{col}_rolling_std_{w}"] = result[col].rolling(w).std()
    return result


def add_calendar_features(df):
    result = df.copy()
    result["dayofweek"] = result.index.dayofweek
    result["month"] = result.index.month
    result["quarter"] = result.index.quarter
    result["dayofyear"] = result.index.dayofyear
    result["is_weekend"] = (result.index.dayofweek >= 5).astype(int)
    return result


def drop_na_features(df):
    return df.dropna()
