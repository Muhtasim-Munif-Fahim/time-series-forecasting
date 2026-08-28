"""Data loading and preprocessing for time series datasets."""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import boxcox as _scipy_boxcox


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


def boxcox_transform(values, lmbda=None):
    """Apply the Box-Cox power transform, estimating lambda when omitted.

    Positive data is mapped through ``(x**lmbda - 1) / lmbda`` for non-zero
    lambda and through ``log(x)`` for lambda zero, which stabilizes variance
    and reduces skewness before modelling. When ``lmbda`` is omitted the
    maximum-likelihood value is fitted from ``values``, so the transform
    adapts to the data instead of requiring a hand-picked power. The input
    must be strictly positive (Box-Cox is undefined for zero and negative
    observations) and non-constant.

    Returns ``(transformed, lmbda)``; pass both to :func:`inverse_boxcox`
    to map fitted values and forecasts back to the original scale.
    """

    observed = np.asarray(values, dtype=float).ravel()
    if observed.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")
    if np.any(observed <= 0):
        raise ValueError("Box-Cox transform requires strictly positive values")
    if np.all(observed == observed[0]):
        raise ValueError("values must not be constant")

    if lmbda is None:
        fitted = float(_scipy_boxcox(observed)[1])
    else:
        fitted = float(lmbda)
        if not np.isfinite(fitted):
            raise ValueError("lmbda must be finite")

    if fitted == 0.0:
        transformed = np.log(observed)
    else:
        transformed = (np.power(observed, fitted) - 1.0) / fitted
    return transformed, fitted


def inverse_boxcox(transformed, lmbda):
    """Undo a Box-Cox transform given the lambda it was fitted with.

    For non-zero lambda the inverse is ``(lmbda * y + 1) ** (1 / lmbda)``
    and for lambda zero it is ``exp(y)``. The inverse is only defined where
    ``lmbda * y + 1`` is positive, so a transformed value outside that domain
    raises instead of silently returning NaN.
    """

    values = np.asarray(transformed, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("transformed values must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("transformed values must contain only finite numbers")
    if not np.isfinite(lmbda):
        raise ValueError("lmbda must be finite")

    if lmbda == 0.0:
        return np.exp(values)
    argument = lmbda * values + 1.0
    if np.any(argument <= 0):
        raise ValueError(
            "inverse Box-Cox is undefined for transformed values outside its domain"
        )
    return np.power(argument, 1.0 / lmbda)


def boxcox_forecast(train, target_col, steps, model_fn, lmbda=None):
    """Forecast through a fitted Box-Cox transform and back-transform the result.

    ``model_fn`` receives ``(transformed_frame, target_col, steps=steps)``
    exactly like the callables accepted by :func:`ts_forecast.models.rolling_origin_backtest`.
    The training target is transformed with a fitted (or given) lambda, the
    model forecasts on the transformed frame, and the forecast is mapped back
    to the original scale, so multiplicative patterns can be modelled on the
    variance-stabilized scale without leaking transformed values into the
    reported output.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if target_col not in train:
        raise KeyError(f"unknown target column: {target_col}")
    if not callable(model_fn):
        raise ValueError("model_fn must be callable")

    transformed_values, fitted = boxcox_transform(train[target_col], lmbda=lmbda)
    transformed_frame = train.copy()
    transformed_frame[target_col] = transformed_values

    prediction = np.asarray(
        model_fn(transformed_frame, target_col, steps=steps), dtype=float
    ).ravel()
    if prediction.size != steps:
        raise ValueError("model_fn must return exactly steps predictions")
    return inverse_boxcox(prediction, fitted)
