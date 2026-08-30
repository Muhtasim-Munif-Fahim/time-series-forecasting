"""Data loading and preprocessing for time series datasets."""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import boxcox as _scipy_boxcox

from ts_forecast.evaluation import seasonal_decompose


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


def impute_missing(values, method="forward", seasonal_period=None):
    """Fill missing values in a time series with a chosen imputation rule.

    ``method`` selects the strategy: ``"forward"`` carries the last
    observation forward, ``"backward"`` carries the next observation
    backward, ``"linear"`` interpolates each gap linearly between the
    surrounding finite values, and ``"seasonal"`` replaces every missing
    observation with the mean of the finite values sharing its seasonal
    phase (a missing Monday is filled with the average Monday), which
    requires ``seasonal_period``. Forward, backward, and linear imputation
    leave edge gaps unfilled because no value exists on one side of them;
    seasonal imputation covers the whole series since the phase means use
    every finite observation. A series without missing values is returned
    unchanged.
    """

    observed = np.asarray(values, dtype=float).ravel()
    if observed.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isnan(observed) | np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers or NaN gaps")
    if np.all(np.isnan(observed)):
        raise ValueError(
            "values must contain at least one non-missing observation"
        )
    if method not in {"forward", "backward", "linear", "seasonal"}:
        raise ValueError(
            "method must be 'forward', 'backward', 'linear', or 'seasonal'"
        )
    if method == "seasonal":
        if seasonal_period is None:
            raise ValueError("seasonal imputation requires seasonal_period")
        if (
            isinstance(seasonal_period, bool)
            or not isinstance(seasonal_period, int)
            or seasonal_period < 2
        ):
            raise ValueError("seasonal_period must be an integer of at least 2")
    elif seasonal_period is not None:
        raise ValueError("seasonal_period is only used with method='seasonal'")

    result = observed.copy()
    missing = np.isnan(observed)
    if not np.any(missing):
        return result

    if method == "forward":
        last = np.nan
        for i in range(observed.size):
            if not np.isnan(observed[i]):
                last = observed[i]
            else:
                result[i] = last
        return result

    if method == "backward":
        next_value = np.nan
        for i in range(observed.size - 1, -1, -1):
            if not np.isnan(observed[i]):
                next_value = observed[i]
            else:
                result[i] = next_value
        return result

    if method == "linear":
        for start, end in _nan_runs(observed):
            if start == 0 or end == observed.size:
                continue
            left = observed[start - 1]
            right = observed[end]
            fill = np.linspace(left, right, end - start + 2)[1:-1]
            result[start:end] = fill
        return result

    period = int(seasonal_period)
    finite = ~np.isnan(observed)
    positions = np.arange(observed.size)
    phase_means = np.array(
        [
            float(np.mean(observed[(positions % period == phase) & finite]))
            for phase in range(period)
        ]
    )
    global_mean = float(np.mean(observed[finite]))
    phase_means = np.where(np.isnan(phase_means), global_mean, phase_means)
    result[missing] = phase_means[positions[missing] % period]
    return result


def _nan_runs(observed):
    """Yield (start, end) half-open index ranges of consecutive NaN values."""

    size = observed.size
    index = 0
    while index < size:
        if np.isnan(observed[index]):
            start = index
            while index < size and np.isnan(observed[index]):
                index += 1
            yield start, index
        else:
            index += 1


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


def detect_outliers(values, seasonal_period=None, window=7, threshold=3.0):
    """Flag point anomalies that exceed a robust scale of the local deviation.

    Two modes cover trended and seasonal series. With ``seasonal_period``
    the series is classically decomposed and the residual (observed minus
    trend minus season) is scored, so both a smooth trend and a repeating
    seasonal pattern are removed before flagging; without it the deviation
    from a centered rolling median of width ``window`` plays that role.
    Every score divides its deviation by a robust scale built from the
    median absolute deviation, falling back to the standard deviation when
    the MAD is zero, and points whose score exceeds ``threshold`` are
    flagged. Scores are ``nan`` where neither the rolling median nor the
    decomposition trend is defined (the series edges), and those points are
    never flagged. A series with constant local structure yields zero
    outliers.
    """

    observed = np.asarray(values, dtype=float).ravel()
    if observed.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or threshold <= 0
    ):
        raise ValueError("threshold must be a positive number")
    if isinstance(window, bool) or not isinstance(window, int) or window < 3:
        raise ValueError("window must be an integer of at least 3")

    scores = np.full(observed.size, np.nan)

    if seasonal_period is not None:
        if (
            isinstance(seasonal_period, bool)
            or not isinstance(seasonal_period, int)
            or seasonal_period < 2
        ):
            raise ValueError("seasonal_period must be an integer of at least 2")
        period = int(seasonal_period)
        if observed.size < 2 * period:
            raise ValueError(
                "seasonal outlier detection requires at least two complete periods"
            )
        residual = seasonal_decompose(observed, period)["residual"]
        finite = np.isfinite(residual)
        center = float(np.median(residual[finite]))
        mad = float(np.median(np.abs(residual[finite] - center)))
        scale = 1.4826 * mad if mad > 0 else float(np.std(residual[finite], ddof=1))
        if scale > 0:
            scores[finite] = (residual[finite] - center) / scale
    else:
        if window > observed.size:
            raise ValueError("window must not exceed the number of observations")
        rolling = (
            pd.Series(observed).rolling(window, center=True).median().to_numpy()
        )
        finite = np.isfinite(rolling)
        deviation = observed[finite] - rolling[finite]
        center = float(np.median(deviation))
        mad = float(np.median(np.abs(deviation - center)))
        scale = 1.4826 * mad if mad > 0 else float(np.std(deviation, ddof=1))
        if scale > 0:
            scores[finite] = (deviation - center) / scale

    return {
        "outlier": scores > threshold,
        "score": scores,
    }
