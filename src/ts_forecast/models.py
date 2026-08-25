"""Time series forecasting models."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def ensemble_forecast(forecasts, weights=None, method="mean"):
    """Blend same-horizon forecasts with mean, median, or trimmed aggregation.

    Parameters
    ----------
    forecasts : sequence of array-like
        Point forecasts from two or more models. Every forecast must have the
        same horizon.
    weights : sequence of float, optional
        Relative model weights. They are normalized internally, so their sum
        need not equal one.
    method : {"mean", "median", "trimmed_mean"}
        Aggregation strategy. Median aggregation is robust to one anomalous
        model forecast and does not accept weights.
    """

    if forecasts is None:
        raise ValueError("at least one forecast is required")
    forecasts = list(forecasts)
    if not forecasts:
        raise ValueError("at least one forecast is required")
    arrays = [np.asarray(forecast, dtype=float).ravel() for forecast in forecasts]
    if any(array.size == 0 for array in arrays):
        raise ValueError("forecasts must not be empty")
    if len({array.size for array in arrays}) != 1:
        raise ValueError("all forecasts must have the same horizon")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("forecasts must contain only finite values")

    if method not in {"mean", "median", "trimmed_mean"}:
        raise ValueError("method must be 'mean', 'median', or 'trimmed_mean'")
    if method == "median":
        if weights is not None:
            raise ValueError("weights are not supported with median aggregation")
        return np.median(np.vstack(arrays), axis=0)
    if method == "trimmed_mean":
        if weights is not None:
            raise ValueError("weights are not supported with trimmed_mean aggregation")
        if len(arrays) < 3:
            raise ValueError("trimmed_mean aggregation requires at least three forecasts")
        ordered = np.sort(np.vstack(arrays), axis=0)
        return np.mean(ordered[1:-1], axis=0)

    if weights is None:
        normalized = np.ones(len(arrays), dtype=float)
    else:
        normalized = np.asarray(weights, dtype=float).ravel()
        if normalized.size != len(arrays):
            raise ValueError("weights must match the number of forecasts")
        if not np.all(np.isfinite(normalized)) or np.any(normalized < 0):
            raise ValueError("weights must be finite and non-negative")
        if normalized.sum() <= 0:
            raise ValueError("at least one weight must be positive")

    return np.average(np.vstack(arrays), axis=0, weights=normalized)


def reconcile_bottom_up(forecasts, structure):
    """Rebuild every hierarchical level from its leaves (bottom-up).

    ``forecasts`` maps node names to point forecasts and ``structure``
    maps aggregate names to their direct children. Leaf forecasts are
    kept unchanged and each aggregate is replaced by the sum of its
    descendants, so totals, regions, and stores become mutually
    consistent. Aggregates named in ``structure`` but absent from
    ``forecasts`` are derived from their children instead of being
    required.
    """

    if not isinstance(forecasts, Mapping):
        raise ValueError("forecasts must be a mapping of node names")
    if not forecasts:
        raise ValueError("forecasts must contain at least one node")
    if not isinstance(structure, Mapping) or not structure:
        raise ValueError("structure must map aggregates to their children")
    arrays = {}
    for name, forecast in forecasts.items():
        array = np.asarray(forecast, dtype=float).ravel()
        if array.size == 0:
            raise ValueError("forecasts must not be empty")
        arrays[name] = array
    if len({array.size for array in arrays.values()}) != 1:
        raise ValueError("all leaf forecasts must have the same horizon")
    for children in structure.values():
        for child in children:
            if child not in arrays and child not in structure:
                raise KeyError(f"unknown node in hierarchy: {child}")

    def total(node, visiting):
        if node in visiting:
            raise ValueError("hierarchy must not contain cycles")
        children = list(structure.get(node, ()))
        if not children:
            return arrays[node]
        return np.sum(
            [total(child, visiting | {node}) for child in children], axis=0
        )

    reconciled = {}
    for name in list(arrays) + [node for node in structure if node not in arrays]:
        reconciled[name] = total(name, frozenset())
    return reconciled


def repair_quantile_crossing(forecasts):
    """Rearrange multi-quantile forecasts until quantiles stop crossing.

    Per-quantile forecasts produced independently routinely cross: some
    horizon ends up with a nominal 0.9 bound below its 0.1 bound, which
    makes interval widths negative and pinball scores misleading.
    Rearrangement sorts each horizon's quantile forecasts into increasing
    order, the unique projection that restores monotonicity while keeping
    every row's original values. ``forecasts`` may be a mapping from
    quantile level to a 1-D forecast, or a 2-D array whose columns follow
    ascending quantile levels. Returns ``(repaired, crossings_fixed)``
    where ``crossings_fixed`` counts adjacent-level violations removed.
    """

    if isinstance(forecasts, Mapping):
        if not forecasts:
            raise ValueError("at least one quantile forecast is required")
        levels = sorted(forecasts)
        if any(not 0.0 < float(level) < 1.0 for level in levels):
            raise ValueError("quantile levels must be strictly between 0 and 1")
        matrix = np.column_stack(
            [np.asarray(forecasts[level], dtype=float).ravel() for level in levels]
        )
    else:
        matrix = np.asarray(forecasts, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("forecasts must be 2-D (horizons x quantiles) or a dict")
    if matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("crossing needs at least two quantile forecasts per horizon")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("forecasts must contain only finite values")

    crossings_fixed = int(np.sum(matrix[:, :-1] > matrix[:, 1:]))
    return np.sort(matrix, axis=1), crossings_fixed


def seasonal_naive_forecast(train, target_col, steps=1, seasonal_period=7):
    """Repeat the most recent observed season for the requested horizon."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if seasonal_period < 1:
        raise ValueError("seasonal_period must be at least 1")
    values = np.asarray(train[target_col].dropna(), dtype=float)
    if len(values) < seasonal_period:
        raise ValueError(
            "training data must contain at least one complete seasonal period"
        )
    season = values[-seasonal_period:]
    return np.resize(season, steps)


def select_season_length(train, target_col, candidates):
    """Pick the candidate period whose lag shows the strongest autocorrelation.

    Every candidate is scored by the autocorrelation of the target series at
    exactly that lag and the strongest score wins; ties go to the shortest
    period so simpler seasonal patterns are preferred. The selected length
    can be handed straight to :func:`seasonal_naive_forecast`.
    """

    if target_col not in train:
        raise KeyError(f"unknown target column: {target_col}")
    try:
        periods = list(candidates)
    except TypeError:
        raise ValueError(
            "candidates must be a non-empty sequence of integer periods"
        ) from None
    if not periods or any(
        isinstance(period, bool) or not isinstance(period, int)
        for period in periods
    ):
        raise ValueError("candidates must be a non-empty sequence of integer periods")
    if any(period < 2 for period in periods):
        raise ValueError("candidate periods must be at least 2")

    values = np.asarray(train[target_col].dropna(), dtype=float)
    if values.size == 0:
        raise ValueError("training data must contain at least one observation")
    if not np.all(np.isfinite(values)):
        raise ValueError("training data must contain only finite values")
    longest = max(periods)
    if values.size <= longest:
        raise ValueError(
            "training data must contain more observations than the longest candidate period"
        )
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError("training data must have non-zero variance")

    best_period = None
    best_score = -np.inf
    for period in sorted(set(periods)):
        score = float(np.dot(centered[period:], centered[:-period])) / denominator
        if score > best_score:
            best_score = score
            best_period = period
    return int(best_period)


def seasonal_naive_auto(train, target_col, steps=1, candidates=(7, 30, 365)):
    """Seasonal-naive forecast using the dominant candidate season length.

    ``select_season_length`` picks the candidate whose lag autocorrelates
    most strongly and :func:`seasonal_naive_forecast` repeats the most
    recent season of that length. The default candidates cover weekly,
    monthly, and annual cycles for daily data; pass a custom sequence to
    search other lags. Every candidate must be shorter than the training
    history.
    """

    period = select_season_length(train, target_col, candidates)
    return seasonal_naive_forecast(
        train, target_col, steps=steps, seasonal_period=period
    )


def _ses_forecast_level(values):
    """Return the fitted simple-exponential-smoothing level of a series."""

    best_sse = np.inf
    best_level = float(values[-1])
    for alpha in np.linspace(0.01, 0.99, 99):
        level = float(values[0])
        sse = 0.0
        for observed in values[1:]:
            sse += (observed - level) ** 2
            level += alpha * (observed - level)
        if sse < best_sse:
            best_sse = sse
            best_level = level
    return best_level


def theta_forecast(train, target_col, steps=1, seasonal_period=None):
    """Forecast with the classical Theta method.

    The series is additively deseasonalized when ``seasonal_period`` is
    given and then split into its two standard theta lines: the
    least-squares linear trend (theta = 0) and the curvature-amplified
    line ``2 * x - trend`` (theta = 2). The trend line is extrapolated
    directly while the theta = 2 line is projected with simple
    exponential smoothing; the two horizon forecasts are averaged and the
    seasonal pattern is reapplied, following Assimakopoulos and
    Nikolopoulos (2000).
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if seasonal_period is not None and seasonal_period < 2:
        raise ValueError("seasonal_period must be at least 2 when provided")
    values = np.asarray(train[target_col].dropna(), dtype=float)
    if values.size < 3:
        raise ValueError("training data must contain at least three observations")
    indices = None
    if seasonal_period is not None:
        if values.size < 2 * seasonal_period:
            raise ValueError(
                "training data must contain at least two complete seasonal periods"
            )
        period = int(seasonal_period)
        cycles = values.size // period
        indices = np.array(
            [values[offset::period][:cycles].mean() for offset in range(period)]
        )
        indices -= indices.mean()
        values = values - indices[np.arange(values.size) % period]

    time = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(time, values, 1)
    trend_line = intercept + slope * time
    theta_two_line = 2.0 * values - trend_line
    level = _ses_forecast_level(theta_two_line)
    horizon = np.arange(1, steps + 1, dtype=float)
    trend_forecast = intercept + slope * (values.size - 1 + horizon)
    forecast = 0.5 * (trend_forecast + level)
    if indices is not None:
        phases = (values.size - 1 + horizon).astype(int) % len(indices)
        forecast = forecast + indices[phases]
    return forecast


def forecast_arima(train, target_col, order=(1, 1, 1), steps=1):
    model = ARIMA(train[target_col].dropna(), order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    return forecast


def evaluate_forecast(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def walk_forward_validation(model_fn, df, target_col, window=30, steps=1):
    predictions = []
    actuals = []
    n = len(df)
    for i in range(window, n - steps + 1):
        train_window = df.iloc[i - window : i]
        pred = model_fn(train_window, target_col, steps=steps)
        actual = df.iloc[i : i + steps][target_col].values
        predictions.append(pred)
        actuals.append(actual)
    return np.array(actuals), np.array(predictions)


def rolling_origin_backtest(
    model_fn,
    df,
    target_col,
    *,
    initial_window,
    horizon=1,
    step=1,
    window=None,
    gap=0,
    max_folds=None,
    origins=None,
):
    """Evaluate a forecasting callable over reproducible rolling origins.

    ``model_fn`` receives ``(training_frame, target_col, steps=horizon)``.
    By default the training set expands; setting ``window`` keeps only the most
    recent observations at each origin. The tidy result preserves timestamps
    and horizon numbers for horizon-specific analysis. ``gap`` leaves the
    observations immediately before each test origin unused, which prevents
    look-ahead when a workflow has a reporting or label delay.
    """

    if target_col not in df:
        raise KeyError(f"unknown target column: {target_col}")
    if initial_window < 1 or horizon < 1 or step < 1:
        raise ValueError("initial_window, horizon, and step must be at least 1")
    if gap < 0:
        raise ValueError("gap must be non-negative")
    if window is not None and window < 1:
        raise ValueError("window must be at least 1 when provided")
    if max_folds is not None and (
        isinstance(max_folds, bool) or not isinstance(max_folds, int) or max_folds < 1
    ):
        raise ValueError("max_folds must be a positive integer or None")
    if initial_window + gap + horizon > len(df):
        raise ValueError("not enough observations for one backtest fold")

    records = []
    fold = 0
    first_origin = initial_window + gap
    if origins is None:
        selected_origins = list(range(first_origin, len(df) - horizon + 1, step))
    else:
        selected_origins = list(origins)
        if not selected_origins or any(
            isinstance(origin, bool) or not isinstance(origin, int)
            for origin in selected_origins
        ):
            raise ValueError("origins must be a non-empty sequence of integer positions")
        if selected_origins != sorted(set(selected_origins)):
            raise ValueError("origins must be unique and sorted in ascending order")
        if any(
            origin < first_origin or origin + horizon > len(df)
            for origin in selected_origins
        ):
            raise ValueError("origins must leave a complete training and forecast window")
    for origin in selected_origins:
        if max_folds is not None and fold >= max_folds:
            break
        train_end = origin - gap
        start = 0 if window is None else max(0, train_end - window)
        train = df.iloc[start:train_end]
        prediction = np.asarray(
            model_fn(train, target_col, steps=horizon), dtype=float
        ).ravel()
        if prediction.size != horizon:
            raise ValueError("model_fn must return exactly horizon predictions")
        actual = np.asarray(df.iloc[origin : origin + horizon][target_col], dtype=float)
        timestamps = df.index[origin : origin + horizon]
        fold += 1
        for offset, (timestamp, observed, predicted) in enumerate(
            zip(timestamps, actual, prediction), start=1
        ):
            records.append(
                {
                    "fold": fold,
                    "origin": df.index[train_end - 1],
                    "timestamp": timestamp,
                    "horizon": offset,
                    "actual": float(observed),
                    "prediction": float(predicted),
                }
            )
    return pd.DataFrame.from_records(records)
