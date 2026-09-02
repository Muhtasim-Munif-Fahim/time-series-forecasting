"""Evaluation utilities for time series forecasting."""

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def conformal_prediction_interval(
    calibration_true,
    calibration_pred,
    forecast,
    coverage=0.9,
):
    """Build split-conformal intervals from held-out absolute residuals.

    The finite-sample corrected quantile provides marginal coverage under the
    usual exchangeability assumption. Calibration observations with non-finite
    values are ignored.
    """

    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between 0 and 1")
    observed = np.asarray(calibration_true, dtype=float).ravel()
    predicted = np.asarray(calibration_pred, dtype=float).ravel()
    if observed.shape != predicted.shape:
        raise ValueError("calibration_true and calibration_pred must have equal length")

    finite = np.isfinite(observed) & np.isfinite(predicted)
    scores = np.abs(observed[finite] - predicted[finite])
    if scores.size == 0:
        raise ValueError("at least one finite calibration residual is required")

    quantile_level = min(
        1.0,
        np.ceil((scores.size + 1) * coverage) / scores.size,
    )
    radius = float(np.quantile(scores, quantile_level, method="higher"))
    point_forecast = np.asarray(forecast, dtype=float)
    return point_forecast - radius, point_forecast + radius


def calibrate_conformal_radii(calibration_true, calibration_pred, radii, coverage=0.9):
    """Return the scale factor aligning model radii with nominal coverage.

    Models that emit per-observation uncertainty estimates (residual
    scales, spread forecasts) still need a global correction before the
    resulting intervals are calibrated. This finds the smallest factor
    such that ``abs(true - pred) <= factor * radius`` holds on at least
    the finite-sample target share of calibration pairs, using the same
    quantile rule as :func:`conformal_prediction_interval`. Multiply the
    raw radii by the returned factor when building forecast intervals.
    """

    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between 0 and 1")
    observed = np.asarray(calibration_true, dtype=float).ravel()
    predicted = np.asarray(calibration_pred, dtype=float).ravel()
    widths = np.asarray(radii, dtype=float).ravel()
    if not (observed.shape == predicted.shape == widths.shape):
        raise ValueError(
            "calibration_true, calibration_pred, and radii must have equal length"
        )

    finite = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(widths)
    usable_widths = widths[finite]
    if usable_widths.size == 0:
        raise ValueError("at least one finite calibration observation is required")
    if np.any(usable_widths <= 0):
        raise ValueError("radii must be strictly positive")
    scores = np.abs(observed[finite] - predicted[finite]) / usable_widths
    quantile_level = min(
        1.0,
        np.ceil((scores.size + 1) * coverage) / scores.size,
    )
    return float(np.quantile(scores, quantile_level, method="higher"))


def compute_metrics(y_true, y_pred, prefix=""):
    return {
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        f"{prefix}mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def forecast_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Return sMAPE as a percentage while handling jointly-zero observations."""

    observed = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    if observed.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have equal length")
    if observed.size == 0:
        raise ValueError("at least one forecast is required")
    if not np.all(np.isfinite(np.concatenate([observed, predicted]))):
        raise ValueError("y_true and y_pred must contain only finite values")
    denominator = np.abs(observed) + np.abs(predicted)
    terms = np.divide(
        2.0 * np.abs(predicted - observed),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(np.mean(terms) * 100)


def residual_autocorrelation(y_true, y_pred, max_lag=1):
    """Measure remaining serial correlation in one-step forecast errors.

    A lag-one autocorrelation close to zero means the errors look like noise;
    strong positive correlation signals that the model still leaves
    exploitable structure on the table. Returns per-lag correlation values.
    """

    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")
    observed = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    if observed.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have equal length")
    if observed.size <= max_lag + 1:
        raise ValueError("need more observations than max_lag to estimate autocorrelation")

    residual = observed - predicted
    centered = residual - residual.mean()
    denominator = np.dot(centered, centered)
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError("residuals must have non-zero variance")

    lags = np.arange(1, max_lag + 1)
    values = []
    for lag in lags:
        numerator = np.dot(centered[lag:], centered[:-lag])
        values.append(float(numerator / denominator))
    return {f"lag_{int(lag)}": value for lag, value in zip(lags, values)}


def ljung_box_test(residuals, lags=None, alpha=0.05):
    """Test residual serial correlation with the Ljung-Box Q statistic.

    White-noise residuals are the target after fitting a model; residuals
    that still autocorrelate mean structure is left on the table. Unlike
    :func:`residual_autocorrelation`, which reports raw lag correlations,
    this is a portmanteau hypothesis test: the Q statistic accumulates
    squared sample autocorrelations with a small-sample correction and its
    p-value is read from a chi-squared null distribution, so the answer is
    a verdict rather than a correlation magnitude. ``lags`` accepts a
    single integer (the test is then performed at every lag up to it) or
    an explicit sequence of lags. The verdict is ``autocorrelated`` when
    any tested lag rejects the white-noise null at ``alpha``.
    """

    from statsmodels.stats.diagnostic import acorr_ljungbox

    observed = np.asarray(residuals, dtype=float).ravel()
    if observed.size == 0:
        raise ValueError("residuals must not be empty")
    if not np.all(np.isfinite(observed)):
        raise ValueError("residuals must contain only finite numbers")
    if observed.size < 2:
        raise ValueError("at least two residuals are required")
    if np.all(observed == observed[0]):
        raise ValueError("residuals must have non-zero variance")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if isinstance(lags, bool):
        raise ValueError("lags must be a positive integer or a sequence of integers")
    if lags is None:
        tested_lags = int(min(10, observed.size - 1))
    elif isinstance(lags, int):
        if lags < 1:
            raise ValueError("lags must be a positive integer")
        tested_lags = lags
    else:
        tested_lags = [int(lag) for lag in lags]
        if not tested_lags or any(lag < 1 for lag in tested_lags):
            raise ValueError("lags must be a non-empty sequence of positive integers")
    if isinstance(tested_lags, list):
        if max(tested_lags) >= observed.size:
            raise ValueError("lags must stay below the number of residuals")
    elif tested_lags >= observed.size:
        raise ValueError("lags must stay below the number of residuals")

    table = acorr_ljungbox(observed, lags=tested_lags, return_df=True)
    tests = [
        {
            "lag": int(lag),
            "lb_stat": float(row["lb_stat"]),
            "p_value": float(row["lb_pvalue"]),
        }
        for lag, row in table.iterrows()
    ]
    return {
        "autocorrelated": bool(any(test["p_value"] < alpha for test in tests)),
        "alpha": float(alpha),
        "significant_lags": [test["lag"] for test in tests if test["p_value"] < alpha],
        "tests": tests,
    }


def residual_quantiles(y_true, y_pred, quantiles=(0.1, 0.5, 0.9)):
    """Return selected quantiles of signed forecast residuals (prediction - actual)."""

    observed = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    levels = np.asarray(quantiles, dtype=float).ravel()
    if observed.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have equal length")
    if observed.size == 0:
        raise ValueError("at least one forecast is required")
    if levels.size == 0 or np.any((levels < 0.0) | (levels > 1.0)):
        raise ValueError("quantiles must be a non-empty sequence between 0 and 1")
    residuals = predicted - observed
    return {
        f"q{int(round(level * 100)):02d}": float(np.quantile(residuals, level))
        for level in levels
    }


def seasonal_strength(values, seasonal_period):
    """Score how strongly a series is driven by seasonality and trend.

    A centered moving average supplies an STL-free trend estimate, the
    detrended values are averaged by seasonal phase into indices, and
    whatever remains is treated as noise. Each strength score is
    ``max(0, 1 - Var(noise) / Var(component + noise))``, so scores near
    one mean the component explains almost everything and scores near
    zero mean the component carries no signal beyond the noise.
    """

    if seasonal_period < 2:
        raise ValueError("seasonal_period must be at least 2")
    observed = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")
    period = int(seasonal_period)
    if observed.size < 2 * period:
        raise ValueError("need at least two complete seasonal periods")

    smoothed = np.convolve(
        observed, np.full(period, 1.0 / period), mode="valid"
    )
    if period % 2 == 0:
        trend = 0.5 * (smoothed[:-1] + smoothed[1:])
    else:
        trend = smoothed
    offset = (observed.size - trend.size) // 2

    detrended = observed[offset : offset + trend.size] - trend
    phases = np.arange(offset, offset + trend.size) % period
    indices = np.array(
        [detrended[phases == phase].mean() for phase in range(period)]
    )
    indices -= indices.mean()
    residual = detrended - indices[phases]

    def strength(component_variance):
        if component_variance == 0:
            return 0.0
        ratio = residual.var() / component_variance
        return float(min(1.0, max(0.0, 1.0 - ratio)))

    return {
        "seasonal_strength": strength(detrended.var()),
        "trend_strength": strength((trend + residual).var()),
    }


def seasonal_decompose(values, seasonal_period, model="additive"):
    """Decompose a series into trend, seasonal, and residual components.

    Classical decomposition estimates the trend with a centered moving
    average, averages the detrended values by seasonal phase into indices,
    and treats whatever remains as residual noise. For ``model="additive"``
    the seasonal indices add to the trend and the residual is
    ``values - trend - seasonal``; for ``model="multiplicative"`` the
    indices multiply the trend and the residual is
    ``values / (trend * seasonal)``, which requires strictly positive
    values. This is the classic method and differs from
    :func:`seasonal_strength`, which compresses the same machinery into
    variance-ratio scores: here the three aligned component series are
    returned for plotting and further analysis. The trend is undefined at
    the series edges, so the returned arrays carry NaN there and every
    component matches the input length.
    """

    if seasonal_period < 2:
        raise ValueError("seasonal_period must be at least 2")
    if model not in {"additive", "multiplicative"}:
        raise ValueError("model must be 'additive' or 'multiplicative'")
    observed = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")
    period = int(seasonal_period)
    if observed.size < 2 * period:
        raise ValueError("need at least two complete seasonal periods")
    if np.all(observed == observed[0]):
        raise ValueError("values must not be constant")
    if model == "multiplicative" and np.any(observed <= 0):
        raise ValueError(
            "multiplicative decomposition requires strictly positive values"
        )

    smoothed = np.convolve(observed, np.full(period, 1.0 / period), mode="valid")
    if period % 2 == 0:
        trend_short = 0.5 * (smoothed[:-1] + smoothed[1:])
    else:
        trend_short = smoothed
    offset = (observed.size - trend_short.size) // 2

    trend = np.full(observed.size, np.nan)
    trend[offset : offset + trend_short.size] = trend_short

    detrended = observed[offset : offset + trend_short.size]
    if model == "multiplicative":
        detrended = detrended / trend_short
    else:
        detrended = detrended - trend_short

    phases = np.arange(offset, offset + trend_short.size) % period
    seasonal_short = np.array(
        [detrended[phases == phase].mean() for phase in range(period)]
    )
    if model == "multiplicative":
        seasonal_short = seasonal_short / seasonal_short.mean()
    else:
        seasonal_short = seasonal_short - seasonal_short.mean()

    seasonal = np.full(observed.size, np.nan)
    seasonal[offset : offset + trend_short.size] = seasonal_short[phases]

    if model == "multiplicative":
        residual_short = detrended / seasonal_short[phases]
    else:
        residual_short = detrended - seasonal_short[phases]

    residual = np.full(observed.size, np.nan)
    residual[offset : offset + trend_short.size] = residual_short

    return {"trend": trend, "seasonal": seasonal, "residual": residual}


def mean_absolute_scaled_error(y_true, y_pred, y_train, seasonal_period=1):
    """Return MASE using an in-sample seasonal-naive scaling error.

    Unlike percentage errors, MASE remains defined when observations are zero
    and is comparable across series. Values below one beat the corresponding
    seasonal-naive forecast on average.
    """

    if seasonal_period < 1:
        raise ValueError("seasonal_period must be at least 1")
    observed = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    training = np.asarray(y_train, dtype=float).ravel()
    if observed.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have equal length")
    if training.size <= seasonal_period:
        raise ValueError("y_train must contain more than one seasonal period")
    scale = np.mean(np.abs(training[seasonal_period:] - training[:-seasonal_period]))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("seasonal-naive training error must be finite and non-zero")
    return float(np.mean(np.abs(observed - predicted)) / scale)


def root_mean_squared_scaled_error(y_true, y_pred, y_train, seasonal_period=1):
    """Return RMSSE against an in-sample seasonal-naive benchmark.

    RMSSE penalises large forecast misses more heavily than MASE while still
    remaining comparable across series with different scales.
    """

    if seasonal_period < 1:
        raise ValueError("seasonal_period must be at least 1")
    observed = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    training = np.asarray(y_train, dtype=float).ravel()
    if observed.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have equal length")
    if training.size <= seasonal_period:
        raise ValueError("y_train must contain more than one seasonal period")
    scale = np.mean((training[seasonal_period:] - training[:-seasonal_period]) ** 2)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("seasonal-naive training error must be finite and non-zero")
    return float(np.sqrt(np.mean((observed - predicted) ** 2) / scale))

def forecast_skill_score(y_true, y_pred, y_baseline, score="mae"):
    """Return signed skill as percent improvement over a reference forecast.

    Raw error magnitudes say nothing about whether a model is actually
    better than a cheaper alternative. This compares the candidate and
    baseline forecasts on identical observations with ``score`` ("mae" or
    "rmse") and combines them into
    ``100 * (1 - model_loss / baseline_loss)``, so positive values mark
    skillful improvements, zero marks parity, and negative values mark
    forecasts that lose to the reference. Unlike MASE or RMSSE the scale
    comes from an explicit out-of-sample forecast rather than in-sample
    training errors.
    """

    if score not in {"mae", "rmse"}:
        raise ValueError("score must be 'mae' or 'rmse'")
    observed = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    baseline = np.asarray(y_baseline, dtype=float).ravel()
    if not (observed.shape == predicted.shape == baseline.shape):
        raise ValueError("y_true, y_pred, and y_baseline must have equal length")
    if observed.size == 0:
        raise ValueError("at least one observation is required")

    def loss(values):
        residual = observed - values
        if score == "mae":
            return float(np.mean(np.abs(residual)))
        return float(np.sqrt(np.mean(residual**2)))

    reference = loss(baseline)
    if not np.isfinite(reference) or reference == 0:
        raise ValueError("baseline loss must be finite and non-zero")
    return 100.0 * (1.0 - loss(predicted) / reference)


def diebold_mariano_test(
    y_true,
    y_pred_a,
    y_pred_b,
    loss="squared",
    max_lag=0,
    small_sample=False,
):
    """Test whether two forecasts differ significantly in accuracy.

    For every observation the losses of both forecasts are compared,
    giving a loss differential whose mean decides which forecast is
    more accurate; a negative statistic favours ``y_pred_a``. The
    denominator uses the Newey-West long-run variance so autocorrelated
    multi-step differentials stay valid: set ``max_lag`` to the forecast
    horizon minus one (zero for one-step-ahead). With ``small_sample``
    the statistic is scaled by the Harvey-Leybourne-Newbold correction
    and referred to a t distribution instead of the standard normal.
    Returns a two-sided test as ``{"dm_stat": ..., "p_value": ...}``.
    """

    if loss not in {"squared", "absolute"}:
        raise ValueError("loss must be 'squared' or 'absolute'")
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 0:
        raise ValueError("max_lag must be a non-negative integer")
    observed = np.asarray(y_true, dtype=float).ravel()
    first = np.asarray(y_pred_a, dtype=float).ravel()
    second = np.asarray(y_pred_b, dtype=float).ravel()
    if not (observed.shape == first.shape == second.shape):
        raise ValueError("y_true, y_pred_a, and y_pred_b must have equal length")
    if observed.size == 0:
        raise ValueError("at least one observation is required")
    if observed.size <= max_lag:
        raise ValueError("need more observations than max_lag")

    errors_first = observed - first
    errors_second = observed - second
    if loss == "squared":
        differential = errors_first**2 - errors_second**2
    else:
        differential = np.abs(errors_first) - np.abs(errors_second)

    count = differential.size
    centered = differential - differential.mean()
    long_run = float(np.mean(centered**2))
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        covariance = float(np.dot(centered[lag:], centered[:-lag])) / count
        long_run += 2.0 * weight * covariance
    if not np.isfinite(long_run) or long_run <= 0:
        raise ValueError("loss differential variance must be finite and positive")

    statistic = float(differential.mean() / np.sqrt(long_run / count))
    if small_sample:
        correction = np.sqrt(
            (count + 1 - 2 * max_lag + max_lag * (max_lag - 1) / count) / count
        )
        statistic *= correction
        p_value = float(2.0 * student_t.sf(abs(statistic), df=count - 1))
    else:
        p_value = float(2.0 * norm.sf(abs(statistic)))
    return {"dm_stat": statistic, "p_value": p_value}


def interval_metrics(y_true, lower, upper, coverage=0.9):
    """Evaluate prediction intervals with coverage, width, and Winkler score."""

    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between 0 and 1")
    observed = np.asarray(y_true, dtype=float).ravel()
    low = np.asarray(lower, dtype=float).ravel()
    high = np.asarray(upper, dtype=float).ravel()
    if not (observed.shape == low.shape == high.shape):
        raise ValueError("y_true, lower, and upper must have equal length")
    if observed.size == 0:
        raise ValueError("at least one interval is required")
    if np.any(low > high):
        raise ValueError("lower bounds must not exceed upper bounds")
    if not np.all(np.isfinite(np.concatenate([observed, low, high]))):
        raise ValueError("interval inputs must contain only finite values")

    width = high - low
    alpha = 1.0 - coverage
    winkler = width.copy()
    below = observed < low
    above = observed > high
    winkler[below] += (2.0 / alpha) * (low[below] - observed[below])
    winkler[above] += (2.0 / alpha) * (observed[above] - high[above])
    return {
        "coverage": float(np.mean((observed >= low) & (observed <= high))),
        "mean_width": float(np.mean(width)),
        "winkler_score": float(np.mean(winkler)),
    }


def interval_calibration_curve(y_true, intervals):
    """Compare nominal and empirical coverage across multiple interval levels.

    ``intervals`` maps each nominal coverage level to a ``(lower, upper)``
    pair. The returned table is sorted by nominal coverage for direct plotting.
    """

    if not isinstance(intervals, dict) or not intervals:
        raise ValueError("intervals must be a non-empty mapping of coverage levels")
    rows = []
    for nominal, bounds in intervals.items():
        try:
            coverage = float(nominal)
            lower, upper = bounds
        except (TypeError, ValueError) as exc:
            raise ValueError("each interval must be a (lower, upper) pair") from exc
        metrics = interval_metrics(y_true, lower, upper, coverage=coverage)
        rows.append(
            {
                "nominal_coverage": coverage,
                "empirical_coverage": metrics["coverage"],
                "mean_width": metrics["mean_width"],
            }
        )
    return pd.DataFrame(rows).sort_values("nominal_coverage").reset_index(drop=True)


def quantile_loss(y_true, forecasts, quantiles=None):
    """Score quantile forecasts with the pinball loss.

    ``forecasts`` may be a 2-D array (rows = horizons, columns = quantiles)
    or a mapping from quantile to a 1-D forecast. The pinball loss is the
    standard proper scoring rule for probabilistic point forecasts: it is
    asymmetric, penalising under-forecasts more at high quantiles and
    over-forecasts more at low quantiles. Lower is better.
    """

    observed = np.asarray(y_true, dtype=float).ravel()
    if quantiles is None:
        quantiles = (0.1, 0.5, 0.9)

    if isinstance(forecasts, dict):
        labels = sorted(forecasts)
        matrix = np.column_stack([np.asarray(forecasts[q], dtype=float).ravel() for q in labels])
        effective_quantiles = [float(q) for q in labels]
    else:
        matrix = np.asarray(forecasts, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("forecasts must be 2-D (horizons x quantiles) or a dict")
        if matrix.shape[1] != len(quantiles):
            raise ValueError(
                f"expected {len(quantiles)} quantile columns, got {matrix.shape[1]}"
            )
        effective_quantiles = [float(q) for q in quantiles]

    if not all(0.0 < q < 1.0 for q in effective_quantiles):
        raise ValueError("quantiles must be strictly between 0 and 1")
    if matrix.shape[0] != observed.size:
        raise ValueError("y_true and forecasts must have equal length")
    if observed.size == 0:
        raise ValueError("at least one forecast is required")

    residuals = observed[:, None] - matrix
    losses = np.where(
        residuals >= 0,
        np.asarray(effective_quantiles) * residuals,
        (np.asarray(effective_quantiles) - 1.0) * residuals,
    )
    mean_loss = float(np.mean(losses))
    per_quantile = {
        f"q{int(round(q * 100)):02d}": float(losses[:, i].mean())
        for i, q in enumerate(effective_quantiles)
    }
    return {"pinball_loss": mean_loss, "per_quantile": per_quantile}


def summarize_backtest(results):
    """Summarize tidy rolling-origin predictions for each forecast horizon."""

    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas.DataFrame")
    required = {"horizon", "actual", "prediction"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"backtest results missing columns: {', '.join(missing)}")
    if results.empty:
        return pd.DataFrame(columns=["count", "mae", "rmse", "bias"]).rename_axis(
            "horizon"
        )

    rows = []
    for horizon, group in results.groupby("horizon", sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        prediction = group["prediction"].to_numpy(dtype=float)
        residual = prediction - actual
        rows.append(
            {
                "horizon": horizon,
                "count": int(len(group)),
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "bias": float(np.mean(residual)),
            }
        )
    return pd.DataFrame(rows).set_index("horizon")


def summarize_interval_backtest(results, coverage=0.9):
    """Summarize rolling-origin interval coverage separately by horizon.

    ``results`` must contain ``horizon``, ``actual``, ``lower``, and ``upper``
    columns. The output combines count, empirical coverage, mean width, and
    Winkler score so interval degradation at longer horizons is visible.
    """

    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas.DataFrame")
    required = {"horizon", "actual", "lower", "upper"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(
            f"interval backtest results missing columns: {', '.join(missing)}"
        )
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between 0 and 1")
    columns = ["count", "coverage", "mean_width", "winkler_score"]
    if results.empty:
        return pd.DataFrame(columns=columns).rename_axis("horizon")

    rows = []
    for horizon, group in results.groupby("horizon", sort=True):
        metrics = interval_metrics(
            group["actual"], group["lower"], group["upper"], coverage=coverage
        )
        rows.append(
            {
                "horizon": horizon,
                "count": int(len(group)),
                **metrics,
            }
        )
    return pd.DataFrame(rows).set_index("horizon")[columns]


def backtest_stability(results, score="absolute_error"):
    """Measure across-origin dispersion of per-horizon backtest scores.

    ``summarize_backtest`` averages accuracy over origins, which hides
    models whose per-horizon errors swing widely between folds. For each
    horizon this collects one score per origin and reports the sample
    standard deviation and interquartile range of those scores next to
    their mean, so a horizon that is accurate only for some origins
    becomes visible. A single origin leaves dispersion unidentified:
    its ``score_std`` is NaN while ``score_iqr`` collapses to zero.

    ``score`` selects the per-fold quantity: ``"absolute_error"``
    (the default) or ``"squared_error"``.
    """

    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas.DataFrame")
    if score not in {"absolute_error", "squared_error"}:
        raise ValueError("score must be 'absolute_error' or 'squared_error'")
    required = {"origin", "horizon", "actual", "prediction"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"backtest results missing columns: {', '.join(missing)}")

    columns = ["count", "score_mean", "score_std", "score_iqr"]
    if results.empty:
        return pd.DataFrame(columns=columns).rename_axis("horizon")

    residuals = (
        results["prediction"].to_numpy(dtype=float)
        - results["actual"].to_numpy(dtype=float)
    )
    values = np.abs(residuals) if score == "absolute_error" else residuals**2
    scored = results[["horizon", "origin"]].assign(score=values)

    rows = []
    for horizon, group in scored.groupby("horizon", sort=True):
        per_origin = group.groupby("origin")["score"].mean().to_numpy(dtype=float)
        q25, q75 = np.quantile(per_origin, [0.25, 0.75])
        std = np.std(per_origin, ddof=1) if per_origin.size > 1 else np.nan
        rows.append(
            {
                "horizon": horizon,
                "count": int(per_origin.size),
                "score_mean": float(np.mean(per_origin)),
                "score_std": float(std),
                "score_iqr": float(q75 - q25),
            }
        )
    return pd.DataFrame(rows).set_index("horizon")[columns]


def compare_models(results):
    comparisons = []
    for name, (y_true, y_pred) in results.items():
        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = name
        comparisons.append(metrics)
    return pd.DataFrame(comparisons).set_index("model")

def recalibrate_conformal_intervals(
    calibration_true,
    calibration_pred,
    heldout_true,
    heldout_pred,
    forecast,
    coverage=0.9,
):
    """Recalibrate split-conformal intervals using held-out miscoverage.

    Standard split-conformal intervals target nominal coverage but can
    under- or over-cover on new data. This function measures the empirical
    coverage of conformal intervals on a held-out set and returns adjusted
    intervals for the forecast by scaling the conformal radius.

    Parameters
    ----------
    calibration_true, calibration_pred : array-like
        Calibration data used to build the initial conformal radius.
    heldout_true, heldout_pred : array-like
        Held-out data for measuring empirical coverage. Must be independent
        of the calibration set.
    forecast : array-like
        Point forecasts to build intervals around.
    coverage : float, default 0.9
        Nominal coverage level for the initial conformal intervals.

    Returns
    -------
    lower, upper : ndarray
        Recalibrated prediction intervals for the forecast.
    scale_factor : float
        Multiplicative factor applied to the original conformal radius.
    empirical_coverage : float
        Observed coverage on the held-out set before recalibration.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between 0 and 1")

    # Build initial conformal radius on calibration data
    observed_cal = np.asarray(calibration_true, dtype=float).ravel()
    predicted_cal = np.asarray(calibration_pred, dtype=float).ravel()
    if observed_cal.shape != predicted_cal.shape:
        raise ValueError("calibration_true and calibration_pred must have equal length")
    finite_cal = np.isfinite(observed_cal) & np.isfinite(predicted_cal)
    scores = np.abs(observed_cal[finite_cal] - predicted_cal[finite_cal])
    if scores.size == 0:
        raise ValueError("at least one finite calibration residual is required")

    quantile_level = min(
        1.0,
        np.ceil((scores.size + 1) * coverage) / scores.size,
    )
    radius = float(np.quantile(scores, quantile_level, method="higher"))

    # Evaluate empirical coverage on held-out data
    observed_ho = np.asarray(heldout_true, dtype=float).ravel()
    predicted_ho = np.asarray(heldout_pred, dtype=float).ravel()
    if observed_ho.shape != predicted_ho.shape:
        raise ValueError("heldout_true and heldout_pred must have equal length")
    finite_ho = np.isfinite(observed_ho) & np.isfinite(predicted_ho)
    if not np.any(finite_ho):
        raise ValueError("held-out set must contain at least one finite pair")
    ho_scores = np.abs(observed_ho[finite_ho] - predicted_ho[finite_ho])

    empirical_coverage = float(np.mean(ho_scores <= radius))

    # Compute scale factor to align empirical with nominal coverage
    target_quantile = min(
        1.0,
        np.ceil((ho_scores.size + 1) * coverage) / ho_scores.size,
    )
    target_radius = float(np.quantile(ho_scores, target_quantile, method="higher"))

    if radius > 0:
        scale_factor = target_radius / radius
    else:
        scale_factor = 1.0

    # Apply scaled radius to forecast
    point_forecast = np.asarray(forecast, dtype=float)
    scaled_radius = radius * scale_factor
    lower = point_forecast - scaled_radius
    upper = point_forecast + scaled_radius

    return lower, upper, scale_factor, empirical_coverage


def forecast_diagnostics_bundle(
    y_true,
    y_pred,
    *,
    seasonal_period: int = 1,
    max_lag: int = 10,
    alpha: float = 0.05,
    baseline: str = "naive",
    train=None,
):
    """Compute a one-shot forecast-diagnostics bundle.

    Combines the standard point-forecast metrics (MAE, RMSE, MAPE, bias,
    sMAPE), residual autocorrelation lags, Ljung-Box test, and the
    forecast-skill score against an in-sample baseline forecast (naive or
    seasonal-naive). The shape is a single dict so triage dashboards and
    CLI outputs can show every relevant diagnostic for one run without
    threading half a dozen function calls together.

    ``seasonal_period`` controls the seasonal-naive baseline and feeds into
    the residual diagnostics; ``max_lag`` is the maximum lag included in
    the autocorrelation and Ljung-Box computations; ``alpha`` is the
    Ljung-Box significance level; ``train`` provides the in-sample series
    for the baseline forecast and must be supplied when ``baseline`` is
    not ``"naive"``.
    """

    if seasonal_period < 1:
        raise ValueError("seasonal_period must be at least 1")
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if baseline not in {"naive", "seasonal_naive"}:
        raise ValueError("baseline must be 'naive' or 'seasonal_naive'")
    if baseline == "seasonal_naive" and train is None:
        raise ValueError("train is required when baseline='seasonal_naive'")

    true_arr = np.asarray(y_true, dtype=float).ravel()
    pred_arr = np.asarray(y_pred, dtype=float).ravel()
    if true_arr.size != pred_arr.size:
        raise ValueError("y_true and y_pred must have the same length")
    if true_arr.size == 0:
        raise ValueError("y_true and y_pred must not be empty")

    bundle: dict[str, object] = {
        "metrics": compute_metrics(true_arr, pred_arr),
        "bias": forecast_bias(true_arr, pred_arr),
        "smape": symmetric_mean_absolute_percentage_error(true_arr, pred_arr),
        "residual_autocorrelation": residual_autocorrelation(
            true_arr, pred_arr, max_lag=max_lag
        ),
        "ljung_box": ljung_box_test(
            true_arr - pred_arr, lags=min(max_lag, max(true_arr.size - 1, 1)), alpha=alpha
        ),
    }
    tests = bundle["ljung_box"].get("tests") or []
    bundle["min_residual_p_value"] = min(
        (float(test["p_value"]) for test in tests), default=float("nan")
    )

    if baseline == "naive":
        naive = np.concatenate([[true_arr[0]], true_arr[:-1]])
    else:
        period = max(int(seasonal_period), 1)
        naive = np.concatenate(
            [true_arr[:period], true_arr[:-period]]
        )

    bundle["skill_score"] = forecast_skill_score(
        true_arr, pred_arr, naive, score="mae"
    )

    return bundle


def coverage_straddle_test(
    intervals,
    nominal: float = 0.9,
    *,
    alpha: float = 0.05,
    min_n: int = 10,
):
    """Test whether the empirical coverage of (lower, upper) intervals matches a nominal rate.

    ``intervals`` is a sequence of precomputed (lower, upper) pairs. The test
    computes the empirical coverage as the share of finite intervals whose
    lower bound does not exceed its upper bound, then runs a two-sided
    proportion test against ``nominal`` (default 0.9). Returns the
    z-statistic, the two-sided p-value, and a string verdict so callers can
    act without recomputing.

    Parameters
    ----------
    intervals : sequence of length-2 sequences
        Precomputed (lower, upper) pairs.
    nominal : float
        Target coverage in (0, 1).
    alpha : float
        Significance level used to translate the p-value into a verdict.
        Values below ``alpha`` flag the deviation as significant.
    min_n : int
        Minimum number of intervals required to run the test; below this
        we return ``"insufficient"`` without a z-score.
    """
    if not 0.0 < nominal < 1.0:
        raise ValueError("nominal must be strictly between 0 and 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if min_n < 1:
        raise ValueError("min_n must be at least 1")

    pairs = [tuple(interval) for interval in intervals]
    n = len(pairs)
    if n < min_n:
        return {
            "n": n,
            "observed_coverage": float("nan"),
            "z": float("nan"),
            "p_value": float("nan"),
            "verdict": "insufficient",
        }

    covered = 0
    for lower, upper in pairs:
        try:
            lo = float(lower)
            hi = float(upper)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if lo <= hi:
            covered += 1
    observed = covered / n

    if nominal <= 0.0 or nominal >= 1.0:
        raise ValueError("nominal must be strictly between 0 and 1")
    se = float(np.sqrt(nominal * (1.0 - nominal) / n))
    if se == 0.0:
        z = 0.0
        p_value = 1.0
    else:
        z = (observed - nominal) / se
        # Two-sided p-value from the standard normal survival function.
        p_value = 2.0 * float(norm.sf(abs(z)))

    if p_value > alpha:
        verdict = "on-target"
    elif observed < nominal:
        verdict = "under-covered"
    else:
        verdict = "over-covered"

    return {
        "n": n,
        "observed_coverage": float(observed),
        "z": float(z),
        "p_value": float(p_value),
        "verdict": verdict,
    }
