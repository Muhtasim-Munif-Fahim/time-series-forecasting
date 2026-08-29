"""Stationarity and unit-root diagnostics for time series."""

import warnings

import numpy as np
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(values, maxlag=None, regression="c", autolag="AIC", alpha=0.05):
    """Run the augmented Dickey-Fuller unit-root test on a series.

    The ADF test checks the null hypothesis that the series has a unit
    root against the stationary alternative. The ``regression`` term
    controls which deterministic components are included: ``"c"`` fits a
    constant, ``"ct"`` adds a linear trend, ``"ctt"`` adds a quadratic
    trend, and ``"n"`` fits neither. ``maxlag`` fixes the number of
    lagged difference terms (the default lets ``autolag`` pick it by
    AIC or BIC). The verdict marks the series stationary when the
    p-value falls below ``alpha``.
    """

    observed = np.asarray(values, dtype=float).ravel()
    if observed.size < 10:
        raise ValueError("at least ten observations are required for the ADF test")
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")
    if regression not in {"c", "ct", "ctt", "n"}:
        raise ValueError("regression must be one of 'c', 'ct', 'ctt', 'n'")
    if maxlag is not None and (
        isinstance(maxlag, bool) or not isinstance(maxlag, int) or maxlag < 0
    ):
        raise ValueError("maxlag must be a non-negative integer or None")
    if autolag not in {"AIC", "BIC", "t-stat", None}:
        raise ValueError("autolag must be 'AIC', 'BIC', 't-stat', or None")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    result = adfuller(
        observed, maxlag=maxlag, regression=regression, autolag=autolag
    )
    if autolag is None:
        statistic, p_value, usedlag, nobs, critical_values = result
    else:
        statistic, p_value, usedlag, nobs, critical_values, _ = result
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "usedlag": int(usedlag),
        "nobs": int(nobs),
        "critical_values": {level: float(value) for level, value in critical_values.items()},
        "stationary": bool(p_value < alpha),
    }


def kpss_test(values, regression="c", nlags="auto", alpha=0.05):
    """Run the KPSS stationarity test on a series.

    KPSS tests the null hypothesis that the series is stationary (level
    stationarity with ``regression="c"``, trend stationarity with
    ``regression="ct"``), the opposite null to :func:`adf_test`. Rejecting
    KPSS while failing to reject ADF is the classic evidence of a
    unit-root series, so the two tests are best read together.
    ``nlags`` fixes the number of lags for the Newey-West estimate of the
    long-run variance, with ``"auto"`` choosing by the standard heuristic.
    The verdict marks the series stationary when the p-value stays at or
    above ``alpha`` (i.e. the stationarity null is not rejected). The
    statistic occasionally lands outside the p-value look-up table; the
    returned p-value then saturates at the nearest extreme (``0.01`` or
    ``0.1``) and the routine warns only about that censoring.
    """

    observed = np.asarray(values, dtype=float).ravel()
    if observed.size < 10:
        raise ValueError("at least ten observations are required for the KPSS test")
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")
    if regression not in {"c", "ct"}:
        raise ValueError("regression must be 'c' or 'ct'")
    if nlags not in {"auto", "legacy"} and (
        isinstance(nlags, bool) or not isinstance(nlags, int) or nlags < 0
    ):
        raise ValueError("nlags must be 'auto', 'legacy', or a non-negative integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        statistic, p_value, usedlag, critical_values = kpss(
            observed, regression=regression, nlags=nlags
        )
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "usedlag": int(usedlag),
        "critical_values": {level: float(value) for level, value in critical_values.items()},
        "stationary": bool(p_value >= alpha),
    }


def stationarity_report(values, max_diffs=2, alpha=0.05, **adf_kwargs):
    """Diagnose stationarity and suggest how many differences are needed.

    The series is tested with the augmented Dickey-Fuller test at zero,
    one, and up to ``max_diffs`` differences. ``suggested_diffs`` is the
    first order at which the test rejects a unit root (``0`` when the
    original series is already stationary); when the series stays
    non-stationary at every tested order the suggestion saturates at
    ``max_diffs`` and the verdict reports non-stationarity. The full
    per-order test results are returned so the p-value trajectory is
    visible, not just the final recommendation.
    """

    if isinstance(max_diffs, bool) or not isinstance(max_diffs, int) or max_diffs < 1:
        raise ValueError("max_diffs must be a positive integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    observed = np.asarray(values, dtype=float).ravel()
    if observed.size < 10:
        raise ValueError("at least ten observations are required for the ADF test")
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")

    tests = []
    suggested = None
    current = observed
    for order in range(max_diffs + 1):
        result = adf_test(current, alpha=alpha, **adf_kwargs)
        tests.append(
            {
                "diffs": order,
                "statistic": result["statistic"],
                "p_value": result["p_value"],
                "stationary": bool(result["p_value"] < alpha),
            }
        )
        if result["p_value"] < alpha and suggested is None:
            suggested = order
        if order < max_diffs:
            current = np.diff(current)

    if suggested is None:
        suggested = max_diffs
    return {
        "verdict": "stationary" if suggested == 0 else "non-stationary",
        "suggested_diffs": int(suggested),
        "alpha": float(alpha),
        "tests": tests,
    }
