"""Forecast rationality, encompassing and market-timing tests.

* :func:`mincer_zarnowitz_test` -- regress outcomes on forecasts and jointly
  test unbiasedness/efficiency (intercept 0, slope 1) with a Newey-West HAC
  covariance (Mincer & Zarnowitz 1969).
* :func:`forecast_encompassing_test` -- Harvey, Leybourne & Newbold (1998)
  test of whether forecast A already contains all the useful information in
  forecast B, plus the implied optimal combination weight.
* :func:`pesaran_timmermann_test` -- Pesaran & Timmermann (1992)
  nonparametric test of directional (sign) predictive ability.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, f as f_dist, norm, t as student_t

__all__ = [
    "mincer_zarnowitz_test",
    "forecast_encompassing_test",
    "pesaran_timmermann_test",
]


def _check_lag(max_lag):
    if isinstance(max_lag, bool) or not isinstance(max_lag, (int, np.integer)) or max_lag < 0:
        raise ValueError("max_lag must be a non-negative integer")
    return int(max_lag)


def _as_vectors(*arrays, names):
    vectors = [np.asarray(a, dtype=float).ravel() for a in arrays]
    shape = vectors[0].shape
    if any(v.shape != shape for v in vectors):
        raise ValueError(f"{', '.join(names)} must have equal length")
    if not all(np.all(np.isfinite(v)) for v in vectors):
        raise ValueError(f"{', '.join(names)} must contain only finite values")
    return vectors


def _newey_west_scores(scores, max_lag):
    """Bartlett-kernel long-run covariance of row-wise score vectors (n x k)."""

    n = scores.shape[0]
    omega = scores.T @ scores / n
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = scores[lag:].T @ scores[:-lag] / n
        omega += weight * (gamma + gamma.T)
    return omega


def mincer_zarnowitz_test(y_true, y_pred, max_lag=0, alpha=0.05):
    """Mincer-Zarnowitz regression test of forecast rationality.

    Fits ``y_t = a + b * f_t + u_t`` by OLS and tests the joint null
    ``a = 0, b = 1`` (an optimal forecast under squared loss is unbiased and
    efficient). The coefficient covariance is Newey-West HAC with
    ``max_lag`` Bartlett lags, so use ``h - 1`` for ``h``-step forecasts
    whose errors are MA(h-1). The Wald statistic is referred to a
    chi-square(2); an F(2, n-2) version (Wald / 2) is also returned.

    Returns a dict with ``intercept``, ``slope``, their HAC standard errors,
    individual t-tests (``a = 0`` and ``b = 1``), ``wald_stat``,
    ``p_value`` (chi-square), ``f_stat``, ``f_p_value``, ``r_squared``,
    ``n`` and ``rational`` (``p_value >= alpha``).
    """

    max_lag = _check_lag(max_lag)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    y, f = _as_vectors(y_true, y_pred, names=("y_true", "y_pred"))
    n = y.size
    if n < 3:
        raise ValueError("at least three observations are required")
    if n <= max_lag + 2:
        raise ValueError("need more observations than max_lag + 2")
    if np.ptp(f) == 0:
        raise ValueError("y_pred must vary for the regression to be identified")

    X = np.column_stack([np.ones(n), f])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    scores = X * resid[:, None]
    omega = _newey_west_scores(scores, max_lag)
    # small-sample dof scaling n/(n-k) as in common HAC implementations
    cov = n * xtx_inv @ omega @ xtx_inv * n / (n - 2)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    diff = beta - np.array([0.0, 1.0])
    try:
        wald = float(diff @ np.linalg.solve(cov, diff))
    except np.linalg.LinAlgError as exc:  # pragma: no cover - degenerate data
        raise ValueError("HAC covariance is singular") from exc
    p_value = float(chi2.sf(wald, df=2))
    f_stat = wald / 2.0
    f_p = float(f_dist.sf(f_stat, 2, n - 2))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_a = float(beta[0] / se[0]) if se[0] > 0 else float("inf")
        t_b = float((beta[1] - 1.0) / se[1]) if se[1] > 0 else float("inf")
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else float("nan")
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "se_intercept": float(se[0]),
        "se_slope": float(se[1]),
        "t_intercept": t_a,
        "p_intercept": float(2.0 * student_t.sf(abs(t_a), df=n - 2)),
        "t_slope": t_b,
        "p_slope": float(2.0 * student_t.sf(abs(t_b), df=n - 2)),
        "wald_stat": wald,
        "p_value": p_value,
        "f_stat": float(f_stat),
        "f_p_value": f_p,
        "r_squared": float(r2),
        "n": int(n),
        "max_lag": max_lag,
        "rational": bool(p_value >= alpha),
    }


def forecast_encompassing_test(
    y_true, y_pred_a, y_pred_b, max_lag=0, small_sample=True, alpha=0.05
):
    """Harvey-Leybourne-Newbold (1998) forecast encompassing test.

    Null: forecast A *encompasses* B, i.e. the optimal combination
    ``(1 - lam) * f_a + lam * f_b`` puts zero weight on B. With errors
    ``e_a, e_b`` the statistic is a Diebold-Mariano type t-ratio on
    ``d_t = e_a,t * (e_a,t - e_b,t)`` using a Newey-West long-run variance
    with ``max_lag`` lags. The alternative is one-sided (``E[d] > 0``: B
    adds information). ``small_sample`` applies the HLN correction and a
    t(n-1) reference distribution; otherwise the standard normal is used.

    Also returns the least-squares combination weight
    ``lam = sum(e_a (e_a - e_b)) / sum((e_a - e_b)^2)`` and the MSEs of
    A, B and the combination.
    """

    max_lag = _check_lag(max_lag)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    y, fa, fb = _as_vectors(
        y_true, y_pred_a, y_pred_b, names=("y_true", "y_pred_a", "y_pred_b")
    )
    n = y.size
    if n < 2:
        raise ValueError("at least two observations are required")
    if n <= max_lag + 1:
        raise ValueError("need more observations than max_lag + 1")

    ea = y - fa
    eb = y - fb
    gap = ea - eb
    d = ea * gap
    centered = d - d.mean()
    long_run = _newey_west_scores(centered[:, None], max_lag)[0, 0]
    if not np.isfinite(long_run) or long_run <= 0:
        raise ValueError("encompassing differential variance must be finite and positive")
    stat = float(d.mean() / np.sqrt(long_run / n))
    if small_sample:
        h = max_lag + 1
        stat *= float(np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))
        p_value = float(student_t.sf(stat, df=n - 1))
    else:
        p_value = float(norm.sf(stat))

    denom = float(gap @ gap)
    lam = float(ea @ gap / denom) if denom > 0 else 0.0
    combo = (1.0 - lam) * fa + lam * fb
    return {
        "statistic": stat,
        "p_value": p_value,
        "a_encompasses_b": bool(p_value >= alpha),
        "combination_weight_b": lam,
        "mse_a": float(np.mean(ea**2)),
        "mse_b": float(np.mean(eb**2)),
        "mse_combination": float(np.mean((y - combo) ** 2)),
        "n": int(n),
        "max_lag": max_lag,
    }


def pesaran_timmermann_test(y_true, y_pred, alpha=0.05, changes=False):
    """Pesaran-Timmermann (1992) test of directional predictive ability.

    Compares the hit rate ``P`` of ``sign(y_pred) == sign(y_true)`` with the
    rate ``P*`` expected if the signs were independent, and refers
    ``(P - P*) / sqrt(Var(P) - Var(P*))`` to the standard normal
    (one-sided: forecasts have *more* sign information than chance).
    With ``changes=True`` the test is applied to first differences, which
    is the natural choice for level forecasts. Zero values count as
    non-positive.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    y, f = _as_vectors(y_true, y_pred, names=("y_true", "y_pred"))
    if changes:
        y, f = np.diff(y), np.diff(f)
    n = y.size
    if n < 3:
        raise ValueError("at least three (differenced) observations are required")
    up_y = y > 0
    up_f = f > 0
    p_hit = float(np.mean(up_y == up_f))
    py = float(up_y.mean())
    pf = float(up_f.mean())
    p_star = py * pf + (1 - py) * (1 - pf)
    var_p = p_star * (1 - p_star) / n
    var_star = (
        (2 * py - 1) ** 2 * pf * (1 - pf) / n
        + (2 * pf - 1) ** 2 * py * (1 - py) / n
        + 4 * py * pf * (1 - py) * (1 - pf) / n**2
    )
    denom = var_p - var_star
    if not denom > 0:
        raise ValueError(
            "test is undefined when actual or forecast signs never vary"
        )
    stat = float((p_hit - p_star) / np.sqrt(denom))
    p_value = float(norm.sf(stat))
    return {
        "statistic": stat,
        "p_value": p_value,
        "hit_rate": p_hit,
        "expected_hit_rate": float(p_star),
        "n": int(n),
        "has_directional_skill": bool(p_value < alpha),
    }
