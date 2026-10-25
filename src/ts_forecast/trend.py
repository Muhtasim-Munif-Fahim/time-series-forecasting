"""Non-parametric trend magnitude, seasonal trend and change-point tests.

:func:`ts_forecast.diagnostics.mann_kendall_test` says *whether* a series has
a monotonic trend. This module adds the companion tools hydrologists and
climatologists use alongside it:

* :func:`sens_slope` is the Theil-Sen (Sen 1968) estimator of the trend
  *magnitude*: the median of all pairwise slopes, with the distribution-free
  confidence interval of Gilbert (1987) built from the Mann-Kendall variance.
* :func:`seasonal_mann_kendall_test` is the Hirsch-Slack-Smith (1982)
  seasonal Kendall test. It runs Mann-Kendall within each season (month,
  weekday, ...) and pools ``S`` and its variance, so a strong seasonal cycle
  does not mask or fake a trend. It also returns the seasonal Sen slope.
* :func:`pettitt_test` is Pettitt's (1979) rank-based test for a single
  change point in location, with the usual approximate p-value.

:func:`trend_report` runs all of them together.
"""

import numpy as np
from scipy.stats import norm, rankdata


def _clean(values, minimum, name="values"):
    observed = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(observed)):
        raise ValueError(f"{name} must contain only finite numbers")
    if observed.size < minimum:
        raise ValueError(f"at least {minimum} observations are required")
    return observed


def _check_alpha(alpha):
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")


def _kendall_s_and_variance(values):
    """Return the Mann-Kendall ``S`` and its tie-corrected null variance."""

    n = values.size
    if n < 2:
        return 0.0, 0.0
    diffs = values[None, :] - values[:, None]
    statistic = float(np.sum(np.sign(diffs[np.triu_indices(n, k=1)])))
    _, counts = np.unique(values, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1.0) * (2.0 * counts + 5.0)))
    variance = (n * (n - 1.0) * (2.0 * n + 5.0) - tie_term) / 18.0
    return statistic, max(variance, 0.0)


def _z_and_p(statistic, variance):
    if variance <= 0:
        return 0.0, 1.0
    if statistic > 0:
        z_score = (statistic - 1.0) / np.sqrt(variance)
    elif statistic < 0:
        z_score = (statistic + 1.0) / np.sqrt(variance)
    else:
        z_score = 0.0
    return float(z_score), float(2.0 * norm.sf(abs(z_score)))


def _pairwise_slopes(values, times):
    i, j = np.triu_indices(values.size, k=1)
    dt = times[j] - times[i]
    keep = dt != 0
    return (values[j] - values[i])[keep] / dt[keep]


def _slope_interval(slopes, variance, alpha):
    """Gilbert (1987) rank interval for the median of ``slopes``.

    Uses the same rank arithmetic as :func:`scipy.stats.theilslopes`, so the
    bounds agree with SciPy for untied time indices.
    """

    sorted_slopes = np.sort(slopes)
    count = sorted_slopes.size
    if count == 0 or variance <= 0:
        return float("nan"), float("nan")
    half_width = norm.ppf(alpha / 2.0) * np.sqrt(variance)  # negative
    upper = min(int(round((count - half_width) / 2.0)), count - 1)
    lower = max(int(round((count + half_width) / 2.0)) - 1, 0)
    return float(sorted_slopes[lower]), float(sorted_slopes[upper])


def sens_slope(values, times=None, alpha=0.05):
    """Return the Theil-Sen trend slope with a distribution-free interval.

    The slope is the median of ``(x_j - x_i) / (t_j - t_i)`` over all pairs
    ``i < j``; it is robust to outliers (breakdown point ~29%) and needs no
    normality assumption. The ``1 - alpha`` interval takes the order
    statistics of the sorted slopes at ``(N' -/+ z sqrt(Var S)) / 2``
    (Gilbert 1987), where ``Var S`` is the tie-corrected Mann-Kendall
    variance. The intercept follows Conover: ``median(x) - slope *
    median(t)``. ``times`` defaults to ``0, 1, ..., n - 1``.
    """

    observed = _clean(values, 3)
    _check_alpha(alpha)
    if times is None:
        t = np.arange(observed.size, dtype=float)
    else:
        t = np.asarray(times, dtype=float).ravel()
        if t.shape != observed.shape:
            raise ValueError("times must have the same length as values")
        if not np.all(np.isfinite(t)):
            raise ValueError("times must contain only finite numbers")
    slopes = _pairwise_slopes(observed, t)
    if slopes.size == 0:
        raise ValueError("times must contain at least two distinct values")
    slope = float(np.median(slopes))
    n = observed.size
    _, counts_x = np.unique(observed, return_counts=True)
    _, counts_t = np.unique(t, return_counts=True)
    variance = (
        n * (n - 1.0) * (2.0 * n + 5.0)
        - float(np.sum(counts_x * (counts_x - 1.0) * (2.0 * counts_x + 5.0)))
        - float(np.sum(counts_t * (counts_t - 1.0) * (2.0 * counts_t + 5.0)))
    ) / 18.0
    low, high = _slope_interval(slopes, variance, alpha)
    intercept = float(np.median(observed) - slope * np.median(t))
    return {
        "slope": slope,
        "intercept": intercept,
        "ci_low": low,
        "ci_high": high,
        "n_slopes": int(slopes.size),
        "alpha": float(alpha),
    }


def seasonal_mann_kendall_test(values, period, alpha=0.05):
    """Run the Hirsch-Slack seasonal Kendall test for monotonic trend.

    Observation ``k`` belongs to season ``k % period`` and cycle
    ``k // period``. Mann-Kendall ``S_g`` and its tie-corrected variance are
    computed within every season and summed; the pooled statistic
    ``Z = (S - sign(S)) / sqrt(sum Var S_g)`` is referred to N(0, 1). The
    test assumes seasons are independent of each other (Hirsch et al.
    1982). Missing observations may be passed as ``NaN`` and are dropped
    season by season.

    The returned ``slope_per_cycle`` is the seasonal Sen slope, the median
    of all within-season pairwise slopes measured per cycle;
    ``slope_per_step`` divides it by ``period``. ``seasons`` lists the
    per-season ``S``, variance, count and Kendall's tau so heterogeneous
    trends can be spotted.
    """

    raw = np.asarray(values, dtype=float).ravel()
    if np.any(np.isinf(raw)):
        raise ValueError("values must not contain infinite numbers")
    period = int(period)
    if period < 2:
        raise ValueError("period must be at least 2")
    _check_alpha(alpha)
    if raw.size < 2 * period:
        raise ValueError("at least two full cycles of observations are required")

    total_s = 0.0
    total_var = 0.0
    seasons = []
    all_slopes = []
    for season in range(period):
        idx = np.arange(season, raw.size, period)
        sub = raw[idx]
        mask = np.isfinite(sub)
        sub = sub[mask]
        cycles = (idx[mask] // period).astype(float)
        s_g, var_g = _kendall_s_and_variance(sub)
        pairs = 0.5 * sub.size * (sub.size - 1.0)
        seasons.append(
            {
                "season": season,
                "n": int(sub.size),
                "s": s_g,
                "var_s": var_g,
                "tau": float(s_g / pairs) if pairs > 0 else 0.0,
            }
        )
        total_s += s_g
        total_var += var_g
        if sub.size >= 2:
            all_slopes.append(_pairwise_slopes(sub, cycles))

    if total_var <= 0:
        raise ValueError("no season has enough non-tied observations to test")
    z_score, p_value = _z_and_p(total_s, total_var)
    slopes = np.concatenate(all_slopes) if all_slopes else np.array([])
    slope = float(np.median(slopes)) if slopes.size else float("nan")
    low, high = _slope_interval(slopes, total_var, alpha)
    if p_value < alpha:
        verdict = "increasing" if total_s > 0 else "decreasing"
    else:
        verdict = "no trend"
    return {
        "s": float(total_s),
        "var_s": float(total_var),
        "z": z_score,
        "p_value": p_value,
        "trend": verdict,
        "slope_per_cycle": slope,
        "slope_per_step": slope / period,
        "slope_ci_per_cycle": (low, high),
        "period": period,
        "seasons": seasons,
        "alpha": float(alpha),
    }


def pettitt_test(values, alpha=0.05):
    """Run Pettitt's (1979) non-parametric single change-point test.

    ``U_t = sum_{i<=t} sum_{j>t} sign(x_j - x_i)`` is computed for every
    split through the (mid-)ranks ``r`` as ``U_t = t (n + 1) - 2 sum_{i<=t}
    r_i``, which is exact with ties. ``K = max |U_t|`` and the approximate
    p-value is ``2 exp(-6 K^2 / (n^3 + n^2))`` (capped at 1). The change
    point ``index`` is the last observation of the first segment, so
    ``values[:index + 1]`` and ``values[index + 1:]`` are the two regimes;
    ``shift`` is the difference of their medians (positive for an upward
    shift).
    """

    observed = _clean(values, 10)
    _check_alpha(alpha)
    if np.all(observed == observed[0]):
        raise ValueError("values must not be constant")
    n = observed.size
    ranks = rankdata(observed)
    t = np.arange(1, n)
    u = t * (n + 1.0) - 2.0 * np.cumsum(ranks)[:-1]
    location = int(np.argmax(np.abs(u)))
    k_stat = float(abs(u[location]))
    p_value = float(min(1.0, 2.0 * np.exp(-6.0 * k_stat**2 / (n**3 + n**2))))
    before = observed[: location + 1]
    after = observed[location + 1 :]
    return {
        "statistic": k_stat,
        "u": u.astype(float),
        "index": location,
        "p_value": p_value,
        "change_point": bool(p_value < alpha),
        "median_before": float(np.median(before)),
        "median_after": float(np.median(after)),
        "shift": float(np.median(after) - np.median(before)),
        "alpha": float(alpha),
    }


def trend_report(values, period=None, alpha=0.05):
    """Combine Mann-Kendall, Sen's slope, Pettitt and (optionally) seasonal Kendall.

    When ``period`` is given the seasonal Kendall test drives the verdict,
    otherwise the plain Mann-Kendall test does. ``summary`` is a short
    human-readable sentence.
    """

    from ts_forecast.diagnostics import mann_kendall_test

    observed = _clean(values, 10)
    report = {
        "mann_kendall": mann_kendall_test(observed, alpha=alpha),
        "sens_slope": sens_slope(observed, alpha=alpha),
        "pettitt": pettitt_test(observed, alpha=alpha),
        "seasonal": None,
    }
    trend = report["mann_kendall"]["trend"]
    slope = report["sens_slope"]["slope"]
    if period is not None:
        report["seasonal"] = seasonal_mann_kendall_test(observed, period, alpha=alpha)
        trend = report["seasonal"]["trend"]
        slope = report["seasonal"]["slope_per_step"]
    report["trend"] = trend
    pettitt = report["pettitt"]
    parts = [f"{trend} (Sen slope {slope:.4g} per step)"]
    if pettitt["change_point"]:
        parts.append(
            f"change point after index {pettitt['index']} "
            f"(shift {pettitt['shift']:.4g}, p={pettitt['p_value']:.3g})"
        )
    else:
        parts.append("no significant change point")
    report["summary"] = "; ".join(parts)
    return report
