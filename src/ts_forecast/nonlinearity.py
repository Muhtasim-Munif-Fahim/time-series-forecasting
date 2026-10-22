"""Tests for nonlinear serial dependence and conditional heteroskedasticity.

The Ljung-Box test in :mod:`ts_forecast.evaluation` only looks at linear
autocorrelation. A residual series can pass it and still be predictable:
volatility clustering (ARCH/GARCH), threshold or bilinear dynamics, and
deterministic chaos all leave the autocorrelations near zero. This module
adds three standard residual checks for that kind of dependence.

* :func:`bds_test` is the Brock-Dechert-Scheinkman test (Broock et al. 1996)
  of the i.i.d. null against any form of dependence.
* :func:`arch_lm_test` is Engle's (1982) Lagrange-multiplier test for ARCH
  effects.
* :func:`mcleod_li_test` is the McLeod-Li (1983) portmanteau test on squared
  residuals.

:func:`nonlinearity_report` runs all three together.
"""

import math

import numpy as np
from scipy.stats import chi2, f as f_dist, norm


def _clean(values, minimum, name):
    observed = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(observed)):
        raise ValueError(f"{name} must contain only finite numbers")
    if observed.size < minimum:
        raise ValueError(f"at least {minimum} observations are required")
    return observed


def _check_alpha(alpha):
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")


def _correlation_integrals(close, max_dim):
    """Return ``C_m`` for every ``m`` from 1 to ``max_dim``, plus the closeness matrices.

    ``close`` is the boolean matrix with entries ``|x_s - x_t| < eps``. Two
    m-histories ``(x_t, x_{t-1}, ..., x_{t-m+1})`` are close when every pair of
    coordinates is, so the m-dimensional matrix is the elementwise product
    of shifted copies of ``close``. ``C_m`` is the share of close pairs among
    the ``n_m (n_m - 1) / 2`` distinct pairs of m-histories.
    """

    n = close.shape[0]
    joint = close.copy()  # joint[i, j]: histories starting at i and j are close
    sums = []
    for m in range(1, max_dim + 1):
        if m > 1:
            joint = joint[1:, 1:] & close[: n - m + 1, : n - m + 1]
        size = joint.shape[0]
        pairs = size * (size - 1) / 2.0
        upper = (int(joint.sum()) - size) / 2.0  # off-diagonal pairs; the diagonal is always close
        sums.append(upper / pairs)
    return sums


def _bds_core(x, max_dim, eps):
    n = x.size
    close = np.abs(x[:, None] - x[None, :]) < eps
    c_m = _correlation_integrals(close, max_dim)
    c1_full = c_m[0]
    row = close.sum(axis=1).astype(float)
    # K = P(x_r, x_s and x_s, x_t both close) for distinct r, s, t
    k = (float(np.sum(row**2)) - 3.0 * float(row.sum()) + 2.0 * n) / (n * (n - 1.0) * (n - 2.0))
    stats = []
    for m in range(2, max_dim + 1):
        lead = m - 1
        sub = close[lead:, lead:]
        size = sub.shape[0]
        c1 = ((int(sub.sum()) - size) / 2.0) / (size * (size - 1) / 2.0)
        effect = c_m[m - 1] - c1**m
        inner = sum(k ** (m - j) * c1_full ** (2 * j) for j in range(1, m))
        variance = 4.0 * (
            k**m
            + 2.0 * inner
            + (m - 1) ** 2 * c1_full ** (2 * m)
            - m**2 * k * c1_full ** (2 * m - 2)
        )
        z = math.sqrt(size) * effect / math.sqrt(variance) if variance > 0 else float("nan")
        stats.append((m, effect, variance, z, c_m[m - 1], c1))
    return stats, c1_full, k


def bds_test(values, max_dim=3, epsilon=None, distance=1.5, n_permutations=0, seed=0, alpha=0.05):
    """Run the BDS test of the null that ``values`` are i.i.d.

    The test looks at the correlation integral ``C_m(eps)``, the share of
    pairs of m-histories that lie within ``eps`` of each other in every
    coordinate. Under independence ``C_m = C_1^m``. For embedding dimension
    ``m`` the statistic is

    ``W_m = sqrt(n - m + 1) (C_m - C_1^m) / sigma_m``,

    where ``sigma_m^2`` is the Broock et al. (1996) asymptotic variance, built
    from ``C_1`` and ``K``, the probability that a point is close to two
    others. ``W_m`` is asymptotically standard normal. ``C_1`` in the effect is
    computed on the observations used by the m-histories (Kanzler 1999).
    This matches ``statsmodels.tsa.stattools.bds``.

    ``epsilon`` defaults to ``distance`` times the sample standard
    deviation; ``0.5``-``2`` standard deviations is the usual range.

    The asymptotic normal approximation is oversized for short series, a
    few hundred observations or fewer. Setting ``n_permutations`` adds an
    exact-size permutation p-value: shuffling destroys all serial
    dependence while keeping the marginal distribution, so
    ``p = (1 + #{|W*| >= |W|}) / (1 + B)``.

    The test should be applied to *residuals* of a fitted linear model
    (e.g. ARIMA). Rejection then means there is dependence the linear
    model missed.
    """

    x = _clean(values, 20, "values")
    if isinstance(max_dim, bool) or not isinstance(max_dim, (int, np.integer)) or max_dim < 2:
        raise ValueError("max_dim must be an integer of at least 2")
    if max_dim > x.size // 4:
        raise ValueError("max_dim is too large for the series length")
    _check_alpha(alpha)
    if epsilon is None:
        if distance <= 0:
            raise ValueError("distance must be positive")
        sd = float(np.std(x, ddof=1))
        if sd == 0:
            raise ValueError("values have no variance")
        epsilon = distance * sd
    elif not epsilon > 0:
        raise ValueError("epsilon must be positive")
    if isinstance(n_permutations, bool) or not isinstance(n_permutations, (int, np.integer)) or n_permutations < 0:
        raise ValueError("n_permutations must be a non-negative integer")

    stats, c1_full, k = _bds_core(x, int(max_dim), float(epsilon))
    exceed = None
    if n_permutations:
        rng = np.random.default_rng(seed)
        exceed = np.zeros(len(stats))
        observed = np.array([abs(s[3]) for s in stats])
        for _ in range(int(n_permutations)):
            perm, _, _ = _bds_core(rng.permutation(x), int(max_dim), float(epsilon))
            exceed += np.array([abs(s[3]) for s in perm]) >= observed - 1e-12

    rows = []
    for i, (m, effect, variance, z, cm, c1) in enumerate(stats):
        row = {
            "dimension": m,
            "statistic": float(z),
            "p_value": float(2.0 * norm.sf(abs(z))),
            "correlation_integral": float(cm),
            "c1_power": float(c1**m),
            "effect": float(effect),
            "variance": float(variance),
        }
        if exceed is not None:
            row["p_value_permutation"] = float((1.0 + exceed[i]) / (1.0 + n_permutations))
        rows.append(row)
    key = "p_value_permutation" if exceed is not None else "p_value"
    min_p = min(r[key] for r in rows)
    return {
        "dimensions": rows,
        "epsilon": float(epsilon),
        "c1": float(c1_full),
        "k": float(k),
        "n_obs": int(x.size),
        "n_permutations": int(n_permutations),
        "p_value_used": key,
        # Bonferroni across embedding dimensions
        "reject_iid": bool(min(1.0, min_p * len(rows)) < alpha),
        "alpha": float(alpha),
    }


def arch_lm_test(residuals, lags=4, demean=False, alpha=0.05):
    """Engle (1982) Lagrange-multiplier test for ARCH effects.

    Regress ``e_t^2`` on a constant and ``e_{t-1}^2, ..., e_{t-q}^2``. Under
    the null of no conditional heteroskedasticity, ``LM = n R^2`` is
    asymptotically ``chi^2_q``, where ``n = T - q`` is the number of
    regression rows. The F form of the same regression is also returned.
    Set ``demean=True`` when the residuals are not already centred.
    Matches ``statsmodels.stats.diagnostic.het_arch`` with ``ddof=0``.
    """

    e = _clean(residuals, 10, "residuals")
    if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)) or lags < 1:
        raise ValueError("lags must be a positive integer")
    _check_alpha(alpha)
    if demean:
        e = e - e.mean()
    q = int(lags)
    T = e.size
    n = T - q
    if n <= q + 1:
        raise ValueError("too many lags for the series length")
    sq = e**2
    y = sq[q:]
    X = np.column_stack([np.ones(n)] + [sq[q - j:T - j] for j in range(1, q + 1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    centred = y - y.mean()
    ss_tot = float(centred @ centred)
    if ss_tot <= 0:
        raise ValueError("squared residuals have no variance")
    r2 = 1.0 - ss_res / ss_tot
    lm = n * r2
    df_resid = n - q - 1
    fstat = (r2 / q) / ((1.0 - r2) / df_resid) if r2 < 1 else float("inf")
    p = float(chi2.sf(lm, q))
    return {
        "lm_statistic": float(lm),
        "p_value": p,
        "f_statistic": float(fstat),
        "f_p_value": float(f_dist.sf(fstat, q, df_resid)),
        "lags": q,
        "nobs": int(n),
        "r_squared": float(r2),
        "coefficients": [float(b) for b in beta],
        "arch_effects": bool(p < alpha),
        "alpha": float(alpha),
    }


def mcleod_li_test(residuals, lags=10, alpha=0.05):
    """McLeod-Li (1983) test: Ljung-Box on squared, demeaned residuals.

    ``Q = T (T + 2) sum_{k=1}^{h} r_k^2 / (T - k)``, where ``r_k`` is the lag-``k``
    autocorrelation of ``e_t^2``. Under the null of no ARCH effects ``Q`` is
    asymptotically ``chi^2_h``. When the squares come from ARMA residuals the
    degrees of freedom are not reduced (McLeod & Li 1983).
    """

    e = _clean(residuals, 10, "residuals")
    if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)) or lags < 1:
        raise ValueError("lags must be a positive integer")
    _check_alpha(alpha)
    T = e.size
    if lags >= T:
        raise ValueError("lags must be smaller than the series length")
    sq = (e - e.mean()) ** 2
    d = sq - sq.mean()
    denom = float(d @ d)
    if denom <= 0:
        raise ValueError("squared residuals have no variance")
    rows = []
    q = 0.0
    for k in range(1, int(lags) + 1):
        r = float(d[k:] @ d[:-k]) / denom
        q += r * r / (T - k)
        stat = T * (T + 2.0) * q
        rows.append({"lag": k, "autocorrelation": r, "statistic": stat, "p_value": float(chi2.sf(stat, k))})
    final = rows[-1]
    return {
        "lags": rows,
        "statistic": final["statistic"],
        "p_value": final["p_value"],
        "arch_effects": bool(final["p_value"] < alpha),
        "alpha": float(alpha),
    }


def nonlinearity_report(residuals, max_dim=3, arch_lags=4, mcleod_lags=10, n_permutations=0, seed=0, alpha=0.05):
    """Run the BDS, ARCH-LM and McLeod-Li tests together and give a verdict."""

    e = _clean(residuals, 20, "residuals")
    centred = e - e.mean()
    bds = bds_test(centred, max_dim=max_dim, n_permutations=n_permutations, seed=seed, alpha=alpha)
    arch = arch_lm_test(centred, lags=arch_lags, alpha=alpha)
    ml = mcleod_li_test(centred, lags=mcleod_lags, alpha=alpha)
    if arch["arch_effects"] or ml["arch_effects"]:
        verdict = "conditional heteroskedasticity (volatility clustering); consider GARCH-type variance or intervals"
    elif bds["reject_iid"]:
        verdict = "nonlinear dependence in the mean; consider threshold, regime or ML models"
    else:
        verdict = "no evidence against i.i.d. residuals"
    return {"bds": bds, "arch_lm": arch, "mcleod_li": ml, "verdict": verdict}
