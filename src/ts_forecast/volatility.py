"""GARCH-family conditional volatility models.

``nonlinearity.arch_lm_test`` and ``mcleod_li_test`` can show that residuals
or returns have volatility clustering. This module models it. It provides the
GARCH(p, q) model of Bollerslev (1986) and its GJR / threshold extension of
Glosten, Jagannathan & Runkle (1993)::

    r_t = mu + eps_t,   eps_t = sigma_t z_t,   z_t iid with mean 0, variance 1
    sigma_t^2 = omega + sum_i (alpha_i + gamma_i 1[eps_{t-i} < 0]) eps_{t-i}^2
                      + sum_j beta_j sigma_{t-j}^2

``z_t`` is standard normal or a standardized Student-t with ``nu > 2`` degrees
of freedom. Parameters are fitted by (quasi-)maximum likelihood with SLSQP
under ``omega > 0``, non-negative coefficients and covariance stationarity
``sum alpha + sum gamma / 2 + sum beta < 1``. Pre-sample values of
``eps^2`` and ``sigma^2`` are backcast with the sample variance. The series is
rescaled to unit variance before optimisation and the parameters are mapped
back afterwards, so tiny daily returns optimise as well as percentages.

* :func:`fit_garch` returns parameters with numerical-Hessian standard errors,
  the log-likelihood, AIC/BIC, conditional volatility, standardized
  residuals, persistence, unconditional variance and volatility half-life.
* :func:`garch_forecast` gives analytic multi-step variance forecasts
  ``E_T[sigma_{T+h}^2]``, which mean-revert to the unconditional variance,
  and the variance of the ``h``-period aggregate return.
* :func:`garch_value_at_risk` gives one-period-ahead VaR and expected
  shortfall under the fitted innovation distribution, for each horizon.
* :func:`simulate_garch` simulates paths for testing and Monte Carlo work.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm, t as student_t

__all__ = ["simulate_garch", "fit_garch", "garch_forecast", "garch_value_at_risk"]


def _as_returns(returns):
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 10:
        raise ValueError("need at least 10 observations")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns must be finite")
    if not np.var(r) > 0:
        raise ValueError("returns must not be constant")
    return r


def _check_orders(p, q):
    if int(q) < 1 or int(p) < 0:
        raise ValueError("need q >= 1 ARCH lags and p >= 0 GARCH lags")
    return int(p), int(q)


def _variance_path(eps, omega, alpha, gamma, beta, backcast):
    """Conditional variances ``sigma_t^2`` for ``t = 0..n-1``."""
    n = eps.size
    q, p = alpha.size, beta.size
    e2 = eps * eps
    neg = (eps < 0).astype(float)
    s2 = np.empty(n)
    for t in range(n):
        v = omega
        for i in range(1, q + 1):
            if t - i >= 0:
                v += (alpha[i - 1] + gamma[i - 1] * neg[t - i]) * e2[t - i]
            else:
                v += (alpha[i - 1] + 0.5 * gamma[i - 1]) * backcast
        for j in range(1, p + 1):
            v += beta[j - 1] * (s2[t - j] if t - j >= 0 else backcast)
        s2[t] = v
    return s2


def _std_t_logpdf(z, nu):
    return (
        gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
        - (nu + 1) / 2 * np.log1p(z * z / (nu - 2))
    )


def _unpack(theta, p, q, asymmetric, mean, dist):
    k = 0
    mu = 0.0
    if mean == "constant":
        mu = theta[0]
        k = 1
    omega = theta[k]
    k += 1
    alpha = np.asarray(theta[k:k + q])
    k += q
    if asymmetric:
        gamma = np.asarray(theta[k:k + q])
        k += q
    else:
        gamma = np.zeros(q)
    beta = np.asarray(theta[k:k + p])
    k += p
    nu = theta[k] if dist == "t" else None
    return mu, omega, alpha, gamma, beta, nu


def _loglik_terms(theta, r, p, q, asymmetric, mean, dist, backcast):
    mu, omega, alpha, gamma, beta, nu = _unpack(theta, p, q, asymmetric, mean, dist)
    eps = r - mu
    s2 = _variance_path(eps, omega, alpha, gamma, beta, backcast)
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return None, None
    z = eps / np.sqrt(s2)
    if dist == "normal":
        ll = -0.5 * (np.log(2 * np.pi) + np.log(s2) + z * z)
    else:
        ll = _std_t_logpdf(z, nu) - 0.5 * np.log(s2)
    return ll, s2


def _numerical_hessian(f, x, rel=1e-4):
    k = x.size
    h = rel * np.maximum(np.abs(x), 1e-2)
    H = np.empty((k, k))
    f0 = f(x)
    for i in range(k):
        for j in range(i, k):
            if i == j:
                xp, xm = x.copy(), x.copy()
                xp[i] += h[i]
                xm[i] -= h[i]
                H[i, i] = (f(xp) - 2 * f0 + f(xm)) / h[i] ** 2
            else:
                vals = []
                for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    xx = x.copy()
                    xx[i] += si * h[i]
                    xx[j] += sj * h[j]
                    vals.append(f(xx))
                H[i, j] = H[j, i] = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * h[i] * h[j])
    return H


def fit_garch(returns, *, p=1, q=1, asymmetric=False, mean="constant", dist="normal"):
    """Fit GARCH(p, q), or GJR-GARCH with ``asymmetric=True``, by maximum likelihood.

    Parameters
    ----------
    returns : 1-D array-like of returns (or residuals).
    p, q : GARCH (lagged variance) and ARCH (lagged squared shock) orders.
    asymmetric : add GJR leverage terms ``gamma_i`` (bad news raises volatility more).
    mean : ``"constant"`` (estimate ``mu``) or ``"zero"``.
    dist : ``"normal"`` or ``"t"`` (standardized Student-t, ``nu`` estimated).

    Returns
    -------
    dict with ``params`` (name -> value), ``std_errors`` (numerical-Hessian,
    ``nan`` if the Hessian is not positive definite), ``loglik``, ``aic``,
    ``bic``, ``conditional_volatility``, ``standardized_residuals``,
    ``persistence``, ``unconditional_variance``, ``half_life`` (periods for a
    volatility shock to halve), ``converged``, and the internal state needed by
    :func:`garch_forecast`.
    """
    from scipy.optimize import minimize

    p, q = _check_orders(p, q)
    if mean not in ("constant", "zero"):
        raise ValueError("mean must be 'constant' or 'zero'")
    if dist not in ("normal", "t"):
        raise ValueError("dist must be 'normal' or 't'")
    r_raw = _as_returns(returns)
    if r_raw.size < 4 * (1 + p + 2 * q):
        raise ValueError("too few observations for the requested orders")
    scale = float(np.std(r_raw))
    r = r_raw / scale
    mu0 = float(np.mean(r)) if mean == "constant" else 0.0
    backcast = float(np.mean((r - mu0) ** 2))

    names = (["mu"] if mean == "constant" else []) + ["omega"]
    names += [f"alpha[{i}]" for i in range(1, q + 1)]
    if asymmetric:
        names += [f"gamma[{i}]" for i in range(1, q + 1)]
    names += [f"beta[{j}]" for j in range(1, p + 1)]
    if dist == "t":
        names.append("nu")

    def negll(theta):
        ll, _ = _loglik_terms(theta, r, p, q, asymmetric, mean, dist, backcast)
        return 1e10 if ll is None else -float(ll.sum())

    def persistence_of(theta):
        _, _, a, g, b, _ = _unpack(theta, p, q, asymmetric, mean, dist)
        return float(a.sum() + 0.5 * g.sum() + b.sum())

    bounds = ([(-10.0, 10.0)] if mean == "constant" else []) + [(1e-8, 10.0)]
    bounds += [(0.0, 1.0)] * q + ([(0.0, 1.0)] * q if asymmetric else []) + [(0.0, 1.0)] * p
    if dist == "t":
        bounds.append((2.05, 200.0))

    starts = []
    for a_tot, b_tot in ((0.05, 0.90), (0.10, 0.80), (0.20, 0.50), (0.03, 0.95)):
        if p == 0:
            a_tot, b_tot = a_tot + 0.5 * b_tot, 0.0
        theta = ([mu0] if mean == "constant" else []) + [max(1e-4, 1.0 - a_tot - b_tot)]
        if asymmetric:
            # split a_tot = alpha + gamma / 2 evenly between the two terms
            theta += [0.5 * a_tot / q] * q + [a_tot / q] * q
        else:
            theta += [a_tot / q] * q
        theta += [b_tot / max(p, 1)] * p
        if dist == "t":
            theta.append(8.0)
        starts.append(np.asarray(theta, dtype=float))
    starts.sort(key=negll)

    cons = [{"type": "ineq", "fun": lambda th: 1.0 - 1e-6 - persistence_of(th)}]
    best = None
    for x0 in starts[:2]:
        res = minimize(negll, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-11})
        if best is None or res.fun < best.fun:
            best = res
    theta = np.asarray(best.x, dtype=float)

    try:
        H = _numerical_hessian(negll, theta)
        cov = np.linalg.inv(H)
        se_scaled = np.sqrt(np.diag(cov)) if np.all(np.diag(cov) > 0) else np.full(theta.size, np.nan)
    except np.linalg.LinAlgError:
        se_scaled = np.full(theta.size, np.nan)

    # map back to the original scale: mu * s, omega * s^2, others unchanged
    factors = np.ones(theta.size)
    k = 0
    if mean == "constant":
        factors[0] = scale
        k = 1
    factors[k] = scale ** 2
    params_arr = theta * factors
    se_arr = se_scaled * factors

    ll_terms, s2_scaled = _loglik_terms(theta, r, p, q, asymmetric, mean, dist, backcast)
    loglik = float(ll_terms.sum()) - r.size * math.log(scale)
    mu, omega, alpha, gamma, beta, nu = _unpack(params_arr, p, q, asymmetric, mean, dist)
    s2 = s2_scaled * scale ** 2
    eps = r_raw - mu
    pers = float(alpha.sum() + 0.5 * gamma.sum() + beta.sum())
    n_par = theta.size
    return {
        "params": dict(zip(names, map(float, params_arr))),
        "std_errors": dict(zip(names, map(float, se_arr))),
        "loglik": loglik,
        "aic": float(-2 * loglik + 2 * n_par),
        "bic": float(-2 * loglik + math.log(r.size) * n_par),
        "n_obs": int(r.size),
        "conditional_volatility": np.sqrt(s2),
        "standardized_residuals": eps / np.sqrt(s2),
        "persistence": pers,
        "unconditional_variance": float(omega / (1 - pers)) if pers < 1 else float("inf"),
        "half_life": float(math.log(0.5) / math.log(pers)) if 0 < pers < 1 else float("inf"),
        "converged": bool(best.success),
        "spec": {"p": p, "q": q, "asymmetric": bool(asymmetric), "mean": mean, "dist": dist},
        "_eps": eps,
        "_sigma2": s2,
    }


def _spec_arrays(fitted):
    spec = fitted["spec"]
    prm = fitted["params"]
    p, q = spec["p"], spec["q"]
    alpha = np.array([prm[f"alpha[{i}]"] for i in range(1, q + 1)])
    gamma = np.array([prm[f"gamma[{i}]"] for i in range(1, q + 1)]) if spec["asymmetric"] else np.zeros(q)
    beta = np.array([prm[f"beta[{j}]"] for j in range(1, p + 1)])
    return prm.get("mu", 0.0), prm["omega"], alpha, gamma, beta, prm.get("nu")


def garch_forecast(fitted, steps=1):
    """Analytic ``h``-step variance forecasts from a :func:`fit_garch` result.

    Future squared shocks are replaced by their conditional expectation
    ``sigma^2`` (and ``1[eps < 0] eps^2`` by ``sigma^2 / 2``, since the
    innovations are symmetric), so the forecasts decay geometrically at rate
    ``persistence`` towards the unconditional variance.

    Returns ``mean``, ``variance``, ``volatility`` (arrays of length ``steps``)
    and ``cumulative_variance``, the variance of the sum of the next ``h``
    returns.
    """
    if int(steps) < 1:
        raise ValueError("steps must be at least 1")
    steps = int(steps)
    mu, omega, alpha, gamma, beta, _ = _spec_arrays(fitted)
    q, p = alpha.size, beta.size
    eps = list(fitted["_eps"])
    e2 = [e * e for e in eps]
    neg_e2 = [e * e if e < 0 else 0.0 for e in eps]
    s2 = list(fitted["_sigma2"])
    out = np.empty(steps)
    for h in range(steps):
        v = omega
        for i in range(1, q + 1):
            v += alpha[i - 1] * e2[-i] + gamma[i - 1] * neg_e2[-i]
        for j in range(1, p + 1):
            v += beta[j - 1] * s2[-j]
        out[h] = v
        s2.append(v)
        e2.append(v)
        neg_e2.append(0.5 * v)
    return {
        "mean": np.full(steps, mu),
        "variance": out,
        "volatility": np.sqrt(out),
        "cumulative_variance": np.cumsum(out),
    }


def garch_value_at_risk(fitted, steps=1, level=0.99):
    """One-period Value-at-Risk and expected shortfall for each of the next ``steps`` periods.

    Losses are reported as positive numbers: ``VaR_h = -(mu + sigma_h q_{1-level})``,
    where ``q`` is the quantile of the fitted standardized innovation
    distribution, and ``ES_h`` is the expected loss beyond ``VaR_h``.
    """
    if not 0.5 < level < 1.0:
        raise ValueError("level must lie in (0.5, 1)")
    fc = garch_forecast(fitted, steps)
    sigma = fc["volatility"]
    mu = fc["mean"]
    tail = 1.0 - level
    if fitted["spec"]["dist"] == "normal":
        z = norm.ppf(tail)
        es_std = norm.pdf(z) / tail
    else:
        nu = fitted["params"]["nu"]
        c = math.sqrt((nu - 2) / nu)
        tq = student_t.ppf(tail, nu)
        z = tq * c
        es_std = c * (nu + tq ** 2) / (nu - 1) * student_t.pdf(tq, nu) / tail
    return {
        "level": level,
        "volatility": sigma,
        "value_at_risk": -(mu + sigma * z),
        "expected_shortfall": -mu + sigma * es_std,
    }


def simulate_garch(n, omega, alpha, beta, gamma=None, mu=0.0, dist="normal", nu=8.0,
                   burn=500, seed=None):
    """Simulate ``n`` returns from a (GJR-)GARCH process.

    ``alpha``, ``beta`` and ``gamma`` may be scalars or sequences. Returns a
    dict with ``returns``, ``volatility`` and the innovations ``z``.
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    beta = np.atleast_1d(np.asarray(beta, dtype=float)) if np.size(beta) else np.zeros(0)
    gamma = np.zeros_like(alpha) if gamma is None else np.atleast_1d(np.asarray(gamma, dtype=float))
    if gamma.size != alpha.size:
        raise ValueError("gamma must have as many entries as alpha")
    if omega <= 0 or np.any(alpha < 0) or np.any(beta < 0) or np.any(gamma < 0):
        raise ValueError("omega must be positive and coefficients non-negative")
    pers = alpha.sum() + 0.5 * gamma.sum() + beta.sum()
    if pers >= 1:
        raise ValueError("process is not covariance stationary (persistence >= 1)")
    if dist not in ("normal", "t"):
        raise ValueError("dist must be 'normal' or 't'")
    if dist == "t" and nu <= 2:
        raise ValueError("nu must exceed 2")
    rng = np.random.default_rng(seed)
    total = int(n) + int(burn)
    if dist == "normal":
        z = rng.standard_normal(total)
    else:
        z = rng.standard_t(nu, total) * math.sqrt((nu - 2) / nu)
    uncond = omega / (1 - pers)
    q, p = alpha.size, beta.size
    eps = np.zeros(total)
    s2 = np.full(total, uncond)
    for t in range(total):
        v = omega
        for i in range(1, q + 1):
            e = eps[t - i] if t - i >= 0 else math.sqrt(uncond)
            g = gamma[i - 1] * (e < 0 if t - i >= 0 else 0.5)
            v += (alpha[i - 1] + g) * (e * e if t - i >= 0 else uncond)
        for j in range(1, p + 1):
            v += beta[j - 1] * (s2[t - j] if t - j >= 0 else uncond)
        s2[t] = v
        eps[t] = math.sqrt(v) * z[t]
    sl = slice(int(burn), total)
    return {"returns": mu + eps[sl], "volatility": np.sqrt(s2[sl]), "z": z[sl]}
