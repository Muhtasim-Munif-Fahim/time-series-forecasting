"""Engle-Granger cointegration test and error-correction models.

Two non-stationary series are *cointegrated* when a linear combination of
them is stationary: they wander, but never drift apart for long (prices of
close substitutes, spot vs. futures, income vs. consumption). The
Engle & Granger (1987) two-step procedure:

1. Estimate the long-run relation ``y_t = a (+ b t) + beta' x_t + u_t`` by
   OLS.
2. Run an augmented Dickey-Fuller regression *without* deterministic terms
   on the residuals, ``du_t = rho u_{t-1} + sum_i phi_i du_{t-i} + e_t``.
   ``rho`` significantly below zero means ``u_t`` is stationary, i.e. the
   series are cointegrated. Because ``beta`` was estimated, the ordinary
   Dickey-Fuller tables do not apply; p-values and critical values use
   MacKinnon's (1994, 2010) response surfaces for ``N = 1 + dim(x)``
   variables.

:func:`error_correction_model` then fits the short-run dynamics

``dy_t = c + alpha u_{t-1} + sum_{j=0}^{p} b_j dx_{t-j} + sum_{j=1}^{p} g_j dy_{t-j} + e_t``

whose ``alpha < 0`` is the speed of adjustment back to equilibrium, with
half-life ``log(0.5) / log(1 + alpha)`` periods.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["engle_granger", "error_correction_model", "ecm_forecast"]

_TRENDS = ("c", "ct", "n")


def _as_xy(y, x):
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x must have the same number of observations as y")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
        raise ValueError("y and x must be finite (drop missing values first)")
    return y, x


def _ols(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    ssr = float(resid @ resid)
    dof = max(n - k, 1)
    sigma2 = ssr / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    llf = -0.5 * n * (math.log(2 * math.pi) + math.log(max(ssr, 1e-300) / n) + 1.0)
    return coef, se, resid, llf


def _deterministic(n, trend):
    cols = []
    if trend in ("c", "ct"):
        cols.append(np.ones(n))
    if trend == "ct":
        cols.append(np.arange(1, n + 1, dtype=float))
    return np.column_stack(cols) if cols else np.empty((n, 0))


def _adf_design(u, lags, start):
    """ADF regression (no deterministics) using observations t >= start."""
    du = np.diff(u)
    rows = np.arange(start, du.shape[0])
    X = [u[rows]]  # u_{t-1} aligned with du_t = u_{t+1} - u_t
    for i in range(1, lags + 1):
        X.append(du[rows - i])
    return np.column_stack(X), du[rows]


def _adf_no_const(u, maxlag, autolag):
    if autolag is None:
        X, z = _adf_design(u, maxlag, maxlag)
        coef, se, _, _ = _ols(X, z)
        return coef[0] / se[0], maxlag, z.shape[0]
    # select the lag on a common sample, then refit on the longest sample
    best_lag, best = 0, np.inf
    for lag in range(maxlag + 1):
        X, z = _adf_design(u, lag, maxlag)
        coef, se, resid, llf = _ols(X, z)
        k = X.shape[1]
        crit = -2 * llf + (2 * k if autolag == "aic" else k * math.log(z.shape[0]))
        if crit < best - 1e-12:
            best, best_lag = crit, lag
    X, z = _adf_design(u, best_lag, best_lag)
    coef, se, _, _ = _ols(X, z)
    return coef[0] / se[0], best_lag, z.shape[0]


def engle_granger(y, x, *, trend="c", maxlag=None, autolag="aic", alpha=0.05):
    """Engle-Granger two-step cointegration test of ``y`` on ``x``.

    Parameters
    ----------
    y : array-like of shape (n,)
    x : array-like of shape (n,) or (n, k)
    trend : {"c", "ct", "n"}
        Deterministic terms in the long-run regression.
    maxlag : int, optional
        Largest ADF augmentation lag; defaults to ``12 * (n / 100) ** 0.25``.
    autolag : {"aic", "bic", None}
        Lag selection criterion; ``None`` uses ``maxlag`` lags.
    alpha : float
        Level for the ``cointegrated`` decision.

    Returns
    -------
    dict
        ``statistic`` (ADF t on residuals), ``p_value``, ``critical_values``
        (1/5/10 %; NaN for ``trend="n"``), ``cointegrated``, ``lags``, ``nobs``, ``long_run``
        (coefficients on ``x``), ``deterministic`` (intercept / trend),
        ``residuals`` (the equilibrium error ``u_t``) and ``trend``.
    """
    if trend not in _TRENDS:
        raise ValueError(f"trend must be one of {_TRENDS}")
    if autolag not in ("aic", "bic", None):
        raise ValueError("autolag must be 'aic', 'bic' or None")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    y, x = _as_xy(y, x)
    n, k = x.shape
    if maxlag is None:
        maxlag = int(math.ceil(12.0 * (n / 100.0) ** 0.25))
        maxlag = min(n // 2 - k - 2, maxlag)
    maxlag = int(maxlag)
    if maxlag < 0:
        raise ValueError("maxlag must be non-negative")
    if n < max(20, 2 * maxlag + k + 5):
        raise ValueError("series too short for the requested lag length")
    D = _deterministic(n, trend)
    X = np.column_stack([D, x])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("long-run regressors are collinear")
    coef, _, u, _ = _ols(X, y)
    stat, lags, nobs = _adf_no_const(u, maxlag, autolag)

    from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp

    p_value = float(mackinnonp(stat, regression=trend, N=k + 1))
    if trend == "n":
        crit = [float("nan")] * 3  # no MacKinnon (2010) tables without deterministics
    else:
        crit = mackinnoncrit(N=k + 1, regression=trend, nobs=n - 1)
    return {
        "statistic": float(stat),
        "p_value": p_value,
        "critical_values": {"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])},
        "cointegrated": bool(p_value < alpha),
        "lags": int(lags),
        "nobs": int(nobs),
        "long_run": coef[D.shape[1]:].copy(),
        "deterministic": coef[: D.shape[1]].copy(),
        "residuals": u,
        "trend": trend,
    }


def error_correction_model(y, x, *, lags=1, trend="c"):
    """Fit an Engle-Granger two-step error-correction model.

    Parameters
    ----------
    y, x : array-like
        Levels of the (cointegrated) target and regressors.
    lags : int
        Number ``p`` of lagged differences of ``y`` and ``x`` (contemporaneous
        ``dx_t`` is always included).
    trend : {"c", "ct", "n"}
        Deterministic terms in the long-run regression.

    Returns
    -------
    dict
        ``speed_of_adjustment`` (``alpha``), its ``alpha_t_value``,
        ``half_life`` (periods; ``inf`` when ``alpha >= 0``), ``long_run``,
        ``deterministic``, ``params`` / ``std_errors`` / ``t_values`` keyed by
        name (``const``, ``ec``, ``dx0_l0``, ``dx0_l1``, ..., ``dy_l1``, ...),
        ``residual_std``, ``nobs``, ``lags`` and ``trend``.
    """
    if trend not in _TRENDS:
        raise ValueError(f"trend must be one of {_TRENDS}")
    lags = int(lags)
    if lags < 0:
        raise ValueError("lags must be non-negative")
    y, x = _as_xy(y, x)
    n, k = x.shape
    if n < 2 * (lags + 1) + 2 * k + 6:
        raise ValueError("series too short for the requested number of lags")
    D = _deterministic(n, trend)
    X_lr = np.column_stack([D, x])
    coef_lr, _, u, _ = _ols(X_lr, y)

    dy = np.diff(y)
    dx = np.diff(x, axis=0)
    t_idx = np.arange(lags, dy.shape[0])  # dy index; dy[t] = y[t+1] - y[t]
    cols, names = [np.ones(t_idx.size)], ["const"]
    cols.append(u[t_idx])  # u at the level date preceding dy[t]
    names.append("ec")
    for j in range(k):
        for lag in range(lags + 1):
            cols.append(dx[t_idx - lag, j])
            names.append(f"dx{j}_l{lag}")
    for lag in range(1, lags + 1):
        cols.append(dy[t_idx - lag])
        names.append(f"dy_l{lag}")
    Z = np.column_stack(cols)
    target = dy[t_idx]
    coef, se, resid, _ = _ols(Z, target)
    tvals = np.divide(coef, se, out=np.full_like(coef, np.nan), where=se > 0)
    a = float(coef[1])
    half_life = math.log(0.5) / math.log(1.0 + a) if -1.0 < a < 0.0 else (
        float("inf") if a >= 0 else float("nan")
    )
    return {
        "speed_of_adjustment": a,
        "alpha_t_value": float(tvals[1]),
        "half_life": half_life,
        "long_run": coef_lr[D.shape[1]:].copy(),
        "deterministic": coef_lr[: D.shape[1]].copy(),
        "params": dict(zip(names, map(float, coef))),
        "std_errors": dict(zip(names, map(float, se))),
        "t_values": dict(zip(names, map(float, tvals))),
        "residual_std": float(np.sqrt(resid @ resid / max(target.size - Z.shape[1], 1))),
        "nobs": int(target.size),
        "lags": lags,
        "trend": trend,
        "_history": (y, x),
    }


def ecm_forecast(model, x_future):
    """Point-forecast ``y`` from a fitted ECM given a future path of ``x``.

    The recursion uses the long-run relation for the equilibrium error and
    the short-run equation for ``dy``; future shocks are set to zero.

    Parameters
    ----------
    model : dict
        Output of :func:`error_correction_model`.
    x_future : array-like of shape (h,) or (h, k)
        Regressor levels for the ``h`` forecast periods.
    """
    y_hist, x_hist = model["_history"]
    x_future = np.asarray(x_future, dtype=float)
    if x_future.ndim == 1:
        x_future = x_future.reshape(-1, 1)
    if x_future.ndim != 2 or x_future.shape[1] != x_hist.shape[1]:
        raise ValueError("x_future must have the same number of columns as x")
    p = model["params"]
    lags, k, trend = model["lags"], x_hist.shape[1], model["trend"]
    beta, det = model["long_run"], model["deterministic"]
    y = list(y_hist)
    x = np.vstack([x_hist, x_future])
    n0 = len(y_hist)
    out = []
    for h in range(x_future.shape[0]):
        t = n0 + h  # index of the level being forecast
        prev = t - 1
        lr = float(x[prev] @ beta)
        if trend in ("c", "ct"):
            lr += det[0]
        if trend == "ct":
            lr += det[1] * (prev + 1)
        ec = y[prev] - lr
        dy_hat = p["const"] + p["ec"] * ec
        for j in range(k):
            for lag in range(lags + 1):
                dy_hat += p[f"dx{j}_l{lag}"] * (x[t - lag, j] - x[t - lag - 1, j])
        for lag in range(1, lags + 1):
            dy_hat += p[f"dy_l{lag}"] * (y[t - lag] - y[t - lag - 1])
        y.append(y[prev] + dy_hat)
        out.append(y[-1])
    return np.asarray(out)
