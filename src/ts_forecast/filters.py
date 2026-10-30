"""Trend-cycle filters from macroeconomics: Hodrick-Prescott, Baxter-King,
Christiano-Fitzgerald and Hamilton.

Each filter splits a series ``x`` into ``trend + cycle``.

* :func:`hp_filter`: Hodrick & Prescott (1997). The trend minimises
  ``sum (x_t - tau_t)^2 + lambda sum (Delta^2 tau_t)^2``, which gives the
  pentadiagonal system ``(I + lambda D'D) tau = x``. It is solved in
  ``O(n)`` with a banded Cholesky factorisation. :func:`hp_lambda` gives the
  Ravn-Uhlig (2002) rule ``lambda = 1600 (f / 4)^4`` for ``f`` observations
  per year (6.25 annual, 1600 quarterly, 129600 monthly).
* :func:`bk_filter`: Baxter & King (1999). A symmetric moving-average
  approximation, truncated at ``K`` leads and lags, to the ideal band-pass
  filter for periods ``[low, high]``. The weights are re-centred so they sum
  to zero and are symmetric. That removes linear trends exactly and turns a
  quadratic trend into a constant, so I(2) series come out stationary. The
  first and last ``K``
  values are lost (returned as ``nan``).
* :func:`cf_filter`: Christiano & Fitzgerald (2003). The full-sample,
  asymmetric band-pass filter that is optimal under a random-walk model. It
  keeps every observation and removes the drift ``(x_n - x_1) / (n - 1)``
  first when ``drift=True``.
* :func:`hamilton_filter`: Hamilton (2018), "Why you should never use the
  Hodrick-Prescott filter". OLS regression of ``x_{t+h}`` on a constant and
  ``x_t, ..., x_{t-p+1}``. The residual is the cycle and the fitted value is
  the trend. For quarterly data the defaults are ``h = 8`` and ``p = 4``.
  The first ``h + p - 1`` values are ``nan``.

All filters take a 1-D array-like and return a dict with ``trend`` and
``cycle`` arrays of the input length. :func:`cycle_summary` reports the
amplitude and turning points of a cycle.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solveh_banded

__all__ = [
    "hp_lambda",
    "hp_filter",
    "bk_filter",
    "cf_filter",
    "hamilton_filter",
    "cycle_summary",
    "CYCLE_FILTERS",
]


def _series(values, minimum: int) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size < minimum:
        raise ValueError(f"need at least {minimum} observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite (impute missing values first)")
    return x


def _band(low, high):
    low, high = float(low), float(high)
    if low < 2:
        raise ValueError("low must be at least 2 periods (the Nyquist limit)")
    if high <= low:
        raise ValueError("high must exceed low")
    # angular frequencies: a passes long cycles (period high), b short (low)
    return 2.0 * np.pi / high, 2.0 * np.pi / low


def hp_lambda(periods_per_year: float) -> float:
    """Ravn-Uhlig smoothing parameter ``1600 * (f / 4) ** 4``."""
    f = float(periods_per_year)
    if f <= 0:
        raise ValueError("periods_per_year must be positive")
    return 1600.0 * (f / 4.0) ** 4


def hp_filter(values, lamb: float = 1600.0) -> dict:
    """Hodrick-Prescott trend and cycle (``cycle = x - trend``)."""
    x = _series(values, 3)
    lamb = float(lamb)
    if lamb < 0:
        raise ValueError("lamb must be non-negative")
    n = x.size
    # Upper banded form of A = I + lamb * D'D, with D the (n-2) x n second difference.
    d0 = np.full(n, 6.0)
    d0[[0, -1]] = 1.0
    d0[[1, -2]] = 5.0
    if n == 3:
        d0 = np.array([1.0, 4.0, 1.0])
    d1 = np.full(n - 1, -4.0)
    d1[[0, -1]] = -2.0
    d2 = np.ones(n - 2)
    ab = np.zeros((3, n))
    ab[2] = 1.0 + lamb * d0
    ab[1, 1:] = lamb * d1
    ab[0, 2:] = lamb * d2
    trend = solveh_banded(ab, x)
    return {"trend": trend, "cycle": x - trend, "lambda": lamb}


def _ideal_weights(a: float, b: float, count: int) -> np.ndarray:
    j = np.arange(1, count + 1)
    return np.r_[(b - a) / np.pi, (np.sin(b * j) - np.sin(a * j)) / (np.pi * j)]


def bk_filter(values, low: float = 6, high: float = 32, K: int = 12) -> dict:
    """Baxter-King band-pass cycle; the ``K`` edge values on each side are ``nan``."""
    a, b = _band(low, high)
    K = int(K)
    if K < 1:
        raise ValueError("K must be positive")
    x = _series(values, 2 * K + 1)
    w = _ideal_weights(a, b, K)
    weights = np.r_[w[:0:-1], w]  # lags K..1, 0, leads 1..K
    weights = weights - weights.sum() / weights.size
    cycle = np.full(x.size, np.nan)
    cycle[K : x.size - K] = np.convolve(x, weights, mode="valid")
    return {"trend": x - cycle, "cycle": cycle, "weights": weights[K:]}


def cf_filter(values, low: float = 6, high: float = 32, drift: bool = True) -> dict:
    """Christiano-Fitzgerald asymmetric random-walk band-pass filter.

    With ``drift=True`` the line through the first and last observations is
    removed first; it is added back into ``trend``.
    """
    a, b = _band(low, high)
    x0 = _series(values, 4)
    n = x0.size
    line = np.arange(n) * (x0[-1] - x0[0]) / (n - 1) if drift else np.zeros(n)
    x = x0 - line
    B = _ideal_weights(a, b, n)
    csum = np.cumsum(B)  # csum[m] = B_0 + ... + B_m
    cycle = np.empty(n)
    for t in range(n):
        f, back = n - 1 - t, t  # observations after / before t
        fwd = csum[f - 1] - B[0] if f >= 1 else 0.0  # sum_{j=1}^{f-1} B_j
        bwd = csum[back - 1] - B[0] if back >= 1 else 0.0  # sum_{j=1}^{back-1} B_j
        end_f = -0.5 * B[0] - fwd
        end_b = -(B[0] + fwd + end_f + bwd)
        val = B[0] * x[t] + end_f * x[-1] + end_b * x[0]
        if f >= 2:
            val += B[1:f] @ x[t + 1 : t + f]
        if back >= 2:
            val += B[1:back] @ x[t - 1 : 0 : -1]
        cycle[t] = val
    return {"trend": x0 - cycle, "cycle": cycle}


def hamilton_filter(values, h: int = 8, p: int = 4) -> dict:
    """Hamilton (2018) regression filter.

    The returned ``trend`` is aligned to the predicted date ``t + h``.
    ``coefficients`` holds ``[const, beta_0, ..., beta_{p-1}]``, where
    ``beta_k`` multiplies ``x_{t-k}``.
    """
    h, p = int(h), int(p)
    if h < 1 or p < 1:
        raise ValueError("h and p must be positive")
    x = _series(values, h + p + p + 2)
    n = x.size
    rows = np.arange(p - 1, n - h)  # t indices
    X = np.column_stack([np.ones(rows.size)] + [x[rows - k] for k in range(p)])
    y = x[rows + h]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    trend = np.full(n, np.nan)
    trend[rows + h] = X @ coef
    return {"trend": trend, "cycle": x - trend, "coefficients": coef}


def cycle_summary(cycle) -> dict:
    """Standard deviation, peaks and troughs (strict local extrema) of a cycle."""
    c = np.asarray(cycle, dtype=float).reshape(-1)
    idx = np.flatnonzero(np.isfinite(c))
    v = c[idx]
    if v.size < 3:
        raise ValueError("need at least three finite cycle values")
    inner = np.arange(1, v.size - 1)
    peaks = idx[inner[(v[inner] > v[inner - 1]) & (v[inner] > v[inner + 1])]]
    troughs = idx[inner[(v[inner] < v[inner - 1]) & (v[inner] < v[inner + 1])]]
    return {
        "std": float(np.std(v, ddof=1)),
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(v.size),
        "peaks": peaks.tolist(),
        "troughs": troughs.tolist(),
    }


CYCLE_FILTERS = {"hp": hp_filter, "bk": bk_filter, "cf": cf_filter, "hamilton": hamilton_filter}
