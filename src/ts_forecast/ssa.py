"""Singular Spectrum Analysis (SSA): decomposition and recurrent forecasting.

SSA (Golyandina, Nekrutkin & Zhigljavsky, 2001) is a non-parametric method
that splits a series into trend, oscillatory and noise components without
fixing a season length in advance.

1. **Embedding.** The series ``x_0 ... x_{N-1}`` is put into the ``L x K``
   Hankel trajectory matrix ``X[i, j] = x_{i+j}``, with ``K = N - L + 1``.
2. **SVD.** ``X = sum_i s_i u_i v_i^T``. Each *eigentriple*
   ``(s_i, u_i, v_i)`` gives an elementary matrix. Its share of the total
   is ``s_i^2 / sum s^2``.
3. **Diagonal averaging.** Each elementary matrix is mapped back to a
   series by averaging its anti-diagonals. The anti-diagonal sums of
   ``u v^T`` are ``np.convolve(u, v)``, so this costs ``O(N L)`` per component.
   The elementary series add up exactly to the original.
4. **Grouping.** Components belong together when their
   **w-correlation** ``(a, b)_w / (|a|_w |b|_w)`` is high, with weights
   ``w_k = min(k + 1, L, K, N - k)``. A sine wave appears as a pair of
   strongly w-correlated components with similar singular values.

**Recurrent forecasting (R-forecasting).** The leading ``r`` left singular
vectors span the signal subspace. Let ``pi`` be their last coordinates and
``nu^2 = |pi|^2 < 1``. The reconstructed signal then satisfies the linear
recurrence ``g_t = sum_k a_k g_{t-L+k}`` with
``a = (1 / (1 - nu^2)) sum_i pi_i u_i[:-1]``, which is iterated forward.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ssa_decompose",
    "w_correlation",
    "fit_ssa",
    "ssa_forecast",
]


def _series(values) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size < 4:
        raise ValueError("SSA needs at least four observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite (impute missing values first)")
    return x


def _window(n: int, window) -> int:
    if window is None:
        return n // 2
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise ValueError("window must be an integer")
    window = int(window)
    if not 2 <= window <= n - 1:
        raise ValueError(f"window must be in [2, {n - 1}] for a series of length {n}")
    return window


def _trajectory(x: np.ndarray, L: int) -> np.ndarray:
    K = x.size - L + 1
    return np.lib.stride_tricks.sliding_window_view(x, K)[:L].copy()


def _hankelize(u: np.ndarray, v: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.convolve(u, v) / counts


def _weights(n: int, L: int) -> np.ndarray:
    K = n - L + 1
    k = np.arange(n)
    return np.minimum.reduce([k + 1, np.full(n, min(L, K)), n - k]).astype(float)


def w_correlation(components, window: int) -> np.ndarray:
    """Weighted correlation matrix of reconstructed series (rows of ``components``)."""
    C = np.atleast_2d(np.asarray(components, dtype=float))
    n = C.shape[1]
    L = _window(n, window)
    w = _weights(n, L)
    gram = (C * w) @ C.T
    norms = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    denom = np.outer(norms, norms)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, gram / denom, 0.0)
    np.fill_diagonal(out, 1.0)
    return out


def ssa_decompose(values, window=None, n_components=None, groups=None):
    """Decompose a series into SSA elementary components.

    Parameters
    ----------
    values : array-like
        The series. It must be finite.
    window : int, optional
        Embedding window ``L``. The default is ``N // 2``. A multiple of the
        dominant period separates seasonal pairs best.
    n_components : int, optional
        How many leading elementary series to reconstruct (default: all).
    groups : dict[str, sequence[int]], optional
        Named groups of component indices. Each group's series is the sum of
        its components.

    Returns
    -------
    dict
        ``singular_values``, ``contribution`` (share of ``sum s^2``),
        ``components`` (``(r, N)`` elementary series), ``w_correlation``
        (``(r, r)``), ``window`` and ``groups`` (name to reconstructed series).
    """
    x = _series(values)
    n = x.size
    L = _window(n, window)
    X = _trajectory(x, L)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    rank = s.size
    if n_components is None:
        r = rank
    else:
        if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)) or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        r = min(int(n_components), rank)
    counts = np.convolve(np.ones(L), np.ones(n - L + 1))
    comps = np.vstack([_hankelize(U[:, i] * s[i], Vt[i], counts) for i in range(r)])
    total = float(np.sum(s * s))
    contribution = s * s / total if total > 0 else np.zeros_like(s)
    out_groups = {}
    if groups is not None:
        for name, idx in dict(groups).items():
            idx = [int(i) for i in idx]
            if not idx or min(idx) < 0 or max(idx) >= r:
                raise ValueError(f"group {name!r} must list component indices in [0, {r - 1}]")
            out_groups[name] = comps[idx].sum(axis=0)
    return {
        "singular_values": s,
        "contribution": contribution,
        "components": comps,
        "w_correlation": w_correlation(comps, L),
        "window": L,
        "groups": out_groups,
    }


def _choose_rank(s: np.ndarray, rank, variance_threshold: float) -> int:
    if rank is not None:
        if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank < 1:
            raise ValueError("rank must be a positive integer")
        return min(int(rank), s.size)
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in (0, 1]")
    share = np.cumsum(s * s) / np.sum(s * s)
    return int(np.searchsorted(share, variance_threshold - 1e-12) + 1)


def fit_ssa(train, target_col, *, window=None, rank=None, variance_threshold=0.9):
    """Fit an SSA signal model with ``rank`` leading components.

    If ``rank`` is None, it is the smallest ``r`` whose components explain at
    least ``variance_threshold`` of ``sum s^2``. Returns a dict with the
    reconstruction (``fitted``), the recurrence coefficients ``lrf`` (length
    ``L - 1``, oldest lag first), ``verticality`` ``nu^2``, the residual
    ``sigma``, ``rank`` and ``window``.
    """
    x = _series(train[target_col].dropna())
    n = x.size
    L = _window(n, window)
    U, s, Vt = np.linalg.svd(_trajectory(x, L), full_matrices=False)
    r = _choose_rank(s, rank, variance_threshold)
    counts = np.convolve(np.ones(L), np.ones(n - L + 1))
    fitted = np.zeros(n)
    for i in range(r):
        fitted += _hankelize(U[:, i] * s[i], Vt[i], counts)
    pi = U[-1, :r]
    nu2 = float(pi @ pi)
    if nu2 >= 1.0 - 1e-10:
        raise ValueError(
            "verticality nu^2 is 1: the signal subspace cannot be continued; "
            "lower the rank or change the window"
        )
    lrf = (U[:-1, :r] @ pi) / (1.0 - nu2)
    resid = x - fitted
    dof = max(n - r, 1)
    return {
        "fitted": fitted,
        "lrf": lrf,
        "verticality": nu2,
        "sigma": float(np.sqrt(resid @ resid / dof)),
        "rank": r,
        "window": L,
        "singular_values": s,
        "n_obs": n,
    }


def ssa_forecast(train, target_col, steps=1, window=None, rank=None, variance_threshold=0.9):
    """Recurrent SSA forecast for the next ``steps`` points."""
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    model = fit_ssa(train, target_col, window=window, rank=rank, variance_threshold=variance_threshold)
    a = model["lrf"]
    m = a.size
    buf = list(model["fitted"][-m:]) if m else []
    out = np.empty(int(steps))
    for h in range(int(steps)):
        nxt = float(np.dot(a, buf[-m:])) if m else 0.0
        out[h] = nxt
        buf.append(nxt)
    return out
