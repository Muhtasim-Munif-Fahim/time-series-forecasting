"""Model Confidence Set (Hansen, Lunde & Nason, 2011).

Given a matrix of per-period forecast losses for several competing models,
the Model Confidence Set (MCS) is the subset of models that contains the
best model(s) with a given confidence level. Models are removed one at a
time by an equivalence test whose null is "all remaining models have equal
expected loss"; the test is bootstrapped with a moving-block bootstrap so
that serially correlated loss differentials are handled.

Two test statistics are supported:

* ``"max"`` -- ``T_max = max_i d_i. / se(d_i.)`` where ``d_i.`` is the mean
  loss of model ``i`` relative to the average of the surviving models.
* ``"range"`` -- ``T_R = max_{i,j} |d_ij| / se(d_ij)`` over all pairs.

Each model receives an MCS p-value; a model belongs to the
``(1 - alpha)`` MCS iff its MCS p-value is at least ``alpha``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_loss_matrix(losses, names=None):
    if isinstance(losses, pd.DataFrame):
        labels = [str(c) for c in losses.columns]
        matrix = losses.to_numpy(dtype=float)
    elif isinstance(losses, dict):
        labels = [str(k) for k in losses]
        columns = [np.asarray(v, dtype=float).ravel() for v in losses.values()]
        if len({c.size for c in columns}) > 1:
            raise ValueError("all loss series must have equal length")
        matrix = np.column_stack(columns) if columns else np.empty((0, 0))
    else:
        matrix = np.asarray(losses, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("losses must be a 2-D array (n_periods, n_models)")
        labels = [f"model_{i}" for i in range(matrix.shape[1])]
    if names is not None:
        labels = [str(n) for n in names]
        if len(labels) != matrix.shape[1]:
            raise ValueError("names must have one entry per model")
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("at least two models are required")
    if matrix.shape[0] < 2:
        raise ValueError("at least two periods are required")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("losses must be finite")
    return matrix, labels


def moving_block_bootstrap_indices(n_periods, block_length, n_boot, seed=None):
    """Return ``(n_boot, n_periods)`` moving-block bootstrap time indices.

    Blocks of ``block_length`` consecutive periods are drawn with
    replacement (circularly wrapped) and concatenated, then truncated to
    ``n_periods``.
    """

    if isinstance(block_length, bool) or not isinstance(block_length, (int, np.integer)):
        raise ValueError("block_length must be a positive integer")
    if block_length < 1 or block_length > n_periods:
        raise ValueError("block_length must be between 1 and n_periods")
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_periods / block_length))
    starts = rng.integers(0, n_periods, size=(n_boot, n_blocks))
    offsets = np.arange(block_length)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n_periods
    return idx.reshape(n_boot, -1)[:, :n_periods]


def model_confidence_set(
    losses,
    alpha=0.10,
    statistic="max",
    block_length=None,
    n_boot=1000,
    seed=None,
    names=None,
):
    """Compute the Hansen-Lunde-Nason Model Confidence Set.

    Parameters
    ----------
    losses : array-like (n_periods, n_models), DataFrame or dict
        Per-period loss of each model (e.g. squared or absolute errors).
        Lower is better.
    alpha : float
        Size of the test; the returned set is the ``(1 - alpha)`` MCS.
    statistic : {"max", "range"}
        Equivalence test statistic (``T_max`` or ``T_R``).
    block_length : int, optional
        Moving-block bootstrap block length. Defaults to
        ``max(1, round(n_periods ** (1/3)))``.
    n_boot : int
        Bootstrap replications.
    seed : int, optional
        Random seed for reproducibility.
    names : sequence of str, optional
        Model labels (overrides DataFrame columns / dict keys).

    Returns
    -------
    dict with keys ``included`` (labels in the MCS, best mean loss first),
    ``eliminated`` (labels in elimination order), ``pvalues`` (label ->
    MCS p-value), ``mean_loss`` (label -> average loss), ``alpha``,
    ``statistic`` and ``table`` (a DataFrame sorted by elimination order
    with the surviving models last).
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if statistic not in {"max", "range"}:
        raise ValueError("statistic must be 'max' or 'range'")
    matrix, labels = _as_loss_matrix(losses, names)
    n_periods, n_models = matrix.shape
    if block_length is None:
        block_length = max(1, int(round(n_periods ** (1.0 / 3.0))))
    idx = moving_block_bootstrap_indices(n_periods, block_length, n_boot, seed)

    mean_loss = matrix.mean(axis=0)
    # Bootstrapped mean losses for every model: (n_boot, n_models).
    boot_mean = matrix[idx].mean(axis=1)

    alive = list(range(n_models))
    eliminated = []
    pvalues = {}
    running_p = 0.0
    tiny = 1e-12

    while len(alive) > 1:
        a = np.array(alive)
        lbar = mean_loss[a]
        bbar = boot_mean[:, a]
        if statistic == "max":
            d = lbar - lbar.mean()
            d_boot = bbar - bbar.mean(axis=1, keepdims=True)
            var = np.mean((d_boot - d) ** 2, axis=0)
            se = np.sqrt(np.maximum(var, tiny))
            t = d / se
            observed = float(t.max())
            boot_stat = ((d_boot - d) / se).max(axis=1)
            worst = int(np.argmax(t))
        else:
            d = lbar[:, None] - lbar[None, :]
            d_boot = bbar[:, :, None] - bbar[:, None, :]
            var = np.mean((d_boot - d) ** 2, axis=0)
            se = np.sqrt(np.maximum(var, tiny))
            np.fill_diagonal(se, 1.0)
            t = d / se
            observed = float(np.abs(t).max())
            boot_stat = np.abs((d_boot - d) / se).max(axis=(1, 2))
            worst = int(np.argmax(t.max(axis=1)))
        p = float(np.mean(boot_stat >= observed))
        running_p = max(running_p, p)
        if running_p >= alpha:
            break
        model = alive.pop(worst)
        eliminated.append(model)
        pvalues[labels[model]] = running_p

    # Models never eliminated share the final p-value (1.0 if only one left).
    final_p = 1.0 if len(alive) == 1 else running_p
    for model in alive:
        pvalues[labels[model]] = final_p

    included = sorted(alive, key=lambda i: mean_loss[i])
    order = eliminated + included
    table = pd.DataFrame(
        {
            "model": [labels[i] for i in order],
            "mean_loss": [float(mean_loss[i]) for i in order],
            "mcs_pvalue": [pvalues[labels[i]] for i in order],
            "in_mcs": [i in alive for i in order],
        }
    )
    return {
        "included": [labels[i] for i in included],
        "eliminated": [labels[i] for i in eliminated],
        "pvalues": {labels[i]: pvalues[labels[i]] for i in range(n_models)},
        "mean_loss": {labels[i]: float(mean_loss[i]) for i in range(n_models)},
        "alpha": float(alpha),
        "statistic": statistic,
        "block_length": int(block_length),
        "table": table,
    }


def forecast_losses(y_true, forecasts, loss="squared"):
    """Build a per-period loss DataFrame from a dict of model forecasts."""

    if loss not in {"squared", "absolute"}:
        raise ValueError("loss must be 'squared' or 'absolute'")
    observed = np.asarray(y_true, dtype=float).ravel()
    out = {}
    for name, pred in forecasts.items():
        pred = np.asarray(pred, dtype=float).ravel()
        if pred.shape != observed.shape:
            raise ValueError(f"forecast {name!r} length differs from y_true")
        err = observed - pred
        out[str(name)] = err**2 if loss == "squared" else np.abs(err)
    return pd.DataFrame(out)
