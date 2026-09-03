"""Tests for granger_causality_matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_forecast.evaluation import granger_causality_matrix


def test_matrix_shape_and_self_entries_are_nan() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "a": rng.standard_normal(60).cumsum(),
        "b": rng.standard_normal(60).cumsum(),
    })
    result = granger_causality_matrix(frame, columns=["a", "b"], max_lag=2)
    assert result["columns"] == ["a", "b"]
    matrix = result["p_value_matrix"]
    assert len(matrix) == 2
    assert all(len(row) == 2 for row in matrix)
    assert matrix[0][0] != matrix[0][0]  # NaN
    assert matrix[1][1] != matrix[1][1]  # NaN


def test_returns_significant_matrix_and_edges() -> None:
    rng = np.random.default_rng(0)
    n = 200
    cause = rng.standard_normal(n)
    # Effect is a lagged function of the cause plus noise.
    effect = np.zeros(n)
    for i in range(1, n):
        effect[i] = 0.7 * cause[i - 1] + 0.1 * effect[i - 1] + rng.standard_normal()
    frame = pd.DataFrame({"cause": cause, "effect": effect})
    result = granger_causality_matrix(frame, columns=["cause", "effect"], max_lag=2)
    p_value = result["p_value_matrix"][1][0]  # cause -> effect
    assert p_value < 0.05
    edges = result["edges"]
    assert any(edge[0] == "cause" and edge[1] == "effect" for edge in edges)


def test_rejects_non_dataframe() -> None:
    with pytest.raises(ValueError, match="DataFrame"):
        granger_causality_matrix("not a frame", columns=["a"])  # type: ignore[arg-type]


def test_rejects_invalid_max_lag() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="max_lag"):
        granger_causality_matrix(frame, columns=["a"], max_lag=0)


def test_rejects_invalid_significance() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="significance"):
        granger_causality_matrix(frame, columns=["a"], significance=0.0)
    with pytest.raises(ValueError, match="significance"):
        granger_causality_matrix(frame, columns=["a"], significance=1.0)


def test_rejects_unknown_columns() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="unknown columns"):
        granger_causality_matrix(frame, columns=["a", "ghost"])


def test_handles_short_series_gracefully() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = granger_causality_matrix(frame, columns=["a"], max_lag=4)
    # Too short to compute a meaningful p-value; entries are NaN.
    assert all(value != value for value in result["p_value_matrix"][0])


def test_handles_nan_drops() -> None:
    rng = np.random.default_rng(1)
    n = 200
    cause = rng.standard_normal(n)
    effect = np.roll(cause, 1) + rng.standard_normal(n) * 0.1
    frame = pd.DataFrame({"cause": cause, "effect": effect})
    frame.loc[5:10, "cause"] = float("nan")
    result = granger_causality_matrix(frame, columns=["cause", "effect"], max_lag=3)
    assert result["edges"]  # at least one significant edge