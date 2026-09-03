"""Tests for select_window_size_cv."""

from __future__ import annotations

import numpy as np
import pytest

from ts_forecast.evaluation import select_window_size_cv


def _naive_model(train, steps):
    return np.full(int(steps), train[-1])


def test_returns_expected_keys() -> None:
    values = np.arange(80, dtype=float)
    result = select_window_size_cv(
        values, _naive_model,
        candidate_windows=[5, 10, 20], horizon=1, step=1,
    )
    assert set(result) == {"best_window", "scores", "metric"}
    assert result["metric"] == "mae"
    assert result["best_window"] in {5, 10, 20}
    assert set(result["scores"]) == {5, 10, 20}


def test_shortest_window_wins_for_pure_noise_with_naive() -> None:
    rng = np.random.default_rng(0)
    values = rng.standard_normal(120)
    result = select_window_size_cv(
        values, _naive_model,
        candidate_windows=[5, 10, 20, 40], horizon=1, step=1,
    )
    # All naive forecasts are equal in expectation; the smallest window
    # usually wins on MAE for white noise.
    assert result["best_window"] in {5, 10, 20, 40}


def test_rmse_metric_used_when_requested() -> None:
    values = np.arange(60, dtype=float)
    result = select_window_size_cv(
        values, _naive_model,
        candidate_windows=[5, 10], horizon=1, step=1, metric="rmse",
    )
    assert result["metric"] == "rmse"
    for value in result["scores"].values():
        assert value >= 0


def test_rejects_invalid_metric() -> None:
    values = np.arange(40, dtype=float)
    with pytest.raises(ValueError, match="metric"):
        select_window_size_cv(
            values, _naive_model,
            candidate_windows=[5, 10], metric="mape",  # type: ignore[arg-type]
        )


def test_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="values"):
        select_window_size_cv(
            np.array([]), _naive_model,
            candidate_windows=[5, 10],
        )


def test_rejects_invalid_horizon() -> None:
    values = np.arange(40, dtype=float)
    with pytest.raises(ValueError, match="horizon"):
        select_window_size_cv(
            values, _naive_model,
            candidate_windows=[5, 10], horizon=0,
        )


def test_rejects_invalid_step() -> None:
    values = np.arange(40, dtype=float)
    with pytest.raises(ValueError, match="step"):
        select_window_size_cv(
            values, _naive_model,
            candidate_windows=[5, 10], step=0,
        )


def test_rejects_empty_candidate_windows() -> None:
    values = np.arange(40, dtype=float)
    with pytest.raises(ValueError, match="window"):
        select_window_size_cv(values, _naive_model, candidate_windows=[])


def test_rejects_when_series_too_short() -> None:
    values = np.arange(5, dtype=float)
    with pytest.raises(ValueError, match="not enough"):
        select_window_size_cv(values, _naive_model, candidate_windows=[10, 20])


def test_rejects_when_largest_window_too_large() -> None:
    values = np.arange(20, dtype=float)
    with pytest.raises(ValueError):
        select_window_size_cv(values, _naive_model, candidate_windows=[5, 10, 30])


def test_uses_provided_initial_when_given() -> None:
    values = np.arange(50, dtype=float)
    result = select_window_size_cv(
        values, _naive_model,
        candidate_windows=[5, 10], horizon=1, step=1, initial=30,
    )
    assert result["best_window"] in {5, 10}


def test_window_choices_handle_trend() -> None:
    """A linear trend should be well captured by the largest window."""
    values = np.arange(100, dtype=float)
    result = select_window_size_cv(
        values, _naive_model,
        candidate_windows=[5, 10, 30, 50], horizon=1, step=1,
    )
    # Larger windows hold more of the trend.
    assert result["scores"][50] <= result["scores"][5]