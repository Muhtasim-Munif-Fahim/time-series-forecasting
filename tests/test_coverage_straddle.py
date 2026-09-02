"""Tests for the coverage_straddle_test feature."""

from __future__ import annotations

import numpy as np
import pytest

from ts_forecast.evaluation import coverage_straddle_test


def _intervals(values):
    return [(low, high) for low, high in values]


def test_flags_undercoverage_with_significant_p_value() -> None:
    # 100 intervals of which only 70 cover -> 0.7 empirical vs 0.9 nominal.
    intervals = _intervals([(0.0, 1.0)] * 70 + [(2.0, 1.0)] * 30)
    result = coverage_straddle_test(intervals, nominal=0.9, alpha=0.05)
    assert result["n"] == 100
    assert result["observed_coverage"] == pytest.approx(0.7)
    assert result["verdict"] == "under-covered"
    assert result["p_value"] < 0.001
    assert result["z"] < 0


def test_flags_overcoverage_with_significant_p_value() -> None:
    intervals = _intervals([(0.0, 1.0)] * 100)
    result = coverage_straddle_test(intervals, nominal=0.5, alpha=0.05)
    assert result["observed_coverage"] == pytest.approx(1.0)
    assert result["verdict"] == "over-covered"
    assert result["p_value"] < 0.001


def test_reports_on_target_when_empirical_matches_nominal() -> None:
    intervals = _intervals([(0.0, 1.0)] * 90 + [(2.0, 1.0)] * 10)
    result = coverage_straddle_test(intervals, nominal=0.9, alpha=0.05)
    assert result["observed_coverage"] == pytest.approx(0.9)
    assert result["verdict"] == "on-target"
    assert result["p_value"] > 0.05


def test_returns_insufficient_when_below_min_n() -> None:
    intervals = _intervals([(0.0, 1.0)] * 5)
    result = coverage_straddle_test(intervals, min_n=10)
    assert result["n"] == 5
    assert result["verdict"] == "insufficient"
    assert np.isnan(result["z"])
    assert np.isnan(result["p_value"])


def test_skips_non_finite_intervals() -> None:
    intervals = [(0.0, 1.0), (np.nan, 1.0), (0.0, np.inf), (float("nan"), float("inf"))]
    result = coverage_straddle_test(intervals, nominal=0.9, min_n=4)
    # Only one interval has finite bounds, so covered=1/4=0.25 vs nominal=0.9
    assert result["observed_coverage"] == pytest.approx(0.25)
    assert result["verdict"] == "under-covered"


def test_rejects_invalid_nominal() -> None:
    intervals = [(0.0, 1.0)] * 10
    with pytest.raises(ValueError, match="nominal must be strictly between 0 and 1"):
        coverage_straddle_test(intervals, nominal=0.0)
    with pytest.raises(ValueError, match="nominal must be strictly between 0 and 1"):
        coverage_straddle_test(intervals, nominal=1.0)
    with pytest.raises(ValueError, match="nominal must be strictly between 0 and 1"):
        coverage_straddle_test(intervals, nominal=-0.1)
    with pytest.raises(ValueError, match="nominal must be strictly between 0 and 1"):
        coverage_straddle_test(intervals, nominal=1.5)


def test_rejects_invalid_alpha() -> None:
    intervals = [(0.0, 1.0)] * 10
    with pytest.raises(ValueError, match="alpha"):
        coverage_straddle_test(intervals, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        coverage_straddle_test(intervals, alpha=1.0)


def test_rejects_invalid_min_n() -> None:
    with pytest.raises(ValueError, match="min_n"):
        coverage_straddle_test([(0.0, 1.0)] * 5, min_n=0)


def test_default_alpha_does_not_break_minimal_input() -> None:
    intervals = _intervals([(0.0, 1.0)] * 90 + [(2.0, 1.0)] * 10)
    result = coverage_straddle_test(intervals, nominal=0.9)
    assert "z" in result and "p_value" in result and "verdict" in result
    assert result["verdict"] == "on-target"


def test_handles_empty_interval_list() -> None:
    result = coverage_straddle_test([], min_n=5)
    assert result["n"] == 0
    assert result["verdict"] == "insufficient"