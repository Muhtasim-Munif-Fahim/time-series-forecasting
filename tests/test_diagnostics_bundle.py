"""Tests for the forecast_diagnostics_bundle feature."""

from __future__ import annotations

import numpy as np
import pytest

from ts_forecast.evaluation import (
    forecast_diagnostics_bundle,
    forecast_skill_score,
)


def test_bundle_contains_all_diagnostic_pieces() -> None:
    rng = np.random.default_rng(1)
    y_true = np.arange(1.0, 16.0)
    y_pred = y_true + rng.standard_normal(y_true.size) * 0.1
    bundle = forecast_diagnostics_bundle(y_true, y_pred, max_lag=3)
    assert "metrics" in bundle
    assert "mae" in bundle["metrics"]
    assert "rmse" in bundle["metrics"]
    assert "mape" in bundle["metrics"]
    assert "bias" in bundle
    assert "smape" in bundle
    assert "residual_autocorrelation" in bundle
    assert "ljung_box" in bundle
    assert "skill_score" in bundle
    assert "min_residual_p_value" in bundle


def test_bundle_seasonal_naive_baseline_differs_from_naive() -> None:
    # The skill score is 100*(1 - model_loss / baseline_loss). With a
    # seasonal predictor (y_pred == deterministic seasonal pattern) and a
    # noisy series, the naive baseline will be a much worse reference than
    # the seasonal-naive baseline. The seasonal-naive bundle should report
    # a smaller (less positive) skill score than the naive bundle because
    # the seasonal-naive baseline loss is closer to the model's loss.
    rng = np.random.default_rng(7)
    base = np.tile(np.array([1.0, 2.0, 3.0]), 8).astype(float)
    y_true = base + rng.standard_normal(base.size) * 0.05
    y_pred = base
    seasonal_bundle = forecast_diagnostics_bundle(
        y_true, y_pred, seasonal_period=3, baseline="seasonal_naive", train=y_true, max_lag=3
    )
    naive_bundle = forecast_diagnostics_bundle(
        y_true, y_pred, seasonal_period=3, baseline="naive", max_lag=3
    )
    assert seasonal_bundle["skill_score"] < naive_bundle["skill_score"]


def test_bundle_rejects_invalid_arguments() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="seasonal_period"):
        forecast_diagnostics_bundle(y_true, y_pred, seasonal_period=0)
    with pytest.raises(ValueError, match="max_lag"):
        forecast_diagnostics_bundle(y_true, y_pred, max_lag=0)
    with pytest.raises(ValueError, match="alpha"):
        forecast_diagnostics_bundle(y_true, y_pred, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        forecast_diagnostics_bundle(y_true, y_pred, alpha=1.0)
    with pytest.raises(ValueError, match="baseline"):
        forecast_diagnostics_bundle(y_true, y_pred, baseline="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="train is required"):
        forecast_diagnostics_bundle(y_true, y_pred, baseline="seasonal_naive")


def test_bundle_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        forecast_diagnostics_bundle(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])
        )


def test_bundle_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        forecast_diagnostics_bundle(np.array([]), np.array([]))


def test_bundle_skill_score_matches_direct_call() -> None:
    rng = np.random.default_rng(0)
    y_true = np.arange(1.0, 12.0)
    y_pred = y_true + 0.1 + rng.standard_normal(y_true.size) * 0.05
    naive = np.concatenate([[y_true[0]], y_true[:-1]])
    bundle = forecast_diagnostics_bundle(y_true, y_pred, max_lag=3)
    assert bundle["skill_score"] == pytest.approx(
        forecast_skill_score(y_true, y_pred, naive, score="mae")
    )


def test_bundle_ljung_box_white_noise_returns_high_p() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.standard_normal(200)
    y_pred = y_true.copy() + rng.standard_normal(200) * 0.05
    bundle = forecast_diagnostics_bundle(y_true, y_pred, max_lag=5)
    assert bundle["min_residual_p_value"] > 0.05