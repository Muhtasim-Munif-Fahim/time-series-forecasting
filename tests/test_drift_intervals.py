"""Tests for prediction_intervals_with_drift."""

from __future__ import annotations

import numpy as np
import pytest

from ts_forecast.evaluation import prediction_intervals_with_drift


def test_returns_expected_keys() -> None:
    rng = np.random.default_rng(0)
    cal_true = rng.standard_normal(80)
    cal_pred = cal_true + rng.standard_normal(80) * 0.5
    ho_true = rng.standard_normal(40)
    ho_pred = ho_true + rng.standard_normal(40) * 0.5
    forecast = np.array([0.0, 0.5])
    result = prediction_intervals_with_drift(
        cal_true, cal_pred, ho_true, ho_pred, forecast, coverage=0.9,
    )
    assert set(result) == {
        "lower", "upper", "radius", "calibration_radius", "drift_scale",
        "drift_detected", "t_statistic", "p_value",
    }
    assert (result["upper"] > result["lower"]).all()


def test_drift_detected_when_held_out_errors_larger() -> None:
    rng = np.random.default_rng(0)
    cal_true = rng.standard_normal(200)
    cal_pred = cal_true + rng.standard_normal(200) * 0.1
    ho_true = rng.standard_normal(100)
    ho_pred = ho_true + rng.standard_normal(100) * 0.5  # larger noise
    forecast = np.array([0.0])
    result = prediction_intervals_with_drift(
        cal_true, cal_pred, ho_true, ho_pred, forecast, coverage=0.9,
    )
    assert result["drift_detected"] is True
    assert result["drift_scale"] > 1.0
    assert result["radius"] > result["calibration_radius"]


def test_no_drift_when_distributions_match() -> None:
    rng = np.random.default_rng(0)
    cal_true = rng.standard_normal(200)
    cal_pred = cal_true + rng.standard_normal(200) * 0.2
    ho_true = rng.standard_normal(200)
    ho_pred = ho_true + rng.standard_normal(200) * 0.2
    forecast = np.array([0.0])
    result = prediction_intervals_with_drift(
        cal_true, cal_pred, ho_true, ho_pred, forecast, coverage=0.9,
    )
    assert result["drift_detected"] is False
    assert result["drift_scale"] == 1.0


def test_rejects_invalid_coverage() -> None:
    rng = np.random.default_rng(0)
    cal = rng.standard_normal(40)
    with pytest.raises(ValueError, match="coverage"):
        prediction_intervals_with_drift(cal, cal, cal, cal, np.array([0.0]), coverage=0.0)


def test_rejects_invalid_drift_alpha() -> None:
    rng = np.random.default_rng(0)
    cal = rng.standard_normal(40)
    with pytest.raises(ValueError, match="drift_alpha"):
        prediction_intervals_with_drift(cal, cal, cal, cal, np.array([0.0]), drift_alpha=0.0)
    with pytest.raises(ValueError, match="drift_alpha"):
        prediction_intervals_with_drift(cal, cal, cal, cal, np.array([0.0]), drift_alpha=1.0)


def test_rejects_length_mismatch() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="calibration"):
        prediction_intervals_with_drift(
            np.array([0.0, 0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
        )
    with pytest.raises(ValueError, match="heldout"):
        prediction_intervals_with_drift(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0]),
            np.array([0.0]),
        )


def test_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        prediction_intervals_with_drift(
            np.array([]), np.array([]),
            np.array([0.0]), np.array([0.0]),
            np.array([0.0]),
        )


def test_drift_scale_floors_at_one_when_calibration_is_zero() -> None:
    cal_true = np.array([1.0, 2.0, 3.0, 4.0])
    cal_pred = np.array([1.0, 2.0, 3.0, 4.0])  # zero error
    ho_true = np.array([5.0, 6.0, 7.0])
    ho_pred = np.array([10.0, 11.0, 12.0])  # much larger error
    result = prediction_intervals_with_drift(
        cal_true, cal_pred, ho_true, ho_pred, np.array([0.0]), coverage=0.9,
    )
    assert result["drift_scale"] >= 1.0


def test_radius_and_calibration_radius_when_drift_floored() -> None:
    cal_true = np.array([1.0, 2.0, 3.0, 4.0])
    cal_pred = np.array([1.0, 2.0, 3.0, 4.0])
    ho_true = np.array([5.0, 6.0, 7.0])
    ho_pred = np.array([10.0, 11.0, 12.0])
    result = prediction_intervals_with_drift(
        cal_true, cal_pred, ho_true, ho_pred, np.array([0.0]), coverage=0.9,
    )
    # calibration_radius is 0 because the calibration set has zero error
    assert result["calibration_radius"] == 0.0
    assert result["radius"] >= 0.0