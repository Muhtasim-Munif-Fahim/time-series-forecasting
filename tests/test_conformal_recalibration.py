"""Tests for conformal interval recalibration."""

import numpy as np
import pytest

from ts_forecast.evaluation import recalibrate_conformal_intervals


def test_recalibrate_expands_intervals_when_heldout_undercovered():
    """When held-out miscoverage is high, intervals should widen (scale > 1)."""
    cal_true = [10.0, 20.0, 30.0, 40.0, 50.0]
    cal_pred = [9.0, 18.0, 33.0, 40.0, 52.0]
    ho_true = [60.0, 70.0, 80.0]
    ho_pred = [50.0, 60.0, 65.0]
    forecast = [90.0, 100.0]

    lower, upper, scale, emp_cov = recalibrate_conformal_intervals(
        cal_true, cal_pred, ho_true, ho_pred, forecast, coverage=0.8
    )

    assert scale > 1.0
    assert emp_cov < 0.8
    assert np.all(lower < upper)
    assert lower.shape == (2,)
    assert upper.shape == (2,)


def test_recalibrate_contracts_intervals_when_heldout_overcovered():
    """When held-out coverage exceeds nominal, intervals should narrow (scale < 1)."""
    cal_true = [10.0, 20.0, 30.0, 40.0, 50.0]
    cal_pred = [9.0, 18.0, 33.0, 40.0, 52.0]
    ho_true = [60.0, 70.0]
    ho_pred = [59.5, 69.8]
    forecast = [80.0, 90.0]

    lower, upper, scale, emp_cov = recalibrate_conformal_intervals(
        cal_true, cal_pred, ho_true, ho_pred, forecast, coverage=0.8
    )

    assert scale < 1.0
    assert emp_cov > 0.8
    assert np.all(lower < upper)


def test_recalibrate_preserves_center_on_point_forecast():
    """Recalibrated intervals should remain centered on the point forecast."""
    cal_true = [10.0, 20.0, 30.0, 40.0, 50.0]
    cal_pred = [9.0, 18.0, 33.0, 40.0, 52.0]
    ho_true = [60.0, 70.0]
    ho_pred = [59.5, 69.8]
    forecast = [80.0, 90.0]

    lower, upper, scale, emp_cov = recalibrate_conformal_intervals(
        cal_true, cal_pred, ho_true, ho_pred, forecast, coverage=0.8
    )

    centers = (lower + upper) / 2
    assert np.allclose(centers, forecast)


def test_recalibrate_validates_coverage_range():
    """Coverage must be strictly between 0 and 1."""
    with pytest.raises(ValueError, match="strictly between"):
        recalibrate_conformal_intervals(
            [1.0], [1.0], [2.0], [2.0], [3.0], coverage=0.0
        )
    with pytest.raises(ValueError, match="strictly between"):
        recalibrate_conformal_intervals(
            [1.0], [1.0], [2.0], [2.0], [3.0], coverage=1.0
        )


def test_recalibrate_validates_calibration_shapes():
    """Calibration true and pred must have equal length."""
    with pytest.raises(ValueError, match="equal length"):
        recalibrate_conformal_intervals(
            [1.0, 2.0], [1.0], [3.0], [3.0], [4.0]
        )


def test_recalibrate_validates_heldout_shapes():
    """Held-out true and pred must have equal length."""
    with pytest.raises(ValueError, match="equal length"):
        recalibrate_conformal_intervals(
            [1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0], [5.0]
        )


def test_recalibrate_requires_finite_calibration_data():
    """At least one finite calibration residual is required."""
    with pytest.raises(ValueError, match="finite calibration residual"):
        recalibrate_conformal_intervals(
            [np.nan], [np.nan], [3.0], [3.0], [4.0]
        )


def test_recalibrate_requires_finite_heldout_data():
    """Held-out set must contain at least one finite pair."""
    with pytest.raises(ValueError, match="held-out set must contain"):
        recalibrate_conformal_intervals(
            [1.0, 2.0], [1.0, 2.0], [np.nan], [np.nan], [4.0]
        )


def test_recalibrate_ignores_nonfinite_calibration_pairs():
    """Non-finite calibration pairs should be ignored."""
    cal_true = [1.0, np.nan, 3.0, 4.0, 5.0]
    cal_pred = [0.0, 2.0, 3.5, 4.0, 5.5]
    ho_true = [6.0, 7.0]
    ho_pred = [5.9, 7.1]
    forecast = [8.0]

    lower, upper, scale, emp_cov = recalibrate_conformal_intervals(
        cal_true, cal_pred, ho_true, ho_pred, forecast, coverage=0.8
    )
    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))


def test_recalibrate_ignores_nonfinite_heldout_pairs():
    """Non-finite held-out pairs should be ignored."""
    cal_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    cal_pred = [0.0, 2.0, 3.5, 4.0, 5.5]
    ho_true = [6.0, np.nan, 8.0]
    ho_pred = [5.9, 7.0, 7.9]
    forecast = [9.0]

    lower, upper, scale, emp_cov = recalibrate_conformal_intervals(
        cal_true, cal_pred, ho_true, ho_pred, forecast, coverage=0.8
    )
    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))


def test_recalibrate_handles_zero_radius():
    """When calibration radius is zero, scale factor should be 1."""
    cal_true = [5.0, 5.0, 5.0, 5.0, 5.0]
    cal_pred = [5.0, 5.0, 5.0, 5.0, 5.0]
    ho_true = [6.0, 7.0]
    ho_pred = [5.9, 7.1]
    forecast = [8.0]

    lower, upper, scale, emp_cov = recalibrate_conformal_intervals(
        cal_true, cal_pred, ho_true, ho_pred, forecast, coverage=0.8
    )
    assert scale == 1.0
    assert np.allclose(lower, forecast)
    assert np.allclose(upper, forecast)


def test_recalibrate_returns_empirical_coverage():
    """Empirical coverage should match held-out scores within radius."""
    cal_true = [10.0, 20.0, 30.0, 40.0, 50.0]
    cal_pred = [9.0, 18.0, 33.0, 40.0, 52.0]
    ho_true = [60.0, 70.0, 80.0]
    ho_pred = [59.0, 69.0, 81.0]
    forecast = [90.0]

    lower, upper, scale, emp_cov = recalibrate_conformal_intervals(
        cal_true, cal_pred, ho_true, ho_pred, forecast, coverage=0.8
    )
    cal_scores = np.abs(np.array(cal_true) - np.array(cal_pred))
    quantile_level = min(1.0, np.ceil((5 + 1) * 0.8) / 5)
    radius = float(np.quantile(cal_scores, quantile_level, method="higher"))
    ho_scores = np.abs(np.array(ho_true) - np.array(ho_pred))
    expected_emp = float(np.mean(ho_scores <= radius))
    assert emp_cov == pytest.approx(expected_emp)
