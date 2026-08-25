"""Tests for signed skill scores against reference forecasts."""

import numpy as np
import pytest

from ts_forecast.evaluation import forecast_skill_score


def test_perfect_forecast_reaches_full_skill():
    observed = np.array([10.0, 12.0, 9.0, 14.0])
    baseline = np.full(4, 11.0)
    skill = forecast_skill_score(observed, observed.copy(), baseline)
    assert skill == 100.0


def test_positive_skill_marks_improvement_over_baseline():
    observed = np.array([10.0, 20.0, 30.0])
    baseline = np.array([12.0, 18.0, 27.0])
    predicted = np.array([11.0, 19.0, 28.5])
    skill = forecast_skill_score(observed, predicted, baseline, score="mae")
    assert 0.0 < skill < 100.0
    expected = 100.0 * (1.0 - (3.5 / 3.0) / (7.0 / 3.0))
    assert skill == pytest.approx(expected)


def test_negative_skill_marks_loss_to_baseline():
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    baseline = observed + 1.0
    predicted = observed + 3.0
    skill = forecast_skill_score(observed, predicted, baseline)
    assert skill == pytest.approx(-200.0)


def test_parity_scores_zero():
    observed = np.array([5.0, 6.0, 7.0])
    same = np.array([4.0, 7.0, 8.0])
    assert forecast_skill_score(observed, same, same) == pytest.approx(0.0)


def test_rmse_option_penalises_large_misses():
    observed = np.full(4, 5.0)
    baseline = np.array([6.0, 4.0, 6.0, 4.0])
    predicted = np.array([4.0, 6.0, 4.0, 20.0])
    mae_skill = forecast_skill_score(observed, predicted, baseline, score="mae")
    rmse_skill = forecast_skill_score(observed, predicted, baseline, score="rmse")
    assert mae_skill == pytest.approx(-350.0)
    assert rmse_skill == pytest.approx(100.0 * (1.0 - np.sqrt(57.0)))
    assert rmse_skill < mae_skill


def test_accepts_naive_and_seasonal_naive_references():
    steps = np.arange(8, dtype=float)
    observed = 10.0 + steps + 3.0 * np.sin(2.0 * np.pi * steps / 4.0)
    naive = np.full(observed.size, observed[-1])
    seasonal = np.resize(observed[-4:], observed.size)
    model = observed + 0.25
    assert forecast_skill_score(observed, model, naive) > 0
    assert forecast_skill_score(observed, model, seasonal) > 0


def test_validates_inputs():
    observed = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="score must be"):
        forecast_skill_score(observed, observed, observed, score="mse")

    with pytest.raises(ValueError, match="equal length"):
        forecast_skill_score(observed, observed, observed[:1])

    with pytest.raises(ValueError, match="at least one observation"):
        forecast_skill_score([], [], [])

    with pytest.raises(ValueError, match="finite and non-zero"):
        forecast_skill_score(observed, observed, observed)