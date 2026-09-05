"""Tests for the bias-variance decomposition of forecast ensembles."""

import numpy as np
import pytest

from ts_forecast.evaluation import bias_variance_decomposition


def test_decomposition_is_additive():
    rng = np.random.default_rng(0)
    observed = np.arange(8.0)
    predictions = np.column_stack(
        [observed + rng.normal(0, 1, 8) for _ in range(5)]
    )
    result = bias_variance_decomposition(observed, predictions)
    assert result["mean_mse"] == pytest.approx(
        result["squared_bias"] + result["variance"]
    )


def test_zero_variance_for_single_member_predictions():
    observed = np.array([1.0, 2.0, 3.0])
    predictions = np.array([[2.0], [4.0], [6.0]])
    result = bias_variance_decomposition(observed, predictions)
    assert result["variance"] == 0.0
    assert result["mean_mse"] == pytest.approx(result["squared_bias"])


def test_zero_bias_when_members_are_centered_on_truth():
    observed = np.array([0.0, 0.0, 0.0])
    predictions = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    result = bias_variance_decomposition(observed, predictions)
    assert result["squared_bias"] == pytest.approx(0.0)
    assert result["mean_mse"] == pytest.approx(result["variance"])


def test_wider_ensemble_has_higher_variance_for_same_center():
    observed = np.zeros(4)
    sharp = np.tile([-1.0, 1.0], (4, 1))
    wide = np.tile([-10.0, 10.0], (4, 1))
    sharp_score = bias_variance_decomposition(observed, sharp)
    wide_score = bias_variance_decomposition(observed, wide)
    assert wide_score["variance"] > sharp_score["variance"]
    assert wide_score["mean_mse"] > sharp_score["mean_mse"]


def test_mean_mse_matches_directly_computed_error():
    observed = np.array([3.0, 5.0])
    predictions = np.array([[1.0, 5.0], [4.0, 8.0]])
    result = bias_variance_decomposition(observed, predictions)
    direct = np.mean(np.mean((predictions - observed[:, None]) ** 2, axis=1))
    assert result["mean_mse"] == pytest.approx(direct)


def test_validates_predictions_shape():
    with pytest.raises(ValueError, match="2-D array"):
        bias_variance_decomposition([1.0, 2.0], np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="equal length"):
        bias_variance_decomposition([1.0, 2.0, 3.0], np.array([[1.0, 2.0]]))


def test_validates_finiteness_and_members():
    with pytest.raises(ValueError, match="finite"):
        bias_variance_decomposition([1.0, np.nan], np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="finite"):
        bias_variance_decomposition([1.0, 2.0], np.array([[1.0, np.inf], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="at least one forecast"):
        bias_variance_decomposition([1.0, 2.0], np.empty((2, 0)))
