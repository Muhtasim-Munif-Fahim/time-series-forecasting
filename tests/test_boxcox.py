"""Tests for the Box-Cox transform and its forecast pipeline wrapper."""

import numpy as np
import pandas as pd
import pytest

from ts_forecast.preprocessing import (
    boxcox_transform,
    inverse_boxcox,
    boxcox_forecast,
)


def _frame(values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)})


def test_estimated_lambda_is_near_zero_for_log_normal_data():
    rng = np.random.default_rng(7)
    values = np.exp(rng.normal(size=300))
    transformed, lmbda = boxcox_transform(values)
    assert isinstance(lmbda, float)
    assert np.abs(lmbda) < 0.5
    assert transformed.shape == values.shape
    assert np.all(np.isfinite(transformed))


def test_transform_inverse_roundtrip():
    rng = np.random.default_rng(11)
    values = rng.uniform(0.5, 20.0, size=100)
    transformed, lmbda = boxcox_transform(values)
    restored = inverse_boxcox(transformed, lmbda)
    np.testing.assert_allclose(restored, values, rtol=1e-10, atol=1e-10)


def test_explicit_lambda_zero_uses_log():
    values = np.linspace(1.0, 5.0, 20)
    transformed, lmbda = boxcox_transform(values, lmbda=0.0)
    assert lmbda == 0.0
    np.testing.assert_allclose(transformed, np.log(values))
    np.testing.assert_allclose(inverse_boxcox(transformed, 0.0), values)


def test_explicit_lambda_one_is_identity_shift():
    values = np.linspace(1.0, 5.0, 20)
    transformed, lmbda = boxcox_transform(values, lmbda=1.0)
    assert lmbda == 1.0
    np.testing.assert_allclose(transformed, values - 1.0)
    np.testing.assert_allclose(inverse_boxcox(transformed, 1.0), values)


def test_boxcox_forecast_roundtrips_naive_model():
    values = np.arange(1.0, 40.0)
    frame = _frame(values)

    def last_value_model(frame, target_col, steps):
        return np.full(steps, frame[target_col].iloc[-1])

    forecast = boxcox_forecast(frame, "value", steps=5, model_fn=last_value_model)
    assert forecast.shape == (5,)
    transformed, lmbda = boxcox_transform(values)
    expected = inverse_boxcox(np.full(5, transformed[-1]), lmbda)
    np.testing.assert_allclose(forecast, expected)


def test_boxcox_forecast_models_on_transformed_target():
    frame = _frame(np.arange(1.0, 30.0))
    seen = {}

    def capturing_model(frame, target_col, steps):
        seen["values"] = frame[target_col].to_numpy(dtype=float)
        return np.full(steps, frame[target_col].iloc[-1])

    forecast = boxcox_forecast(frame, "value", steps=3, model_fn=capturing_model)
    assert forecast.shape == (3,)
    assert np.all(np.isfinite(forecast))
    raw = frame["value"].to_numpy(dtype=float)
    assert not np.allclose(seen["values"], raw)
    assert np.all(seen["values"] >= 0.0)


def test_boxcox_forecast_matches_manual_pipeline():
    values = np.arange(1.0, 50.0)
    frame = _frame(values)
    transformed, lmbda = boxcox_transform(values)

    def naive(frame, target_col, steps):
        return np.full(steps, frame[target_col].iloc[-1])

    pipeline_forecast = boxcox_forecast(frame, "value", steps=4, model_fn=naive)
    manual = inverse_boxcox(np.full(4, transformed[-1]), lmbda)
    np.testing.assert_allclose(pipeline_forecast, manual)


def test_boxcox_forecast_accepts_explicit_lambda():
    frame = _frame(np.linspace(1.0, 10.0, 25))
    forecast = boxcox_forecast(
        frame,
        "value",
        steps=2,
        model_fn=lambda f, c, steps: np.full(steps, f[c].iloc[-1]),
        lmbda=0.5,
    )
    assert forecast.shape == (2,)
    assert np.all(np.isfinite(forecast))


def test_transform_rejects_non_positive_values():
    with pytest.raises(ValueError, match="strictly positive"):
        boxcox_transform([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="strictly positive"):
        boxcox_transform([-1.0, 2.0, 3.0])


def test_transform_rejects_constant_series():
    with pytest.raises(ValueError, match="constant"):
        boxcox_transform(np.ones(10))


def test_transform_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        boxcox_transform([1.0, np.nan, 3.0])


def test_transform_rejects_empty_input():
    with pytest.raises(ValueError, match="not be empty"):
        boxcox_transform([])


def test_inverse_rejects_out_of_domain_values():
    transformed = np.array([-2.0, -3.0])
    with pytest.raises(ValueError, match="outside its domain"):
        inverse_boxcox(transformed, 0.5)


def test_inverse_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        inverse_boxcox([1.0, np.inf], 0.5)


def test_forecast_validates_model_output_length():
    frame = _frame(np.arange(1.0, 25.0))
    with pytest.raises(ValueError, match="exactly steps predictions"):
        boxcox_forecast(
            frame,
            "value",
            steps=4,
            model_fn=lambda f, c, steps: np.array([1.0, 2.0]),
        )


def test_forecast_validates_unknown_target_column():
    frame = _frame(np.arange(1.0, 25.0))
    with pytest.raises(KeyError, match="unknown target column"):
        boxcox_forecast(
            frame,
            "missing",
            steps=2,
            model_fn=lambda f, c, steps: np.full(steps, 1.0),
        )


def test_forecast_validates_steps():
    frame = _frame(np.arange(1.0, 25.0))
    with pytest.raises(ValueError, match="steps must be at least 1"):
        boxcox_forecast(
            frame,
            "value",
            steps=0,
            model_fn=lambda f, c, steps: np.full(steps, 1.0),
        )
