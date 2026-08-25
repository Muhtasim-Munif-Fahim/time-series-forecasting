"""Tests for quantile crossing repair of multi-quantile forecasts."""

import numpy as np
import pytest

from ts_forecast.models import repair_quantile_crossing


def test_monotone_forecasts_pass_through():
    forecasts = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    repaired, crossings = repair_quantile_crossing(forecasts)
    np.testing.assert_allclose(repaired, forecasts)
    assert crossings == 0


def test_crossed_horizons_are_sorted_and_counted():
    forecasts = np.array(
        [
            [10.0, 8.0, 12.0],
            [5.0, 5.0, 5.0],
            [7.0, 3.0, 2.0],
        ]
    )
    repaired, crossings = repair_quantile_crossing(forecasts)
    np.testing.assert_allclose(
        repaired,
        np.array(
            [
                [8.0, 10.0, 12.0],
                [5.0, 5.0, 5.0],
                [2.0, 3.0, 7.0],
            ]
        ),
    )
    assert crossings == 3


def test_dict_input_sorts_quantile_levels_first():
    forecasts = {
        0.9: np.array([1.0, 0.0]),
        0.1: np.array([0.0, 2.0]),
        0.5: np.array([0.5, 1.0]),
    }
    repaired, crossings = repair_quantile_crossing(forecasts)
    np.testing.assert_allclose(
        repaired,
        np.array(
            [
                [0.0, 0.5, 1.0],
                [0.0, 1.0, 2.0],
            ]
        ),
    )
    assert crossings == 2


def test_repair_preserves_each_horizons_values():
    rng = np.random.default_rng(42)
    forecasts = rng.normal(size=(20, 5))
    repaired, _ = repair_quantile_crossing(forecasts)
    for row_original, row_repaired in zip(forecasts, repaired):
        assert sorted(row_original) == pytest.approx(list(row_repaired))


def test_validates_inputs():
    good = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError, match="at least one quantile"):
        repair_quantile_crossing({})

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        repair_quantile_crossing({0.5: [1.0], 1.5: [2.0]})

    with pytest.raises(ValueError, match="at least two quantile forecasts"):
        repair_quantile_crossing({0.5: [1.0, 2.0]})

    with pytest.raises(ValueError, match="at least two quantile forecasts"):
        repair_quantile_crossing(np.array([[1.0]]))

    with pytest.raises(ValueError, match="must be 2-D"):
        repair_quantile_crossing(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="only finite values"):
        crossed = np.array([[1.0, np.nan]])
        repair_quantile_crossing(crossed)