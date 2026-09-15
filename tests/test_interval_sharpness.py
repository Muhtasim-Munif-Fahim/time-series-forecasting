import numpy as np
import pytest

from ts_forecast.evaluation import interval_sharpness


def test_reports_width_summary_and_training_normalization():
    result = interval_sharpness([1, 2, 4], [3, 5, 5], y_train=[0, 10, 5])
    assert result["count"] == 3
    assert result["mean_width"] == pytest.approx(2.0)
    assert result["median_width"] == pytest.approx(2.0)
    assert result["normalized_mean_width"] == pytest.approx(0.2)


@pytest.mark.parametrize("lower,upper", [([1], [0]), ([1], [1, 2]), ([], [])])
def test_rejects_invalid_intervals(lower, upper):
    with pytest.raises(ValueError):
        interval_sharpness(lower, upper)


def test_rejects_nonfinite_and_constant_training_data():
    with pytest.raises(ValueError):
        interval_sharpness([0], [np.inf])
    with pytest.raises(ValueError):
        interval_sharpness([0], [1], y_train=[2, 2])
