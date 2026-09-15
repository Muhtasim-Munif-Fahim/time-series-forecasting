import numpy as np
import pytest

from ts_forecast.evaluation import interval_score


def test_interval_score_equals_width_for_covered_observations():
    result = interval_score([1, 2], [0, 0], [2, 4], coverage=0.8)
    assert result["score"] == pytest.approx(3.0)
    assert np.allclose(result["scores"], [2.0, 4.0])


def test_interval_score_penalizes_missed_observations():
    covered = interval_score([1], [0], [2], coverage=0.8)["score"]
    missed = interval_score([4], [0], [2], coverage=0.8)["score"]
    assert missed > covered


@pytest.mark.parametrize("coverage", [0, 1, -0.1, 1.1])
def test_interval_score_validates_coverage(coverage):
    with pytest.raises(ValueError, match="coverage"):
        interval_score([1], [0], [2], coverage=coverage)
