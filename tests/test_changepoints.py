"""Tests for CUSUM changepoint detection."""

import numpy as np
import pytest

from ts_forecast.diagnostics import detect_changepoints


def _three_regimes(seed=0, n=60):
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.normal(0.0, 1.0, n), rng.normal(6.0, 1.0, n), rng.normal(-3.0, 1.0, n)]
    )


def test_two_clear_shifts_are_located():
    result = detect_changepoints(_three_regimes())
    assert result["n_changepoints"] == 2
    assert result["changepoints"] == [60, 120]


def test_segment_means_track_the_generating_regimes():
    segments = detect_changepoints(_three_regimes())["segments"]
    means = [segment["mean"] for segment in segments]
    assert means[0] == pytest.approx(0.0, abs=0.5)
    assert means[1] == pytest.approx(6.0, abs=0.5)
    assert means[2] == pytest.approx(-3.0, abs=0.5)


def test_segments_tile_the_series_without_gaps_or_overlap():
    result = detect_changepoints(_three_regimes())
    segments = result["segments"]
    assert segments[0]["start"] == 0
    assert segments[-1]["end"] == 180
    for earlier, later in zip(segments[:-1], segments[1:]):
        assert earlier["end"] == later["start"]
    assert sum(segment["length"] for segment in segments) == 180


def test_a_single_shift_gives_one_break_and_two_segments():
    rng = np.random.default_rng(3)
    series = np.concatenate([rng.normal(0.0, 1.0, 80), rng.normal(8.0, 1.0, 80)])
    result = detect_changepoints(series)
    assert result["changepoints"] == [80]
    assert len(result["segments"]) == 2


def test_homogeneous_noise_yields_no_changepoints():
    rng = np.random.default_rng(11)
    result = detect_changepoints(rng.normal(0.0, 1.0, 200))
    assert result["n_changepoints"] == 0
    assert len(result["segments"]) == 1
    assert result["segments"][0]["length"] == 200


def test_constant_series_has_no_changepoints():
    # Zero spread must not divide through and must not report a break.
    result = detect_changepoints(np.full(100, 5.0))
    assert result["n_changepoints"] == 0


def test_a_higher_threshold_reports_fewer_breaks():
    series = _three_regimes()
    strict = detect_changepoints(series, threshold=5.0)
    assert strict["n_changepoints"] <= detect_changepoints(series)["n_changepoints"]
    assert strict["n_changepoints"] == 0


def test_max_breaks_caps_the_result():
    rng = np.random.default_rng(5)
    series = np.concatenate([rng.normal(level, 0.5, 40) for level in (0, 5, 10, 15, 20, 25)])
    result = detect_changepoints(series, max_breaks=2)
    assert result["n_changepoints"] == 2


def test_max_breaks_zero_disables_detection():
    assert detect_changepoints(_three_regimes(), max_breaks=0)["n_changepoints"] == 0


def test_min_segment_prevents_short_regimes():
    result = detect_changepoints(_three_regimes(), min_segment=40)
    for segment in result["segments"]:
        assert segment["length"] >= 40


def test_the_default_threshold_is_the_five_percent_critical_value():
    rng = np.random.default_rng(101)
    spurious = sum(
        detect_changepoints(rng.normal(0.0, 1.0, 180))["n_changepoints"] > 0
        for _ in range(100)
    )
    # Family-wise across recursive scans, so allow headroom over the nominal
    # 5%; 1.0 would land near 23% here.
    assert spurious <= 12


def test_detection_is_invariant_to_shifting_and_scaling():
    series = _three_regimes()
    base = detect_changepoints(series)["changepoints"]
    assert detect_changepoints(series * 10.0 + 100.0)["changepoints"] == base


def test_too_short_a_series_is_rejected():
    with pytest.raises(ValueError, match="at least 16 observations"):
        detect_changepoints(np.arange(10.0))


def test_non_finite_values_are_rejected():
    series = _three_regimes()
    series[5] = np.nan
    with pytest.raises(ValueError, match="finite numbers"):
        detect_changepoints(series)


@pytest.mark.parametrize("bad", [0, -1.0])
def test_threshold_must_be_positive(bad):
    with pytest.raises(ValueError, match="threshold must be strictly positive"):
        detect_changepoints(_three_regimes(), threshold=bad)


def test_min_segment_must_be_a_sensible_integer():
    with pytest.raises(ValueError, match="min_segment must be an integer"):
        detect_changepoints(_three_regimes(), min_segment=1)


def test_max_breaks_must_be_non_negative():
    with pytest.raises(ValueError, match="max_breaks must be a non-negative integer"):
        detect_changepoints(_three_regimes(), max_breaks=-1)


def test_result_reports_the_settings_used():
    result = detect_changepoints(_three_regimes(), threshold=2.0, min_segment=10)
    assert result["threshold"] == 2.0
    assert result["min_segment"] == 10
