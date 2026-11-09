import numpy as np
import pandas as pd
import pytest

from ts_forecast.mcs import (
    forecast_losses,
    model_confidence_set,
    moving_block_bootstrap_indices,
)


def _losses(seed=0, n=300):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n) ** 2
    return pd.DataFrame(
        {
            "good_a": base + rng.normal(scale=0.05, size=n) ** 2,
            "good_b": base + rng.normal(scale=0.05, size=n) ** 2,
            "bad": base + 1.5 + rng.normal(scale=0.3, size=n),
            "awful": base + 4.0 + rng.normal(scale=0.3, size=n),
        }
    )


@pytest.mark.parametrize("stat", ["max", "range"])
def test_mcs_drops_clearly_inferior_models(stat):
    out = model_confidence_set(_losses(), alpha=0.1, statistic=stat, n_boot=500, seed=1)
    assert set(out["included"]) == {"good_a", "good_b"}
    assert out["eliminated"][0] == "awful"
    assert set(out["eliminated"]) == {"awful", "bad"}
    assert out["pvalues"]["awful"] < 0.1
    assert all(out["pvalues"][m] >= 0.1 for m in out["included"])


def test_pvalues_monotone_in_elimination_order():
    out = model_confidence_set(_losses(3), alpha=0.5, n_boot=300, seed=2)
    seq = out["table"]["mcs_pvalue"].to_numpy()
    assert np.all(np.diff(seq) >= -1e-12)
    assert out["table"]["in_mcs"].iloc[-1]


def test_equal_models_all_retained():
    rng = np.random.default_rng(5)
    losses = rng.normal(size=(400, 3)) ** 2
    out = model_confidence_set(losses, alpha=0.05, n_boot=400, seed=0)
    assert len(out["included"]) >= 2
    assert out["included"][0].startswith("model_")


def test_dict_input_and_names_override():
    d = {"x": [1.0, 2.0, 1.5, 1.2], "y": [5.0, 6.0, 5.5, 5.2]}
    out = model_confidence_set(d, n_boot=200, seed=0, names=["p", "q"])
    assert set(out["pvalues"]) == {"p", "q"}
    assert out["included"] == ["p"]
    assert out["pvalues"]["p"] == 1.0


def test_reproducible_with_seed():
    a = model_confidence_set(_losses(7), n_boot=200, seed=11)
    b = model_confidence_set(_losses(7), n_boot=200, seed=11)
    assert a["pvalues"] == b["pvalues"]


def test_block_bootstrap_indices_shape_and_contiguity():
    idx = moving_block_bootstrap_indices(10, 3, 4, seed=0)
    assert idx.shape == (4, 10)
    assert idx.min() >= 0 and idx.max() < 10
    # within a block consecutive indices step by one (mod n)
    assert np.all((idx[:, 1] - idx[:, 0]) % 10 == 1)


def test_forecast_losses():
    y = np.array([1.0, 2.0, 3.0])
    out = forecast_losses(y, {"a": [1.0, 2.0, 4.0], "b": [0.0, 2.0, 3.0]}, loss="absolute")
    assert list(out.columns) == ["a", "b"]
    assert out["a"].tolist() == [0.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        forecast_losses(y, {"a": [1.0]})


@pytest.mark.parametrize(
    "kwargs",
    [{"alpha": 0.0}, {"statistic": "sum"}, {"block_length": 0}, {"n_boot": 0}],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        model_confidence_set(_losses(), **kwargs)


def test_requires_two_models():
    with pytest.raises(ValueError):
        model_confidence_set(np.ones((10, 1)))
