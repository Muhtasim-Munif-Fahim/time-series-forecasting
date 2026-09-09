"""Tests for least-squares (MinT family) hierarchical reconciliation."""

import numpy as np
import pytest

from ts_forecast.models import reconcile_optimal

STRUCTURE = {"total": ["a", "b"]}
INCOHERENT = {"total": [100.0], "a": [40.0], "b": [50.0]}


def _coherence_error(reconciled, structure):
    worst = 0.0
    for parent, children in structure.items():
        implied = np.sum([reconciled[child] for child in children], axis=0)
        worst = max(worst, float(np.max(np.abs(reconciled[parent] - implied))))
    return worst


@pytest.mark.parametrize("method", ["ols", "wls"])
def test_result_is_coherent(method):
    reconciled = reconcile_optimal(INCOHERENT, STRUCTURE, method=method)
    assert _coherence_error(reconciled, STRUCTURE) == pytest.approx(0.0, abs=1e-9)


def test_a_deeper_hierarchy_is_coherent_at_every_level():
    structure = {"total": ["north", "south"], "north": ["n1", "n2"], "south": ["s1", "s2"]}
    forecasts = {
        "total": [100.0, 110.0],
        "north": [45.0, 50.0],
        "south": [50.0, 55.0],
        "n1": [20.0, 22.0],
        "n2": [24.0, 27.0],
        "s1": [25.0, 28.0],
        "s2": [24.0, 26.0],
    }
    reconciled = reconcile_optimal(forecasts, structure)
    assert _coherence_error(reconciled, structure) == pytest.approx(0.0, abs=1e-9)


def test_already_coherent_forecasts_are_left_alone():
    coherent = {"total": [90.0], "a": [40.0], "b": [50.0]}
    reconciled = reconcile_optimal(coherent, STRUCTURE)
    for node, values in coherent.items():
        assert reconciled[node] == pytest.approx(values)


def test_the_adjustment_lands_between_the_base_forecasts():
    # 40 + 50 = 90 against a stated total of 100, so the reconciled total
    # must sit inside that gap rather than adopting either extreme.
    reconciled = reconcile_optimal(INCOHERENT, STRUCTURE)
    assert 90.0 < float(reconciled["total"][0]) < 100.0


def test_ols_and_wls_disagree_on_how_to_split_the_gap():
    ols = reconcile_optimal(INCOHERENT, STRUCTURE, method="ols")["total"][0]
    wls = reconcile_optimal(INCOHERENT, STRUCTURE, method="wls")["total"][0]
    assert float(ols) != pytest.approx(float(wls))


def test_mint_trusts_the_node_with_the_smaller_residual_variance():
    rng = np.random.default_rng(0)
    noisy_total = {
        "total": rng.normal(0.0, 5.0, 60),
        "a": rng.normal(0.0, 0.2, 60),
        "b": rng.normal(0.0, 0.2, 60),
    }
    reconciled = reconcile_optimal(
        INCOHERENT, STRUCTURE, method="mint_diagonal", residuals=noisy_total
    )
    # Precise leaves and a noisy total should pull the answer toward 90.
    assert float(reconciled["total"][0]) < 92.0

    noisy_leaves = {
        "total": rng.normal(0.0, 0.2, 60),
        "a": rng.normal(0.0, 5.0, 60),
        "b": rng.normal(0.0, 5.0, 60),
    }
    reconciled = reconcile_optimal(
        INCOHERENT, STRUCTURE, method="mint_diagonal", residuals=noisy_leaves
    )
    # A precise total should pull the answer toward 100 instead.
    assert float(reconciled["total"][0]) > 98.0


def test_mint_is_coherent_too():
    residuals = {name: np.linspace(-1.0, 1.0, 40) * scale for name, scale in
                 (("total", 3.0), ("a", 1.0), ("b", 2.0))}
    reconciled = reconcile_optimal(
        INCOHERENT, STRUCTURE, method="mint_diagonal", residuals=residuals
    )
    assert _coherence_error(reconciled, STRUCTURE) == pytest.approx(0.0, abs=1e-9)


def test_a_multi_step_horizon_is_reconciled_per_step():
    forecasts = {"total": [100.0, 200.0], "a": [40.0, 90.0], "b": [50.0, 100.0]}
    reconciled = reconcile_optimal(forecasts, STRUCTURE)
    assert reconciled["total"].shape == (2,)
    assert _coherence_error(reconciled, STRUCTURE) == pytest.approx(0.0, abs=1e-9)


def test_an_aggregate_without_a_base_forecast_is_derived():
    forecasts = {"a": [40.0], "b": [50.0]}
    reconciled = reconcile_optimal(forecasts, STRUCTURE)
    assert float(reconciled["total"][0]) == pytest.approx(90.0)
    assert float(reconciled["a"][0]) == pytest.approx(40.0)


def test_forecasts_must_share_a_horizon():
    with pytest.raises(ValueError, match="same horizon"):
        reconcile_optimal({"total": [1.0, 2.0], "a": [1.0], "b": [1.0]}, STRUCTURE)


def test_non_finite_forecasts_are_rejected():
    with pytest.raises(ValueError, match="finite numbers"):
        reconcile_optimal({"total": [np.nan], "a": [1.0], "b": [1.0]}, STRUCTURE)


def test_an_unknown_child_is_rejected():
    with pytest.raises(KeyError, match="unknown node in hierarchy"):
        reconcile_optimal({"total": [1.0], "a": [1.0]}, {"total": ["a", "ghost"]})


def test_a_cycle_reaching_a_leaf_is_rejected_as_a_cycle():
    with pytest.raises(ValueError, match="must not contain cycles"):
        reconcile_optimal(
            {"a": [1.0], "b": [1.0], "leaf": [1.0]},
            {"a": ["b", "leaf"], "b": ["a"]},
        )


def test_a_hierarchy_that_is_all_cycle_has_no_leaves_to_solve_for():
    with pytest.raises(ValueError, match="at least one leaf node"):
        reconcile_optimal(
            {"a": [1.0], "b": [1.0]}, {"a": ["b"], "b": ["a"]}
        )


def test_an_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method must be one of"):
        reconcile_optimal(INCOHERENT, STRUCTURE, method="magic")


def test_mint_without_residuals_is_rejected():
    with pytest.raises(ValueError, match="requires residuals"):
        reconcile_optimal(INCOHERENT, STRUCTURE, method="mint_diagonal")


def test_mint_with_a_missing_node_is_rejected():
    with pytest.raises(KeyError, match="missing residuals for node"):
        reconcile_optimal(
            INCOHERENT, STRUCTURE, method="mint_diagonal",
            residuals={"total": [1.0, 2.0], "a": [1.0, 2.0]},
        )


def test_mint_needs_more_than_one_residual():
    with pytest.raises(ValueError, match="at least two residuals"):
        reconcile_optimal(
            INCOHERENT, STRUCTURE, method="mint_diagonal",
            residuals={"total": [1.0], "a": [1.0], "b": [1.0]},
        )


def test_a_zero_variance_node_does_not_divide_through():
    residuals = {"total": [1.0, -1.0, 2.0], "a": [0.0, 0.0, 0.0], "b": [1.0, -1.0, 0.5]}
    reconciled = reconcile_optimal(
        INCOHERENT, STRUCTURE, method="mint_diagonal", residuals=residuals
    )
    assert np.all(np.isfinite(reconciled["total"]))


def test_empty_inputs_are_rejected():
    with pytest.raises(ValueError, match="at least one node"):
        reconcile_optimal({}, STRUCTURE)
    with pytest.raises(ValueError, match="structure must map aggregates"):
        reconcile_optimal(INCOHERENT, {})
