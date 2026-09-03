import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.hybrid import (
    apply_hybrid_score,
    assign_cluster_categories,
    classifier_missed_hybrid_caught,
    cluster_category,
    compute_category_weights,
    second_look_mask,
    second_look_sensitivity_curve,
)


@pytest.mark.parametrize("size,has_identity,has_behavioral,expected", [
    (1, False, False, "isolated"),
    (1, True, True, "isolated"),  # size wins regardless of edge flags
    (3, True, True, "both"),
    (3, True, False, "identity_only"),
    (3, False, True, "behavioral_only"),
    (3, False, False, "neither"),
])
def test_cluster_category(size, has_identity, has_behavioral, expected):
    assert cluster_category(size, has_identity, has_behavioral) == expected


def test_assign_cluster_categories_maps_transactions_correctly():
    # cluster 100: {1, 2, 3} linked by an identity edge
    # cluster 200: {4, 5} linked by a behavioral edge
    # cluster 300: {6} isolated singleton
    partition = {1: 100, 2: 100, 3: 100, 4: 200, 5: 200, 6: 300}

    identity_graph = nx.Graph()
    identity_graph.add_edge(1, 2, weight=0.5)

    behavioral_graph = nx.Graph()
    behavioral_graph.add_edge(4, 5, weight=0.1)

    categories = assign_cluster_categories([1, 2, 3, 4, 5, 6], partition, identity_graph, behavioral_graph)

    assert categories[1] == "identity_only"
    assert categories[3] == "identity_only"  # in cluster 100 even without a direct edge to it
    assert categories[4] == "behavioral_only"
    assert categories[6] == "isolated"


def test_assign_cluster_categories_uses_global_cluster_size_not_subset_size():
    # Cluster 100 has 3 members total, but only 1 is in the queried subset --
    # it must still be classified using the true size of 3, not look isolated.
    partition = {1: 100, 2: 100, 3: 100}

    identity_graph = nx.Graph()
    identity_graph.add_edge(1, 2, weight=0.5)
    identity_graph.add_edge(2, 3, weight=0.5)

    behavioral_graph = nx.Graph()

    # Only transaction 3 is queried (as if it were the "holdout" subset).
    categories = assign_cluster_categories([3], partition, identity_graph, behavioral_graph)

    assert categories[3] == "identity_only"  # not "isolated"


def test_compute_category_weights_reflects_input_subset_only():
    categories = pd.Series(["behavioral_only", "behavioral_only", "identity_only", "identity_only"])
    is_fraud = pd.Series([1, 1, 0, 0])  # behavioral_only: 100% fraud, identity_only: 0% fraud
    # population fraud rate = 0.5

    weights = compute_category_weights(categories, is_fraud)

    assert weights["behavioral_only"] == pytest.approx(1.0 / 0.5)
    assert weights["identity_only"] == pytest.approx(0.0 / 0.5)


def test_compute_category_weights_forces_neutral_for_isolated_and_neither():
    # Even with fabricated data implying these categories are highly predictive,
    # isolated/neither must stay pinned at 1.0.
    categories = pd.Series(["isolated", "isolated", "neither", "neither"])
    is_fraud = pd.Series([1, 1, 1, 1])

    weights = compute_category_weights(categories, is_fraud)

    assert weights["isolated"] == 1.0
    assert weights["neither"] == 1.0


def test_compute_category_weights_falls_back_to_default_when_category_absent():
    categories = pd.Series(["behavioral_only", "behavioral_only"])
    is_fraud = pd.Series([1, 0])

    weights = compute_category_weights(categories, is_fraud, default_weight=1.0)

    assert weights["identity_only"] == 1.0
    assert weights["both"] == 1.0


def test_apply_hybrid_score_multiplies_and_is_unchanged_at_neutral_weight():
    proba = np.array([0.2, 0.5, 0.8])
    categories = pd.Series(["behavioral_only", "isolated", "identity_only"])
    weights = {"behavioral_only": 1.5, "isolated": 1.0, "identity_only": 0.5}

    score = apply_hybrid_score(proba, categories, weights)

    assert score[0] == pytest.approx(0.3)
    assert score[1] == pytest.approx(0.5)  # neutral weight -> unchanged
    assert score[2] == pytest.approx(0.4)


def test_classifier_missed_hybrid_caught_identifies_flips():
    y_true = np.array([1, 1, 1, 0])
    classifier_proba = np.array([0.5, 0.9, 0.5, 0.5])  # case 1 already caught by classifier
    hybrid_score = np.array([0.9, 0.9, 0.5, 0.9])
    classifier_threshold = 0.8
    hybrid_threshold = 0.8

    mask = classifier_missed_hybrid_caught(y_true, classifier_proba, hybrid_score, classifier_threshold, hybrid_threshold)

    assert mask[0]  # classifier missed (0.5 < 0.8), hybrid catches (0.9 >= 0.8) -> True
    assert not mask[1]  # classifier already caught it -> not a "miss", so False
    assert not mask[2]  # classifier missed, hybrid also misses -> False
    assert not mask[3]  # not actually fraud -> False regardless of scores


def test_second_look_mask_only_flags_missed_transactions_with_ring_evidence():
    classifier_proba = np.array([0.9, 0.5, 0.5, 0.2, 0.5])
    categories = pd.Series(["behavioral_only", "behavioral_only", "isolated", "behavioral_only", "identity_only"])

    mask = second_look_mask(classifier_proba, categories, lower_bound=0.3, classifier_threshold=0.83)

    assert not mask[0]  # already flagged by classifier (0.9 >= 0.83) -- not a "missed" case
    assert mask[1]      # missed (0.5 < 0.83), in-band (>= 0.3), trusted category -> True
    assert not mask[2]  # missed and in-band, but isolated (not a boost category) -> False
    assert not mask[3]  # missed, trusted category, but below the lower_bound (0.2 < 0.3) -> False
    assert not mask[4]  # missed, in-band, but identity_only is not a default boost category -> False


def test_second_look_sensitivity_curve_computes_expected_columns():
    y_true = np.array([1, 1, 0, 0])
    classifier_proba = np.array([0.5, 0.6, 0.5, 0.6])
    categories = pd.Series(["behavioral_only", "behavioral_only", "behavioral_only", "behavioral_only"])
    amounts = np.array([100.0, 200.0, 50.0, 50.0])

    curve = second_look_sensitivity_curve(
        y_true, classifier_proba, categories, amounts,
        classifier_threshold=0.83, lower_bounds=[0.4, 0.55], review_cost=10.0,
    )

    row_04 = curve[curve["lower_bound"] == 0.4].iloc[0]
    assert row_04["n_flagged"] == 4
    assert row_04["n_fraud_caught"] == 2
    assert row_04["precision"] == 0.5
    assert row_04["capture_rate_of_misses"] == 1.0  # both fraud cases are classifier misses here
    assert row_04["fraud_amount_recovered_rs"] == 300.0
    assert row_04["review_cost_rs"] == 40.0
    assert row_04["net_rs"] == 260.0

    row_055 = curve[curve["lower_bound"] == 0.55].iloc[0]
    assert row_055["n_flagged"] == 2  # only the two 0.6-scored transactions qualify
    assert row_055["n_fraud_caught"] == 1
