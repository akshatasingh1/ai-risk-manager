import networkx as nx

from src.ring_eval import classify_cluster_edge_types, cluster_stats, evaluate_flagged_clusters


def test_cluster_stats_computes_size_and_fraud_rate():
    stats = cluster_stats(
        transaction_ids=[1, 2, 3, 4, 5],
        cluster_ids=[1, 1, 1, 2, 2],
        is_fraud=[1, 0, 0, 1, 1],
    )
    assert stats.loc[1, "size"] == 3
    assert stats.loc[1, "fraud_count"] == 1
    assert stats.loc[1, "fraud_rate"] == 1 / 3
    assert stats.loc[2, "size"] == 2
    assert stats.loc[2, "fraud_rate"] == 1.0


def test_evaluate_flagged_clusters_at_min_size_threshold():
    stats = cluster_stats(
        transaction_ids=[1, 2, 3, 4, 5],
        cluster_ids=[1, 1, 1, 2, 2],
        is_fraud=[1, 0, 0, 1, 1],
    )
    # total dataset fraud = 3, population fraud rate = 3/5 = 0.6
    result = evaluate_flagged_clusters(stats, min_size=3, total_fraud=3, population_fraud_rate=0.6)

    assert result["n_clusters_flagged"] == 1  # only cluster 1 (size 3) qualifies
    assert result["n_transactions_flagged"] == 3
    assert result["n_fraud_captured"] == 1
    assert result["precision"] == 1 / 3
    assert abs(result["lift"] - (1 / 3) / 0.6) < 1e-9
    assert abs(result["capture_rate"] - 1 / 3) < 1e-9


def test_evaluate_flagged_clusters_lower_threshold_includes_both_clusters():
    stats = cluster_stats(
        transaction_ids=[1, 2, 3, 4, 5],
        cluster_ids=[1, 1, 1, 2, 2],
        is_fraud=[1, 0, 0, 1, 1],
    )
    result = evaluate_flagged_clusters(stats, min_size=2, total_fraud=3, population_fraud_rate=0.6)

    assert result["n_clusters_flagged"] == 2
    assert result["n_transactions_flagged"] == 5
    assert result["n_fraud_captured"] == 3
    assert result["capture_rate"] == 1.0  # captures all fraud in the dataset


def test_classify_cluster_edge_types():
    partition = {1: 100, 2: 100, 3: 100, 4: 200, 5: 200}

    identity_graph = nx.Graph()
    identity_graph.add_edge(1, 2, weight=0.5)  # both in cluster 100 -> has_identity

    behavioral_graph = nx.Graph()
    behavioral_graph.add_edge(4, 5, weight=0.1)  # both in cluster 200 -> has_behavioral

    result = classify_cluster_edge_types(partition, identity_graph, behavioral_graph)

    assert result["has_identity"].get(100) is True
    assert result["has_identity"].get(200) is None  # cluster 200 has no identity edge
    assert result["has_behavioral"].get(200) is True
    assert result["has_behavioral"].get(100) is None  # cluster 100 has no behavioral edge
