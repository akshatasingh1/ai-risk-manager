import networkx as nx
import pytest

from src.precompute import build_cluster_edges, build_cluster_summary, build_transaction_graph_features


def _sample_graphs():
    # cluster 100: {1, 2, 3} linked by an identity edge -> identity_only
    # cluster 200: {4, 5} linked by a behavioral edge -> behavioral_only
    # cluster 300: {6} isolated singleton
    partition = {1: 100, 2: 100, 3: 100, 4: 200, 5: 200, 6: 300}

    identity_graph = nx.Graph()
    identity_graph.add_nodes_from(partition)
    identity_graph.add_edge(1, 2, weight=0.5)

    behavioral_graph = nx.Graph()
    behavioral_graph.add_nodes_from(partition)
    behavioral_graph.add_edge(4, 5, weight=0.2)

    combined_graph = nx.Graph()
    combined_graph.add_nodes_from(partition)
    combined_graph.add_edge(1, 2, weight=0.5)
    combined_graph.add_edge(4, 5, weight=0.2)

    return partition, identity_graph, behavioral_graph, combined_graph


def test_build_transaction_graph_features_covers_every_node():
    partition, _, _, combined_graph = _sample_graphs()
    df = build_transaction_graph_features(partition, combined_graph)

    assert set(df["TransactionID"]) == {1, 2, 3, 4, 5, 6}
    row1 = df[df["TransactionID"] == 1].iloc[0]
    assert row1["cluster_id"] == 100
    assert row1["graph_degree"] == 1
    assert row1["graph_weighted_degree"] == 0.5

    row6 = df[df["TransactionID"] == 6].iloc[0]
    assert row6["cluster_id"] == 300
    assert row6["graph_degree"] == 0
    assert row6["graph_weighted_degree"] == 0.0


def test_build_cluster_summary_omits_singletons():
    partition, identity_graph, behavioral_graph, _ = _sample_graphs()
    df = build_cluster_summary(partition, identity_graph, behavioral_graph, min_size=2)

    assert set(df["cluster_id"]) == {100, 200}  # cluster 300 (singleton) omitted
    row100 = df[df["cluster_id"] == 100].iloc[0]
    assert row100["size"] == 3
    assert row100["has_identity"] and not row100["has_behavioral"]

    row200 = df[df["cluster_id"] == 200].iloc[0]
    assert row200["size"] == 2
    assert row200["has_behavioral"] and not row200["has_identity"]


def test_build_cluster_edges_only_intra_cluster_and_typed_correctly():
    partition, identity_graph, behavioral_graph, _ = _sample_graphs()
    df = build_cluster_edges(partition, identity_graph, behavioral_graph, min_size=2)

    assert len(df) == 2  # one edge in cluster 100, one in cluster 200
    edge_100 = df[df["cluster_id"] == 100].iloc[0]
    assert {edge_100["source"], edge_100["target"]} == {1, 2}
    assert edge_100["edge_type"] == "identity"
    assert edge_100["weight"] == 0.5

    edge_200 = df[df["cluster_id"] == 200].iloc[0]
    assert {edge_200["source"], edge_200["target"]} == {4, 5}
    assert edge_200["edge_type"] == "behavioral"


def test_build_cluster_edges_marks_both_when_identity_and_behavioral_overlap():
    partition = {1: 100, 2: 100}
    identity_graph = nx.Graph()
    identity_graph.add_edge(1, 2, weight=0.3)
    behavioral_graph = nx.Graph()
    behavioral_graph.add_edge(1, 2, weight=0.4)

    df = build_cluster_edges(partition, identity_graph, behavioral_graph, min_size=2)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["edge_type"] == "both"
    assert row["weight"] == pytest.approx(0.7)
