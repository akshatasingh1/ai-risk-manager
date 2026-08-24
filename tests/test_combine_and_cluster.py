import networkx as nx

from src.graph import cluster_graph, combine_graphs


def test_combine_graphs_sums_overlapping_edge_weight():
    g1 = nx.Graph()
    g1.add_nodes_from([1, 2, 3])
    g1.add_edge(1, 2, weight=0.5)

    g2 = nx.Graph()
    g2.add_nodes_from([1, 2, 3])
    g2.add_edge(1, 2, weight=0.2)
    g2.add_edge(2, 3, weight=0.1)

    combined = combine_graphs(g1, g2)

    assert set(combined.nodes) == {1, 2, 3}
    assert combined[1][2]["weight"] == 0.7
    assert combined[2][3]["weight"] == 0.1


def test_cluster_graph_separates_two_dense_groups():
    graph = nx.Graph()
    # Two tightly connected triangles, no edge between them.
    graph.add_weighted_edges_from([
        (1, 2, 1.0), (2, 3, 1.0), (1, 3, 1.0),
        (4, 5, 1.0), (5, 6, 1.0), (4, 6, 1.0),
    ])

    partition = cluster_graph(graph, seed=42)

    assert partition[1] == partition[2] == partition[3]
    assert partition[4] == partition[5] == partition[6]
    assert partition[1] != partition[4]


def test_cluster_graph_isolated_nodes_get_singleton_clusters():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3])  # no edges at all

    partition = cluster_graph(graph, seed=42)

    assert len({partition[1], partition[2], partition[3]}) == 3
