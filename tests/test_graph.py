import pandas as pd

from src.graph import _edges_for_attribute, build_identity_graph, load_graph, save_graph


def test_edges_only_within_time_window():
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionDT": [0, 3600, 100000],  # 1&2 are 1h apart; 3 is ~27.8h after both
        "card1": [100, 100, 100],
    })
    edges = list(_edges_for_attribute(df, "card1", time_window=86400))
    pairs = {(a, b) for a, b, _ in edges}

    assert (1, 2) in pairs
    assert (1, 3) not in pairs
    assert (2, 3) not in pairs


def test_edge_weight_is_inverse_of_group_size():
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionDT": [0, 10, 20],
        "card1": [100, 100, 100],
    })
    edges = list(_edges_for_attribute(df, "card1", time_window=86400))
    assert all(w == 1 / 3 for _, _, w in edges)


def test_group_larger_than_max_size_is_excluded():
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [0, 10, 20, 30],
        "card1": [999, 999, 999, 999],
    })
    edges = list(_edges_for_attribute(df, "card1", time_window=86400, max_group_size=3))
    assert edges == []


def test_singleton_group_produces_no_edges():
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [0, 10],
        "card1": [100, 200],  # no shared value
    })
    edges = list(_edges_for_attribute(df, "card1", time_window=86400))
    assert edges == []


def test_build_identity_graph_sums_weight_across_shared_attributes():
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionDT": [0, 100, 999999],
        "card1": [100, 100, 200],
        "card2": [50, 50, 999],
    })
    graph = build_identity_graph(df, attributes=["card1", "card2"], time_window=86400)

    assert set(graph.nodes) == {1, 2, 3}
    assert graph.has_edge(1, 2)
    # weight = 1/2 (card1 group of 2) + 1/2 (card2 group of 2) = 1.0
    assert graph[1][2]["weight"] == 1.0
    assert not graph.has_edge(1, 3)
    assert not graph.has_edge(2, 3)


def test_build_identity_graph_ignores_missing_attribute_values():
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [0, 10],
        "card1": [None, None],
    })
    graph = build_identity_graph(df, attributes=["card1"], time_window=86400)
    assert set(graph.nodes) == {1, 2}
    assert graph.number_of_edges() == 0


def test_save_and_load_graph_roundtrip(tmp_path):
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [0, 10],
        "card1": [100, 100],
    })
    graph = build_identity_graph(df, attributes=["card1"], time_window=86400)

    path = tmp_path / "graph.pkl"
    save_graph(graph, path)
    loaded = load_graph(path)

    assert set(loaded.nodes) == set(graph.nodes)
    assert set(loaded.edges) == set(graph.edges)
