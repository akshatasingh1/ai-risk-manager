import pandas as pd

from src.behavioral import BEHAVIORAL_FEATURES, build_behavioral_graph


def _make_row(txn_id, dt, amt, c_val, d_val):
    row = {"TransactionID": txn_id, "TransactionDT": dt, "TransactionAmt": amt}
    for c in [f"C{i}" for i in range(1, 15)]:
        row[c] = c_val
    for d in [f"D{i}" for i in range(1, 16)]:
        row[d] = d_val
    return row


def test_similar_transactions_in_same_bucket_get_an_edge():
    df = pd.DataFrame([
        _make_row(1, 0, 100.0, 5, 2),
        _make_row(2, 60, 101.0, 5, 2),   # nearly identical, 1 minute later
        _make_row(3, 30, 9000.0, 500, 800),  # very different pattern, same bucket
    ])
    graph = build_behavioral_graph(df, time_bucket=900, similarity_threshold=0.9)

    assert set(graph.nodes) == {1, 2, 3}
    assert graph.has_edge(1, 2)
    assert not graph.has_edge(1, 3)
    assert not graph.has_edge(2, 3)


def test_similar_transactions_in_different_buckets_get_no_edge():
    df = pd.DataFrame([
        _make_row(1, 0, 100.0, 5, 2),
        _make_row(2, 100000, 100.0, 5, 2),  # identical pattern, but far apart in time
    ])
    graph = build_behavioral_graph(df, time_bucket=900, similarity_threshold=0.9)
    assert not graph.has_edge(1, 2)


def test_edge_weight_is_scaled_below_raw_similarity():
    # A third, very different row gives StandardScaler real variance to work
    # with -- with only two identical rows, both scale to all-zero vectors
    # and cosine similarity is undefined, not 1.0.
    df = pd.DataFrame([
        _make_row(1, 0, 100.0, 5, 2),
        _make_row(2, 10, 100.0, 5, 2),  # nearly identical to row 1
        _make_row(3, 20, 9000.0, 500, 800),  # very different, gives the scaler variance
    ])
    graph = build_behavioral_graph(df, time_bucket=900, similarity_threshold=0.5, weight_multiplier=0.2)
    assert graph.has_edge(1, 2)
    weight = graph[1][2]["weight"]
    assert 0 < weight <= 0.2 + 1e-9
