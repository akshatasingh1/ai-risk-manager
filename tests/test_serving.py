import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.serving import _expected_columns_by_kind, add_graph_features, score_batch


class _FakeContext:
    """Stand-in for ScoringContext -- exercises add_graph_features without
    needing the real (large, gitignored) model/graph artifacts on disk.
    """

    def __init__(self, partition, identity_graph, behavioral_graph, combined_graph, model=None, amount_threshold_new_card=1e9):
        from src.ring_eval import classify_cluster_edge_types

        self.partition = partition
        self.identity_graph = identity_graph
        self.behavioral_graph = behavioral_graph
        self.combined_graph = combined_graph

        self.degree = dict(combined_graph.degree())
        self.weighted_degree = dict(combined_graph.degree(weight="weight"))
        self.pagerank = nx.pagerank(combined_graph, weight="weight") if combined_graph.number_of_nodes() else {}
        self.min_pagerank = min(self.pagerank.values()) if self.pagerank else 0.0
        self.cluster_sizes = pd.Series(partition).value_counts()

        edge_types = classify_cluster_edge_types(partition, identity_graph, behavioral_graph)
        self.has_identity = edge_types["has_identity"]
        self.has_behavioral = edge_types["has_behavioral"]

        self.model = model
        self.amount_threshold_new_card = amount_threshold_new_card
        if model is not None:
            self.expected_columns_by_kind = _expected_columns_by_kind(model)
            self.expected_columns = [c for cols in self.expected_columns_by_kind.values() for c in cols]


@pytest.fixture
def ctx():
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

    return _FakeContext(partition, identity_graph, behavioral_graph, combined_graph)


def test_add_graph_features_assigns_category_size_and_degree(ctx):
    df = pd.DataFrame({"TransactionID": [1, 4, 6]})

    out = add_graph_features(df, ctx)

    row1 = out[out["TransactionID"] == 1].iloc[0]
    assert row1["ring_category"] == "identity_only"
    assert row1["ring_cluster_size"] == 3
    assert row1["graph_degree"] == 1

    row4 = out[out["TransactionID"] == 4].iloc[0]
    assert row4["ring_category"] == "behavioral_only"
    assert row4["ring_cluster_size"] == 2

    row6 = out[out["TransactionID"] == 6].iloc[0]
    assert row6["ring_category"] == "isolated"
    assert row6["ring_cluster_size"] == 1
    assert row6["graph_degree"] == 0
    assert row6["graph_weighted_degree"] == 0.0


def test_add_graph_features_unseen_transaction_falls_back_to_isolated(ctx):
    # TransactionID 999 was never part of the training-time graph at all.
    df = pd.DataFrame({"TransactionID": [999]})

    out = add_graph_features(df, ctx)

    row = out.iloc[0]
    assert row["ring_category"] == "isolated"
    assert row["ring_cluster_size"] == 1
    assert row["graph_degree"] == 0
    assert row["graph_weighted_degree"] == 0.0
    assert row["graph_pagerank"] == ctx.min_pagerank


def test_score_batch_matches_regardless_of_single_row_json_dtype_inference(ctx):
    """Regression test for a real bug: a DataFrame built from a single JSON
    request (e.g. from a `/score` API call) infers `object` dtype -- holding
    Python `None` -- for any feature column whose only value is missing,
    instead of the float64 NaN / proper missing-string a multi-row batch
    would get. That used to silently change the classifier's score for the
    exact same logical transaction depending on whether it arrived alone or
    as part of a batch. score_batch must produce the same score either way.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from src.features import build_preprocessor

    train = pd.DataFrame({
        "amt_feat": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan],
        "cat_feat": ["a", "b", None, "a", "b", "a"],
    })
    y_train = [0, 1, 0, 1, 0, 1]
    model = Pipeline([
        ("prep", build_preprocessor(["amt_feat"], ["cat_feat"])),
        ("clf", LogisticRegression()),
    ])
    model.fit(train, y_train)

    ctx.model = model
    from src.serving import _expected_columns_by_kind
    ctx.expected_columns_by_kind = _expected_columns_by_kind(model)
    ctx.expected_columns = [c for cols in ctx.expected_columns_by_kind.values() for c in cols]
    ctx.amount_threshold_new_card = 1e9

    # A row where both feature values are missing -- constructed two ways:
    # (a) as a proper multi-row-inferred float64/object-string batch (what a
    #     parquet-backed replay produces), (b) as a lone single-row frame
    #     with Python `None` (what json.loads -> pydantic -> DataFrame([...])
    #     produces for a `/score` call). Same logical transaction, same card.
    base = pd.DataFrame({
        "TransactionID": [1],
        "TransactionDT": [1000],
        "TransactionAmt": [10.0],
        "card1": [111],
        "card2": [222],
    })
    batch_style = pd.concat([base, pd.DataFrame({"amt_feat": [np.nan], "cat_feat": pd.array([None], dtype="object")})], axis=1)
    single_row_style = pd.DataFrame([{**base.iloc[0].to_dict(), "amt_feat": None, "cat_feat": None}])
    assert single_row_style["amt_feat"].dtype == object  # this is the dtype pandas actually infers here

    batch_result = score_batch(batch_style, ctx)
    single_result = score_batch(single_row_style, ctx)

    assert batch_result.iloc[0]["classifier_score"] == pytest.approx(single_result.iloc[0]["classifier_score"])
