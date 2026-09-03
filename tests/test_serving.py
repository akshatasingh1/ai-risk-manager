from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.precompute import build_cluster_edges, build_cluster_summary, build_transaction_graph_features
from src.serving import _expected_columns_by_kind, add_graph_features, score_batch


class _FakeContext:
    """Stand-in for ScoringContext -- exercises add_graph_features/
    get_cluster_alerts/get_cluster_graph without needing the real (large,
    gitignored) model/graph artifacts on disk. Built via the same
    src.precompute functions the real ScoringContext's artifacts are
    generated with, so this fixture matches the real attribute types
    (pandas Series, not dicts) and structure, not just a hand-rolled lookalike.
    """

    def __init__(self, partition, identity_graph, behavioral_graph, combined_graph, model=None, amount_threshold_new_card=1e9):
        graph_features = build_transaction_graph_features(partition, combined_graph).set_index("TransactionID")
        self.partition = graph_features["cluster_id"]
        self.degree = graph_features["graph_degree"]
        self.weighted_degree = graph_features["graph_weighted_degree"]
        self.pagerank = graph_features["graph_pagerank"]
        self.min_pagerank = self.pagerank.min() if len(self.pagerank) else 0.0

        cluster_summary = build_cluster_summary(partition, identity_graph, behavioral_graph).set_index("cluster_id")
        self.cluster_sizes = cluster_summary["size"]
        self.has_identity = cluster_summary["has_identity"].to_dict()
        self.has_behavioral = cluster_summary["has_behavioral"].to_dict()

        self.cluster_edges = build_cluster_edges(partition, identity_graph, behavioral_graph)
        self.category_weights: dict = {}
        self._cluster_members: dict | None = None

        self.model = model
        self.amount_threshold_new_card = amount_threshold_new_card
        if model is not None:
            self.expected_columns_by_kind = _expected_columns_by_kind(model)
            self.expected_columns = [c for cols in self.expected_columns_by_kind.values() for c in cols]

    def cluster_members(self) -> dict:
        if self._cluster_members is None:
            members = defaultdict(list)
            for node, cid in self.partition.items():
                members[cid].append(node)
            self._cluster_members = members
        return self._cluster_members


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


def test_get_cluster_alerts_ranks_and_filters_by_category(ctx):
    from src.serving import get_cluster_alerts

    ctx.category_weights = {"identity_only": 0.58, "behavioral_only": 1.51}

    alerts = get_cluster_alerts(ctx, min_size=2, categories=("identity_only", "behavioral_only"), limit=10)

    assert {a["cluster_id"] for a in alerts} == {100, 200}  # cluster 300 is a singleton, excluded
    # behavioral_only (weight 1.51) must outrank identity_only (weight 0.58)
    assert alerts[0]["cluster_id"] == 200
    assert alerts[0]["category"] == "behavioral_only"
    assert alerts[1]["cluster_id"] == 100
    assert alerts[1]["category"] == "identity_only"


def test_get_cluster_alerts_respects_single_category_filter(ctx):
    ctx.category_weights = {"identity_only": 0.58, "behavioral_only": 1.51}
    from src.serving import get_cluster_alerts

    alerts = get_cluster_alerts(ctx, min_size=2, categories=("identity_only",), limit=10)

    assert [a["cluster_id"] for a in alerts] == [100]


def test_get_cluster_graph_returns_only_edges_between_shown_members(ctx):
    from src.serving import get_cluster_graph

    result = get_cluster_graph(ctx, cluster_id=100, max_nodes=30)

    assert result["cluster_id"] == 100
    assert result["total_size"] == 3
    assert set(result["nodes"]) == {1, 2, 3}
    assert len(result["edges"]) == 1
    edge = result["edges"][0]
    assert {edge["source"], edge["target"]} == {1, 2}
    assert edge["type"] == "identity"


def test_get_cluster_graph_caps_shown_nodes_and_excludes_edges_outside_the_cap(ctx):
    from src.serving import get_cluster_graph

    # cluster 100 has 3 members {1,2,3} but only the edge (1,2) exists; capping
    # to 1 node should show a single node and zero edges, not error.
    result = get_cluster_graph(ctx, cluster_id=100, max_nodes=1)

    assert result["total_size"] == 3  # true size preserved even though display is capped
    assert len(result["nodes"]) == 1
    assert result["edges"] == []


def test_get_cluster_graph_unknown_cluster_returns_empty():
    from src.serving import get_cluster_graph

    partition = {1: 100, 2: 100}
    identity_graph = nx.Graph()
    identity_graph.add_edge(1, 2, weight=0.5)
    behavioral_graph = nx.Graph()
    combined_graph = nx.Graph()
    combined_graph.add_edge(1, 2, weight=0.5)
    ctx = _FakeContext(partition, identity_graph, behavioral_graph, combined_graph)

    result = get_cluster_graph(ctx, cluster_id=999)
    assert result["nodes"] == []
    assert result["edges"] == []
