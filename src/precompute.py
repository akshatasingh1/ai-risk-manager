"""Precomputed, serving-friendly graph artifacts -- built once, offline, from
the full graphs, so the deployed API never needs to hold `identity_graph`,
`behavioral_graph`, or `combined_graph` in memory at all.

Everything the serving layer reads from the graphs is static per transaction
-- degree, weighted degree, PageRank, cluster membership, which clusters
carry identity/behavioral edges -- none of it changes per request. Measured
cost of holding the live networkx graphs just to look these values up:
~1.39GB in RAM for graphs that are 154MB on disk (see docs/case-queue-and-api.md).
These functions turn that into a handful of small, flat tables instead.
"""

import networkx as nx
import pandas as pd

from src.ring_eval import classify_cluster_edge_types


def build_transaction_graph_features(partition: dict, combined_graph: nx.Graph) -> pd.DataFrame:
    """One row per transaction: cluster_id, graph_degree, graph_weighted_degree, graph_pagerank."""
    degree = dict(combined_graph.degree())
    weighted_degree = dict(combined_graph.degree(weight="weight"))
    pagerank = nx.pagerank(combined_graph, weight="weight")

    transaction_ids = list(combined_graph.nodes())
    return pd.DataFrame({
        "TransactionID": transaction_ids,
        "cluster_id": [partition.get(t) for t in transaction_ids],
        "graph_degree": [degree.get(t, 0) for t in transaction_ids],
        "graph_weighted_degree": [weighted_degree.get(t, 0.0) for t in transaction_ids],
        "graph_pagerank": [pagerank.get(t) for t in transaction_ids],
    })


def build_cluster_summary(
    partition: dict, identity_graph: nx.Graph, behavioral_graph: nx.Graph, min_size: int = 2
) -> pd.DataFrame:
    """One row per non-trivial cluster (size >= min_size): size, has_identity, has_behavioral.

    Singleton clusters are deliberately omitted -- cluster_category() treats
    any cluster of size <= 1 as 'isolated' regardless of edge flags, so
    there's nothing worth storing for the ~85% of clusters that are singletons.
    """
    cluster_sizes = pd.Series(partition).value_counts()
    edge_types = classify_cluster_edge_types(partition, identity_graph, behavioral_graph)

    non_trivial = cluster_sizes[cluster_sizes >= min_size]
    return pd.DataFrame({
        "cluster_id": non_trivial.index,
        "size": non_trivial.to_numpy(),
        "has_identity": [edge_types["has_identity"].get(c, False) for c in non_trivial.index],
        "has_behavioral": [edge_types["has_behavioral"].get(c, False) for c in non_trivial.index],
    })


def build_cluster_edges(
    partition: dict, identity_graph: nx.Graph, behavioral_graph: nx.Graph, min_size: int = 2
) -> pd.DataFrame:
    """One row per intra-cluster edge, for non-trivial clusters only: cluster_id, source,
    target, edge_type ('identity'/'behavioral'/'both'), weight -- everything the network-view
    endpoint needs, without ever holding the full graphs in the serving process.

    Cross-cluster edges are dropped -- measured at 108,864 of 1,073,904 combined-graph
    edges (10.1%). Louvain optimizes global modularity, not a zero-cut partition, so some
    edges connecting two different communities are expected, not a bug. They're excluded
    here for the same reason the old `graph.subgraph(cluster_members)` call implicitly
    excluded them too: a specific cluster's network view can only show edges between two
    of *that cluster's own* members.
    """
    cluster_sizes = pd.Series(partition).value_counts()
    eligible = set(cluster_sizes[cluster_sizes >= min_size].index)

    edge_weights: dict[tuple, dict] = {}
    for graph, kind in ((identity_graph, "identity"), (behavioral_graph, "behavioral")):
        for u, v, w in graph.edges(data="weight"):
            cu = partition.get(u)
            if cu is None or cu != partition.get(v) or cu not in eligible:
                continue
            key = (cu, u, v) if u < v else (cu, v, u)
            edge_weights.setdefault(key, {"identity": 0.0, "behavioral": 0.0})[kind] = w

    rows = []
    for (cluster_id, u, v), weights in edge_weights.items():
        edge_type = "both" if weights["identity"] and weights["behavioral"] else ("identity" if weights["identity"] else "behavioral")
        rows.append({
            "cluster_id": cluster_id, "source": u, "target": v,
            "edge_type": edge_type, "weight": weights["identity"] + weights["behavioral"],
        })
    return pd.DataFrame(rows, columns=["cluster_id", "source", "target", "edge_type", "weight"])
