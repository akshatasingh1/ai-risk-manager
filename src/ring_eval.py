"""Evaluating the blind-built clusters against isFraud -- Day 7.

This is the ONLY place fraud labels are used in the graph pipeline, and only
to evaluate what unsupervised clustering already produced -- never to build
the graph or the clusters themselves. See docs/identity-graph.md and
docs/behavioral-graph.md for why that separation matters.
"""

import networkx as nx
import pandas as pd


def cluster_stats(transaction_ids, cluster_ids, is_fraud) -> pd.DataFrame:
    """Per-cluster size and fraud rate, indexed by cluster_id."""
    df = pd.DataFrame({"cluster_id": cluster_ids, "isFraud": is_fraud})
    grouped = df.groupby("cluster_id")["isFraud"].agg(size="size", fraud_count="sum")
    grouped["fraud_rate"] = grouped["fraud_count"] / grouped["size"]
    return grouped


def evaluate_flagged_clusters(
    stats: pd.DataFrame, min_size: int, total_fraud: int, population_fraud_rate: float
) -> dict:
    """Metrics for clusters of at least `min_size`, treating every member as a flagged case.

    - precision: of everything flagged, what fraction is actually fraud
    - lift: how much more concentrated that is than the population baseline
    - capture_rate: what fraction of ALL fraud in the dataset these flagged clusters contain
    """
    flagged = stats[stats["size"] >= min_size]
    flagged_transactions = int(flagged["size"].sum())
    flagged_fraud = int(flagged["fraud_count"].sum())

    precision = flagged_fraud / flagged_transactions if flagged_transactions else 0.0
    lift = precision / population_fraud_rate if population_fraud_rate else float("nan")
    capture_rate = flagged_fraud / total_fraud if total_fraud else 0.0

    return {
        "min_size": min_size,
        "n_clusters_flagged": int(len(flagged)),
        "n_transactions_flagged": flagged_transactions,
        "n_fraud_captured": flagged_fraud,
        "precision": precision,
        "lift": lift,
        "capture_rate": capture_rate,
    }


def classify_cluster_edge_types(
    cluster_ids_by_node: dict, identity_graph: nx.Graph, behavioral_graph: nx.Graph
) -> dict:
    """For each cluster id, whether any of its members are linked by an identity edge
    and/or a behavioral edge (checked by scanning each graph's edges once, not per-cluster).
    """
    has_identity = {}
    has_behavioral = {}

    for u, v in identity_graph.edges():
        cu, cv = cluster_ids_by_node.get(u), cluster_ids_by_node.get(v)
        if cu is not None and cu == cv:
            has_identity[cu] = True

    for u, v in behavioral_graph.edges():
        cu, cv = cluster_ids_by_node.get(u), cluster_ids_by_node.get(v)
        if cu is not None and cu == cv:
            has_behavioral[cu] = True

    return {"has_identity": has_identity, "has_behavioral": has_behavioral}
