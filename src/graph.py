"""Identity graph: links transactions that share a card/address/email attribute.

Built without ever looking at `isFraud` -- clustering must stay blind to the
label so evaluating it later (Day 7) is genuine, not circular. See
docs/identity-graph.md for the reasoning behind which attributes were used,
which were excluded, and why.
"""

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# card3, card4, card5, card6, addr2 excluded: each is dominated by one or two
# near-universal values (e.g. one card4 value covers 384,767 of 590,540 rows),
# so sharing them is not identity evidence. card1, card2, addr1 have enough
# distinct values to plausibly represent "the same card/address". See
# docs/identity-graph.md for the actual cardinality numbers behind this.
IDENTITY_ATTRIBUTES = ["card1", "card2", "addr1", "P_emaildomain"]

DEFAULT_TIME_WINDOW_SECONDS = 24 * 60 * 60  # rings operate in bursts, not across 6 months
DEFAULT_MAX_GROUP_SIZE = 1000  # skip attribute values too common to be meaningful (e.g. "gmail.com")


def _edges_for_attribute(
    df: pd.DataFrame,
    attribute: str,
    id_col: str = "TransactionID",
    time_col: str = "TransactionDT",
    time_window: int = DEFAULT_TIME_WINDOW_SECONDS,
    max_group_size: int = DEFAULT_MAX_GROUP_SIZE,
):
    """Yield (id_a, id_b, weight) for transactions sharing `attribute` within `time_window`.

    weight = 1 / (number of transactions sharing this attribute value) --
    a rarer shared value is stronger evidence of a real link.
    """
    sub = df[[id_col, time_col, attribute]].dropna(subset=[attribute])
    group_sizes = sub.groupby(attribute)[id_col].transform("size")
    sub = sub[(group_sizes >= 2) & (group_sizes <= max_group_size)]

    for _, group in sub.groupby(attribute):
        group = group.sort_values(time_col)
        ids = group[id_col].to_numpy()
        times = group[time_col].to_numpy()
        weight = 1.0 / len(ids)

        # For each transaction, find how far forward in time we can pair it
        # with later transactions in this group, all in one vectorized call.
        right_bounds = np.searchsorted(times, times + time_window, side="right")
        for left in range(len(ids)):
            for right in range(left + 1, right_bounds[left]):
                yield ids[left], ids[right], weight


def build_identity_graph(
    df: pd.DataFrame,
    attributes: list[str] = IDENTITY_ATTRIBUTES,
    id_col: str = "TransactionID",
    time_col: str = "TransactionDT",
    time_window: int = DEFAULT_TIME_WINDOW_SECONDS,
    max_group_size: int = DEFAULT_MAX_GROUP_SIZE,
) -> nx.Graph:
    """Build the identity graph: one node per transaction, edges for shared attributes.

    A pair sharing multiple attributes gets a summed edge weight (stronger
    evidence than sharing just one).
    """
    edge_weights: dict[tuple, float] = {}
    for attribute in attributes:
        for a, b, w in _edges_for_attribute(df, attribute, id_col, time_col, time_window, max_group_size):
            key = (a, b) if a < b else (b, a)
            edge_weights[key] = edge_weights.get(key, 0.0) + w

    graph = nx.Graph()
    graph.add_nodes_from(df[id_col])
    graph.add_weighted_edges_from((a, b, w) for (a, b), w in edge_weights.items())
    return graph


def combine_graphs(*graphs: nx.Graph) -> nx.Graph:
    """Union graphs, summing edge weight where the same pair appears in more than one."""
    combined = nx.Graph()
    for g in graphs:
        combined.add_nodes_from(g.nodes)
    for g in graphs:
        for u, v, w in g.edges(data="weight"):
            if combined.has_edge(u, v):
                combined[u][v]["weight"] += w
            else:
                combined.add_edge(u, v, weight=w)
    return combined


def cluster_graph(graph: nx.Graph, seed: int = 42) -> dict:
    """Run Louvain community detection; return {node: cluster_id}.

    Isolated nodes (no edges) each end up in their own singleton cluster.
    """
    communities = nx.community.louvain_communities(graph, weight="weight", seed=seed)
    return {node: cluster_id for cluster_id, community in enumerate(communities) for node in community}


def save_graph(graph: nx.Graph, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path: Path) -> nx.Graph:
    with open(path, "rb") as f:
        return pickle.load(f)
