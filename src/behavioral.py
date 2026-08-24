"""Behavioral-similarity graph: links transactions with suspiciously similar
amount/timing/velocity patterns, even when no identifier is shared.

Catches rings that deliberately rotate card/address/email to avoid the
identity graph, but can't easily fake matching operational behavior. Treated
as weaker evidence than a shared identifier when combined -- see
docs/behavioral-graph.md.
"""

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
BEHAVIORAL_FEATURES = ["TransactionAmt"] + C_COLS + D_COLS

DEFAULT_TIME_BUCKET_SECONDS = 15 * 60  # bursts, not the full 6-month timeline
DEFAULT_SIMILARITY_THRESHOLD = 0.98
BEHAVIORAL_WEIGHT_MULTIPLIER = 0.2  # weaker evidence than a shared identifier


def _scaled_features(df: pd.DataFrame, features: list[str] = BEHAVIORAL_FEATURES) -> pd.DataFrame:
    X = df[features].copy()
    X = X.fillna(X.median())
    X["TransactionAmt"] = np.log1p(X["TransactionAmt"])
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=features, index=df.index)


def build_behavioral_graph(
    df: pd.DataFrame,
    id_col: str = "TransactionID",
    time_col: str = "TransactionDT",
    features: list[str] = BEHAVIORAL_FEATURES,
    time_bucket: int = DEFAULT_TIME_BUCKET_SECONDS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    weight_multiplier: float = BEHAVIORAL_WEIGHT_MULTIPLIER,
) -> nx.Graph:
    """Edge if two transactions land in the same time bucket and their
    standardized behavioral feature vectors are cosine-similar above threshold.
    """
    scaled = _scaled_features(df, features)
    work = scaled.copy()
    work[id_col] = df[id_col].to_numpy()
    work["_bucket"] = (df[time_col] // time_bucket).astype(int).to_numpy()

    graph = nx.Graph()
    graph.add_nodes_from(df[id_col])

    for _, group in work.groupby("_bucket"):
        n = len(group)
        if n < 2:
            continue
        ids = group[id_col].to_numpy()
        vectors = group[features].to_numpy()
        sims = cosine_similarity(vectors)

        iu, ju = np.triu_indices(n, k=1)
        sim_values = sims[iu, ju]
        mask = sim_values >= similarity_threshold

        for a, b, s in zip(ids[iu[mask]], ids[ju[mask]], sim_values[mask]):
            graph.add_edge(a, b, weight=float(s) * weight_multiplier)

    return graph
