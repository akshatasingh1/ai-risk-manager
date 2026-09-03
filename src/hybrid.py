"""Combining the classifier's score with ring/graph evidence into one hybrid
score -- Day 8.

Category weights must be fit ONLY on the train split (never the holdout) to
avoid the holdout's own labels leaking into the weights used to score it --
see docs/hybrid-merge.md.
"""

import networkx as nx
import numpy as np
import pandas as pd

from src.cost import DEFAULT_REVIEW_COST
from src.ring_eval import classify_cluster_edge_types

NEUTRAL_WEIGHT = 1.0
DEFAULT_BOOST_CATEGORIES = ("behavioral_only", "both")


def cluster_category(cluster_size: int, has_identity: bool, has_behavioral: bool) -> str:
    """Classify a transaction's cluster into one of five evidence categories.

    'isolated': singleton cluster, no graph evidence at all.
    'neither': non-trivial cluster with no detected edge of either type
    inside it -- a theoretical Louvain edge case (Day 7 found 0 of these).
    """
    if cluster_size <= 1:
        return "isolated"
    if has_identity and has_behavioral:
        return "both"
    if has_identity:
        return "identity_only"
    if has_behavioral:
        return "behavioral_only"
    return "neither"


def assign_cluster_categories(
    transaction_ids,
    cluster_ids_by_node: dict,
    identity_graph: nx.Graph,
    behavioral_graph: nx.Graph,
) -> pd.Series:
    """Per-transaction evidence category, indexed by transaction id.

    Cluster size is computed from the FULL partition (`cluster_ids_by_node`,
    covering all transactions), not just from `transaction_ids` -- a cluster
    split across train and holdout must not look artificially smaller (or
    "isolated") when scoring one subset alone.
    """
    transaction_ids = list(transaction_ids)
    global_cluster_sizes = pd.Series(cluster_ids_by_node).value_counts()

    cluster_ids = pd.Series(
        [cluster_ids_by_node[t] for t in transaction_ids], index=transaction_ids
    )
    cluster_sizes = cluster_ids.map(global_cluster_sizes)

    edge_types = classify_cluster_edge_types(cluster_ids_by_node, identity_graph, behavioral_graph)
    has_identity = cluster_ids.map(lambda c: edge_types["has_identity"].get(c, False))
    has_behavioral = cluster_ids.map(lambda c: edge_types["has_behavioral"].get(c, False))

    categories = [
        cluster_category(size, ident, behav)
        for size, ident, behav in zip(cluster_sizes, has_identity, has_behavioral)
    ]
    return pd.Series(categories, index=transaction_ids, name="category")


def compute_category_weights(categories: pd.Series, is_fraud: pd.Series, default_weight: float = NEUTRAL_WEIGHT) -> dict:
    """Lift weight per category = (category fraud rate) / (population fraud rate),
    computed only from the rows passed in. 'isolated' and 'neither' are always
    hardcoded to neutral (1.0), never estimated empirically.
    """
    population_fraud_rate = is_fraud.mean()
    df = pd.DataFrame({"category": categories.values, "isFraud": is_fraud.values})

    weights: dict[str, float] = {}
    for category, group in df.groupby("category"):
        if category in ("isolated", "neither"):
            weights[category] = NEUTRAL_WEIGHT
            continue
        category_fraud_rate = group["isFraud"].mean()
        weights[category] = category_fraud_rate / population_fraud_rate if population_fraud_rate else default_weight

    weights.setdefault("isolated", NEUTRAL_WEIGHT)
    weights.setdefault("neither", NEUTRAL_WEIGHT)
    for category in ("identity_only", "behavioral_only", "both"):
        weights.setdefault(category, default_weight)

    return weights


def apply_hybrid_score(classifier_proba: np.ndarray, categories: pd.Series, weights: dict) -> np.ndarray:
    """hybrid_score = classifier_proba * weight(category).

    Not clipped to [0, 1] -- this is an unnormalized risk score, not a
    calibrated probability (a 1.5x weight can push a score above 1.0).
    """
    weight_values = categories.map(weights).to_numpy()
    return np.asarray(classifier_proba) * weight_values


def classifier_missed_hybrid_caught(
    y_true, classifier_proba, hybrid_score, classifier_threshold: float, hybrid_threshold: float
) -> np.ndarray:
    """Boolean mask: real fraud the classifier's threshold missed that the
    hybrid threshold catches. The headline-finding primitive.
    """
    y_true = np.asarray(y_true)
    classifier_proba = np.asarray(classifier_proba)
    hybrid_score = np.asarray(hybrid_score)

    classifier_missed = (y_true == 1) & (classifier_proba < classifier_threshold)
    hybrid_caught = hybrid_score >= hybrid_threshold
    return classifier_missed & hybrid_caught


def second_look_mask(
    classifier_proba,
    categories: pd.Series,
    lower_bound: float,
    classifier_threshold: float,
    boost_categories: tuple = DEFAULT_BOOST_CATEGORIES,
) -> np.ndarray:
    """Boolean mask for an opt-in 'second look': transactions the classifier
    did NOT flag (score below `classifier_threshold`), but which scored at
    least `lower_bound` AND carry ring evidence from a trusted category.

    Never touches transactions the classifier already flagged -- this is
    additive only, so it can never cause a regression on the classifier's
    own correct calls.
    """
    classifier_proba = np.asarray(classifier_proba)
    in_band = (classifier_proba >= lower_bound) & (classifier_proba < classifier_threshold)
    has_ring_evidence = categories.isin(boost_categories).to_numpy()
    return in_band & has_ring_evidence


def second_look_sensitivity_curve(
    y_true,
    classifier_proba,
    categories: pd.Series,
    amounts,
    classifier_threshold: float,
    lower_bounds,
    review_cost: float = DEFAULT_REVIEW_COST,
    boost_categories: tuple = DEFAULT_BOOST_CATEGORIES,
) -> pd.DataFrame:
    """The analyst's sensitivity dial: for each candidate lower_bound, how many
    extra reviews it costs vs. how much extra fraud it catches -- reported as
    precision/capture-rate/review-burden, not as a single $-optimal answer.
    Not expected to show a net $ saving (see docs/hybrid-merge.md) -- this is
    an opt-in higher-recall mode, evaluated on its own terms.
    """
    y_true = np.asarray(y_true)
    classifier_proba = np.asarray(classifier_proba)
    amounts = np.asarray(amounts)

    total_fraud_missed = int(((y_true == 1) & (classifier_proba < classifier_threshold)).sum())

    rows = []
    for lb in lower_bounds:
        mask = second_look_mask(classifier_proba, categories, lb, classifier_threshold, boost_categories)
        n_flagged = int(mask.sum())
        n_fraud_caught = int(y_true[mask].sum())
        fraud_recovered = float(amounts[mask & (y_true == 1)].sum())
        review_cost_total = n_flagged * review_cost

        rows.append({
            "lower_bound": lb,
            "n_flagged": n_flagged,
            "n_fraud_caught": n_fraud_caught,
            "precision": n_fraud_caught / n_flagged if n_flagged else 0.0,
            "capture_rate_of_misses": n_fraud_caught / total_fraud_missed if total_fraud_missed else 0.0,
            "fraud_amount_recovered_rs": fraud_recovered,
            "review_cost_rs": review_cost_total,
            "net_rs": fraud_recovered - review_cost_total,
        })

    return pd.DataFrame(rows)
