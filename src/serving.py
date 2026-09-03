"""Inference-time feature assembly and scoring -- Day 9.

Reproduces, for one or more incoming transactions, exactly the feature
pipeline v3.1 was trained on (see notebooks/08e_v3_1_structural_features.ipynb):
graph degree/weighted-degree/PageRank + ring category from the saved graphs
and cluster partition, plus the rule-based flags from src/rules.py -- then
scores through the locked v3.1 classifier (models/classifier_v3_1.joblib).

A transaction whose TransactionID isn't in the training-time cluster
partition (never seen while building the graph) falls back to the same
neutral values a true singleton node gets: ring_category='isolated',
zero degree, population-minimum PageRank. This keeps scoring well-defined
for a genuinely new transaction, not just a holdout replay.

Graph lookups are served from precomputed tables (src/precompute.py), not
from the live identity/behavioral/combined graphs -- holding those three
networkx graphs in memory measured at ~1.39GB for data that's 154MB on disk
(networkx's per-edge Python dict overhead), for values (degree, PageRank,
cluster membership) that never change per request. See
docs/case-queue-and-api.md for the measured before/after.
"""

import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.case_reason import build_shap_explainer, combine_case_reason, ring_phrase, top_classifier_reasons
from src.hybrid import cluster_category, second_look_mask
from src.rules import apply_all_rules

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Locked in notebooks/08e_v3_1_structural_features.ipynb / results/v3_1_structural_features_metrics.json
CLASSIFIER_THRESHOLD = 0.84


class ScoringContext:
    """Loads every serving artifact once; reused across requests.

    `degree`/`weighted_degree`/`pagerank`/`partition`/`cluster_sizes` are
    pandas Series (not dicts): a Series backed by a numpy array costs a few
    MB for 590K entries, where the equivalent Python dict measured ~100MB --
    and every place downstream that reads them (`.map()`, `.get()`,
    `.items()`) works identically against a Series, so nothing else in this
    module needed to change to get that saving.
    """

    def __init__(self, models_dir: Path = MODELS_DIR, processed_dir: Path = PROCESSED_DIR):
        self.model = joblib.load(models_dir / "classifier_v3_1.joblib")

        with open(models_dir / "category_weights.json") as f:
            self.category_weights = json.load(f)["weights"]
        with open(models_dir / "rule_thresholds.json") as f:
            self.amount_threshold_new_card = json.load(f)["amount_threshold_new_card"]

        graph_features = pd.read_parquet(processed_dir / "transaction_graph_features.parquet").set_index("TransactionID")
        self.partition = graph_features["cluster_id"]
        self.degree = graph_features["graph_degree"]
        self.weighted_degree = graph_features["graph_weighted_degree"]
        self.pagerank = graph_features["graph_pagerank"]
        self.min_pagerank = self.pagerank.min() if len(self.pagerank) else 0.0

        cluster_summary = pd.read_parquet(processed_dir / "cluster_summary.parquet").set_index("cluster_id")
        self.cluster_sizes = cluster_summary["size"]
        self.has_identity = cluster_summary["has_identity"].to_dict()
        self.has_behavioral = cluster_summary["has_behavioral"].to_dict()

        self.cluster_edges = pd.read_parquet(processed_dir / "cluster_edges.parquet")

        self.expected_columns_by_kind = _expected_columns_by_kind(self.model)
        self.expected_columns = [c for cols in self.expected_columns_by_kind.values() for c in cols]
        self.shap_explainer = build_shap_explainer(self.model)
        self._cluster_members: dict | None = None  # built lazily, only if /cluster-alerts is used

    def cluster_members(self) -> dict:
        """cluster_id -> [TransactionID, ...], built once on first use."""
        if self._cluster_members is None:
            members = defaultdict(list)
            for node, cid in self.partition.items():
                members[cid].append(node)
            self._cluster_members = members
        return self._cluster_members


def _expected_columns_by_kind(model) -> dict[str, list[str]]:
    """The fitted preprocessor's raw column names, grouped by transformer
    ('num'/'cat') -- ColumnTransformer raises if any of these are absent
    from the input, and 'num' columns must additionally arrive as an
    actual numeric dtype (see score_batch).
    """
    prep = model.named_steps["prep"]
    by_kind: dict[str, list[str]] = {}
    for name, _transformer, cols in prep.transformers_:
        if name == "remainder":
            continue
        by_kind[name] = list(cols)
    return by_kind


def add_graph_features(df: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    """Attach ring_category/ring_cluster_size/graph_degree/graph_weighted_degree/
    graph_pagerank -- the same columns v3.1 was trained on.
    """
    df = df.copy()
    txn_ids = df["TransactionID"]
    cluster_ids = txn_ids.map(ctx.partition)

    df["ring_cluster_size"] = cluster_ids.map(ctx.cluster_sizes).fillna(1).astype(int)
    df["graph_degree"] = txn_ids.map(ctx.degree).fillna(0)
    df["graph_weighted_degree"] = txn_ids.map(ctx.weighted_degree).fillna(0.0)
    df["graph_pagerank"] = txn_ids.map(ctx.pagerank).fillna(ctx.min_pagerank)

    df["ring_category"] = [
        "isolated" if pd.isna(cid) else cluster_category(size, ctx.has_identity.get(cid, False), ctx.has_behavioral.get(cid, False))
        for size, cid in zip(df["ring_cluster_size"], cluster_ids)
    ]

    return df


def score_batch(
    df: pd.DataFrame,
    ctx: ScoringContext,
    threshold: float = CLASSIFIER_THRESHOLD,
    second_look_floor: float | None = None,
    include_reason: bool = True,
) -> pd.DataFrame:
    """Score a batch of raw transactions (same schema as the merged IEEE-CIS
    train/holdout data -- TransactionID, TransactionDT, TransactionAmt, card1/
    card2, and the full V/C/D/id_ feature set the classifier was trained on).

    `include_reason` (default True) computes a plain-English `case_reason`
    (SHAP-driven classifier signal + ring evidence + rule flags -- see
    src/case_reason.py) for every row NOT decided 'pass', since only flagged/
    second-look cases actually reach the case queue. SHAP is real per-row
    compute, so a caller scoring a very large batch with no interest in the
    reason text (e.g. a bulk cost-curve sweep) can set this False to skip it.

    Rules are evaluated causally across whatever rows are passed in, sorted
    per-card by TransactionDT (see src/rules.py) -- for a genuine single
    transaction with no batch context, pass its card's recent history
    concatenated ahead of it (see score_one).

    Any raw feature column the classifier was trained on but that isn't
    present in `df` (e.g. a client posting a partial transaction over the
    API) is filled with NaN -- the same as a genuinely missing value in the
    source data, handled by the model's own median/constant imputation.

    Numeric feature columns are coerced with `pd.to_numeric`, and categorical
    ones have any Python `None` normalized to `np.nan`. This matters more
    than it looks: a DataFrame built from a single JSON request (e.g.
    `pd.DataFrame([{"V1": None, "M1": None, ...}])`) infers `object` dtype
    for any column whose only value is `None`, instead of the `float64`
    NaN / proper missing-string a multi-row DataFrame would get -- and
    SimpleImputer/OneHotEncoder treat a raw `None` in an object column
    differently enough (measured: it isn't reliably recognized as "missing")
    to silently shift the score. A DataFrame read straight from parquet
    (e.g. a holdout replay) never hits this, since pandas infers the real
    dtype from hundreds of rows in the same column -- so this bug is
    invisible until the API is scoring one JSON transaction at a time,
    exactly the case this function exists to serve correctly.
    """
    missing_cols = [c for c in ctx.expected_columns if c not in df.columns]
    if missing_cols:
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing_cols)], axis=1)
    else:
        df = df.copy()

    numeric_cols = [c for c in ctx.expected_columns_by_kind.get("num", []) if c in df.columns]
    categorical_cols = [c for c in ctx.expected_columns_by_kind.get("cat", []) if c in df.columns]
    df = df.assign(**{c: pd.to_numeric(df[c], errors="coerce") for c in numeric_cols})
    df = df.assign(**{c: df[c].where(df[c].notna(), np.nan) for c in categorical_cols})

    scored = add_graph_features(df, ctx)

    proba = ctx.model.predict_proba(scored)[:, 1]
    scored["classifier_score"] = proba
    scored["flagged"] = proba >= threshold

    rule_flags = apply_all_rules(scored, amount_threshold_for_new_card=ctx.amount_threshold_new_card)
    scored["rule_reasons"] = rule_flags["rule_reasons"]
    scored["any_rule_triggered"] = rule_flags["any_rule_triggered"]

    if second_look_floor is not None:
        scored["second_look"] = second_look_mask(proba, scored["ring_category"], second_look_floor, threshold)
    else:
        scored["second_look"] = False

    scored["decision"] = "pass"
    scored.loc[scored["second_look"], "decision"] = "second_look"
    scored.loc[scored["flagged"], "decision"] = "flag"

    scored["case_reason"] = ""
    if include_reason:
        needs_reason = scored["decision"] != "pass"
        if needs_reason.any():
            subset = scored.loc[needs_reason]
            classifier_phrases = top_classifier_reasons(ctx.model, ctx.shap_explainer, subset, categorical_cols)
            reasons = [
                combine_case_reason(phrase, ring_phrase(category, size), rule_text)
                for phrase, category, size, rule_text in zip(
                    classifier_phrases, subset["ring_category"], subset["ring_cluster_size"], subset["rule_reasons"]
                )
            ]
            scored.loc[needs_reason, "case_reason"] = reasons

    return scored[[
        "TransactionID", "classifier_score", "decision", "ring_category",
        "ring_cluster_size", "rule_reasons", "any_rule_triggered", "second_look", "case_reason",
    ]]


def score_one(
    transaction: dict,
    ctx: ScoringContext,
    history: pd.DataFrame | None = None,
    threshold: float = CLASSIFIER_THRESHOLD,
    second_look_floor: float | None = None,
) -> dict:
    """Score a single transaction.

    `history` (optional): that card's own recent transactions, strictly
    before this one, so the velocity/amount-spike rules have context to
    compare against. Omit it and those two rules simply can't fire -- the
    correct, honest behavior for a card with no known history, not a bug.
    """
    row = pd.DataFrame([transaction])
    batch = pd.concat([history, row], ignore_index=True) if history is not None else row
    scored = score_batch(batch, ctx, threshold=threshold, second_look_floor=second_look_floor)
    return scored.iloc[-1].to_dict()


DEFAULT_ALERT_CATEGORIES = ("behavioral_only", "both", "identity_only")


def get_cluster_alerts(
    ctx: ScoringContext,
    min_size: int = 2,
    categories: tuple = DEFAULT_ALERT_CATEGORIES,
    limit: int = 20,
    max_members_shown: int = 20,
) -> list[dict]:
    """Non-trivial clusters ranked by train-only trust weight, then size --
    the ring-alert feed for the case queue's ring-based cases.

    Never looks at isFraud: a live cluster's own members have no labels yet.
    Ranking uses the category weights fit offline on train data (see
    docs/hybrid-merge.md) -- a purely structural signal (cluster category,
    size) applied to a purely structural artifact (the cluster itself).
    """
    members = ctx.cluster_members()

    rows = []
    for cluster_id, size in ctx.cluster_sizes.items():
        if size < min_size:
            continue
        category = cluster_category(size, ctx.has_identity.get(cluster_id, False), ctx.has_behavioral.get(cluster_id, False))
        if category not in categories:
            continue
        rows.append({
            "cluster_id": int(cluster_id),
            "size": int(size),
            "category": category,
            "trust_weight": ctx.category_weights.get(category, 1.0),
            "member_transaction_ids": members[cluster_id][:max_members_shown],
        })

    rows.sort(key=lambda r: (r["trust_weight"], r["size"]), reverse=True)
    return rows[:limit]


def get_cluster_graph(ctx: ScoringContext, cluster_id: int, max_nodes: int = 30) -> dict:
    """Nodes + edges for one cluster, for the case queue's network view --
    why these transactions are linked, not just that they are. Capped at
    `max_nodes`: a large cluster's true member count is preserved in
    `total_size`, but rendering hundreds of nodes as a graph isn't legible
    or fast in a browser, so only a subset is shown (same trade-off
    get_cluster_alerts already makes for `member_transaction_ids`).

    Edge type is 'identity' (shared card/address/email), 'behavioral'
    (similar amount/timing/velocity), or 'both'.
    """
    members = ctx.cluster_members().get(cluster_id, [])
    subset = members[:max_nodes]
    subset_set = set(subset)

    cluster_rows = ctx.cluster_edges[ctx.cluster_edges["cluster_id"] == cluster_id]
    edges = [
        {"source": int(row.source), "target": int(row.target), "type": row.edge_type, "weight": float(row.weight)}
        for row in cluster_rows.itertuples()
        if row.source in subset_set and row.target in subset_set
    ]

    return {
        "cluster_id": cluster_id,
        "total_size": int(ctx.cluster_sizes.get(cluster_id, len(members))),
        "nodes": [int(n) for n in subset],
        "edges": edges,
    }
