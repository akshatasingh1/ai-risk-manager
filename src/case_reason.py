"""Plain-English case reasons for the case queue -- Day 10.

Translates the classifier's SHAP-driven signal, the graph's ring category,
and the rule-based flags (already plain English, see src/rules.py) into the
one or two sentences an analyst actually needs, instead of a raw score --
see Section 2a of the project plan: "a plain-English reason, not a raw SHAP
plot."

Feature -> plain-English category groupings follow docs/classifier.md's own
Day 4 SHAP write-up, so the case queue's language matches what's already
documented and defended for the classifier.
"""

import numpy as np
import pandas as pd
import shap

C_COLS = {f"C{i}" for i in range(1, 15)}
D_COLS = {f"D{i}" for i in range(1, 16)}
M_COLS = {f"M{i}" for i in range(1, 10)}
ID_COLS = {f"id_{i:02d}" for i in range(1, 12)}

EXPLICIT_DESCRIPTIONS = {
    "TransactionAmt": "an unusual transaction amount",
    "ProductCD": "the type of product purchased",
    "addr1": "the billing address region",
    "addr2": "the billing country region",
    "dist1": "the distance between billing and shipping/device",
    "dist2": "the distance between billing and shipping/device",
    "P_emaildomain": "the purchaser's email domain",
    "R_emaildomain": "the recipient's email domain",
    "DeviceType": "the device type used",
    "ring_cluster_size": "how many other transactions this one is linked to",
    "graph_degree": "how many other transactions this one is directly linked to",
    "graph_weighted_degree": "the strength of this transaction's links to others",
    "graph_pagerank": "how central this transaction is within its linked cluster",
}

RING_CATEGORY_PHRASES = {
    "behavioral_only": "shares suspicious behavioral patterns (amount/timing/velocity) with",
    "identity_only": "shares a card, address, or email with",
    "both": "shares both identity details and behavioral patterns with",
}


def describe_feature(base_name: str) -> str:
    """Map a raw column name to the plain-English category it belongs to."""
    if base_name in EXPLICIT_DESCRIPTIONS:
        return EXPLICIT_DESCRIPTIONS[base_name]
    if base_name in C_COLS:
        return "a high number of linked identities/cards/addresses"
    if base_name in D_COLS:
        return "the time since related past activity on this card"
    if base_name in M_COLS:
        return "a mismatch between billing and shipping details"
    if base_name in ID_COLS:
        return "a device/identity signal"
    if base_name.startswith("card"):
        return "characteristics of the card used"
    if base_name.startswith("V"):
        return f"an anonymized behavioral signal ({base_name})"
    return base_name


def _base_feature_name(transformed_name: str, categorical_cols: list[str]) -> str:
    """Undo the ColumnTransformer's 'num__'/'cat__' prefixing (and one-hot's
    trailing '_<category value>') to recover the original raw column name.
    """
    if transformed_name.startswith("num__"):
        return transformed_name[len("num__"):]
    if transformed_name.startswith("cat__"):
        rest = transformed_name[len("cat__"):]
        for col in sorted(categorical_cols, key=len, reverse=True):
            if rest == col or rest.startswith(col + "_"):
                return col
        return rest
    return transformed_name


def build_shap_explainer(model):
    """A shap.TreeExplainer built once (at ScoringContext startup) and reused
    -- rebuilding it per request would be needless, repeated work.
    """
    return shap.TreeExplainer(model.named_steps["clf"])


def top_classifier_reasons(
    model, explainer, raw_df: pd.DataFrame, categorical_cols: list[str], top_n: int = 2
) -> list[str]:
    """For each row, the top `top_n` distinct features pushing the score
    *toward* fraud, translated to plain English. Empty string if nothing
    pushes toward fraud (SHAP explains a specific prediction, not always in
    the fraud direction, e.g. a case only flagged via the ring/rules path).
    """
    if raw_df.empty:
        return []

    transformed = model.named_steps["prep"].transform(raw_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = model.named_steps["prep"].get_feature_names_out()

    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):  # some SHAP/model combinations return [neg_class, pos_class]
        shap_values = shap_values[1]

    reasons = []
    for row_values in shap_values:
        order = np.argsort(row_values)[::-1]  # most positive (toward fraud) first
        picked, seen_descriptions = [], set()
        for idx in order:
            if row_values[idx] <= 0:
                break
            base = _base_feature_name(feature_names[idx], categorical_cols)
            description = describe_feature(base)
            if description in seen_descriptions:  # two different columns can share one description
                continue
            seen_descriptions.add(description)
            picked.append(description)
            if len(picked) >= top_n:
                break
        reasons.append("driven mainly by " + " and ".join(picked) if picked else "")
    return reasons


def ring_phrase(category: str, cluster_size: int) -> str:
    if category not in RING_CATEGORY_PHRASES:
        return ""
    others = cluster_size - 1
    return f"{RING_CATEGORY_PHRASES[category]} {others} other transaction{'s' if others != 1 else ''}"


def _sentence_case(text: str) -> str:
    """Capitalize only the first letter -- str.capitalize() would also
    lowercase mid-sentence tokens like 'V312' into 'v312'.
    """
    return text[:1].upper() + text[1:] if text else text


def combine_case_reason(classifier_phrase: str, ring_text: str, rule_reasons: str) -> str:
    """Join whichever pieces of evidence actually fired into one sentence.
    Any of the three can be empty (a case can be flagged by only one signal).
    """
    parts = []
    if classifier_phrase:
        parts.append(_sentence_case(classifier_phrase))
    if ring_text:
        parts.append(f"Also, this transaction {ring_text}" if parts else _sentence_case(ring_text))
    if rule_reasons:
        parts.append(f"Rule flags: {rule_reasons}")
    return ". ".join(parts) + ("." if parts else "")
