"""Case-queue UI -- Day 10.

The actual product surface (see docs/problem-and-approach.md, Section 2a):
a prioritized queue of open cases -- solo-fraud transactions, and rings --
each with a plain-English reason instead of a raw score, resolving into an
analyst action. A thin client of the FastAPI service (src/api.py); it never
touches the model, graphs, or rules directly.

Run with (two terminals):
    uvicorn src.api:app --reload
    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from pyvis.network import Network

st.set_page_config(page_title="AI Risk Manager -- Case Queue", layout="wide")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DECISION_ACTIONS = ["approve", "hold", "decline", "escalate"]
DECISION_ICONS = {"approve": "✅", "hold": "⏸️", "decline": "❌", "escalate": "🚨"}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

st.session_state.setdefault("resolved_cases", set())
st.session_state.setdefault("txn_cases", None)
st.session_state.setdefault("cluster_cases", None)
st.session_state.setdefault("last_error", None)
st.session_state.setdefault("stream_seed", None)

STREAM_BATCH_SIZE = 20


# ---------------------------------------------------------------------------
# API client helpers -- every one of these is a plain HTTP call to src/api.py
# ---------------------------------------------------------------------------

def api_base_url() -> str:
    return st.session_state.get("api_base_url", "http://127.0.0.1:8000")


def fetch_sample_transactions(n: int, seed: int) -> list[dict]:
    """A stand-in for a live transaction stream -- Day 11 replaces this with
    the real simulated-stream replay. Pulled from the API's own
    /dev/sample-transactions endpoint (not read from the raw dataset here)
    so this UI stays a true thin client -- it works even when deployed
    separately from the API and never needs the large gitignored dataset
    on its own host.
    """
    resp = requests.get(f"{api_base_url()}/dev/sample-transactions", params={"n": n, "seed": seed}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_batch_scores(transactions: list[dict], second_look_floor: float | None) -> list[dict]:
    payload = {"transactions": transactions, "second_look_floor": second_look_floor}
    resp = requests.post(f"{api_base_url()}/batch-score", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_cluster_graph(cluster_id: int, max_nodes: int = 30) -> dict:
    resp = requests.get(
        f"{api_base_url()}/cluster-alerts/{cluster_id}/graph", params={"max_nodes": max_nodes}, timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def fetch_cluster_alerts(min_size: int, limit: int, categories: list[str] | None = None) -> list[dict]:
    params = {"min_size": min_size, "limit": limit}
    if categories:
        params["category"] = categories  # requests repeats the key for a list value
    resp = requests.get(f"{api_base_url()}/cluster-alerts", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def post_decision(payload: dict) -> dict:
    resp = requests.post(f"{api_base_url()}/decision", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_decisions() -> list[dict]:
    resp = requests.get(f"{api_base_url()}/decisions", timeout=15)
    resp.raise_for_status()
    return resp.json()


EDGE_COLORS = {"identity": "#4C78A8", "behavioral": "#F58518", "both": "#54A24B"}


def render_cluster_network(graph_data: dict) -> str:
    """A pyvis network (nodes = transactions, edges colored by identity/
    behavioral/both) as an HTML string, for embedding via components.html.
    Shows *why* transactions are linked, not just that they are -- the
    member list alone can't distinguish a shared-card ring from a
    shared-behavior ring.
    """
    net = Network(height="380px", width="100%", bgcolor="#ffffff", font_color="#222222", notebook=False)
    net.barnes_hut(gravity=-3000, spring_length=120)

    for node in graph_data["nodes"]:
        net.add_node(node, label=str(node), title=f"Transaction {node}")
    for edge in graph_data["edges"]:
        net.add_edge(
            edge["source"], edge["target"],
            color=EDGE_COLORS.get(edge["type"], "#999999"),
            title=f"{edge['type']} (weight {edge['weight']:.2f})",
        )
    return net.generate_html()


@st.cache_data(show_spinner=False)
def load_second_look_curve() -> pd.DataFrame | None:
    """The real, offline-evaluated ₹ trade-off for the second-look dial (see
    docs/hybrid-merge.md) -- shown as reference context next to the live
    slider, since a small in-session sample is far too noisy to show the
    real cost/benefit itself.
    """
    path = RESULTS_DIR / "hybrid_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    curve = data.get("attempt_2_second_look", {}).get("sensitivity_curve")
    return pd.DataFrame(curve) if curve else None


# ---------------------------------------------------------------------------
# Sidebar -- controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Session controls")
    st.session_state["api_base_url"] = st.text_input("API base URL", value=api_base_url())
    analyst_name = st.text_input("Analyst", value="priya")

    st.divider()
    st.subheader("Transaction queue")
    sample_size = st.slider("Transactions to pull", min_value=20, max_value=500, value=150, step=10)
    sample_seed = st.number_input("Sample seed", value=42, step=1)

    st.divider()
    st.subheader("Sensitivity dial")
    enable_second_look = st.checkbox("Enable second-look mode", value=True)
    second_look_floor = None
    if enable_second_look:
        second_look_floor = st.slider(
            "Second-look floor", min_value=0.05, max_value=0.80, value=0.50, step=0.05,
            help="Lower = catches more graph-flagged fraud the classifier missed, at a higher review cost. "
                 "See docs/hybrid-merge.md -- this never removes a classifier flag, only adds extra reviews.",
        )
        curve = load_second_look_curve()
        if curve is not None:
            nearest = curve.iloc[(curve["lower_bound"] - second_look_floor).abs().idxmin()]
            st.caption(
                f"**Offline holdout reference @ floor {nearest['lower_bound']:.2f}:** "
                f"+{int(nearest['n_fraud_caught'])} fraud caught, +{int(nearest['n_flagged']):,} extra reviews, "
                f"net ₹{nearest['net_rs']:,.0f}. This is the real, full-holdout evaluated trade-off -- "
                f"not a live number from this small session sample."
            )

    st.divider()
    st.subheader("Ring alerts")
    ring_min_size = st.slider("Minimum cluster size", min_value=2, max_value=50, value=5)
    ring_limit = st.slider("Max ring cases shown", min_value=1, max_value=20, value=5)
    ring_categories = st.multiselect(
        "Ring categories to include",
        options=["behavioral_only", "both", "identity_only"],
        default=["behavioral_only", "both", "identity_only"],
        help="All three are ranked together by trust weight by default -- identity_only scores "
        "0.58x (below baseline), so it's almost never in the top results unless you narrow the "
        "filter down to it specifically.",
    )

    st.divider()
    load_clicked = st.button("Load / refresh queue", type="primary", width='stretch')


# ---------------------------------------------------------------------------
# Load queue
# ---------------------------------------------------------------------------

if load_clicked or st.session_state["txn_cases"] is None:
    try:
        sample = fetch_sample_transactions(int(sample_size), int(sample_seed))
        txn_scores = fetch_batch_scores(sample, second_look_floor)
        cluster_alerts = fetch_cluster_alerts(int(ring_min_size), int(ring_limit), ring_categories)
        st.session_state["txn_cases"] = [row for row in txn_scores if row["decision"] != "pass"]
        st.session_state["cluster_cases"] = cluster_alerts
        st.session_state["resolved_cases"] = set()
        st.session_state["stream_seed"] = None  # a fresh load restarts the simulated stream too
        st.session_state["last_error"] = None
    except requests.exceptions.RequestException as exc:
        st.session_state["last_error"] = str(exc)

if st.session_state["last_error"]:
    st.error(
        f"Couldn't reach the API at {api_base_url()}. Is it running? "
        f"Start it with `uvicorn src.api:app --reload`.\n\nDetails: {st.session_state['last_error']}"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Main -- open cases / review history
# ---------------------------------------------------------------------------

st.title("AI Risk Manager -- Case Queue")

txn_cases = st.session_state["txn_cases"] or []
cluster_cases = st.session_state["cluster_cases"] or []

open_txn_cases = [c for c in txn_cases if f"txn:{c['TransactionID']}" not in st.session_state["resolved_cases"]]
open_cluster_cases = [c for c in cluster_cases if f"cluster:{c['cluster_id']}" not in st.session_state["resolved_cases"]]

tab_queue, tab_history = st.tabs(["Open cases", "Review history"])

with tab_queue:
    col1, col2, col3 = st.columns(3)
    col1.metric("Open transaction cases", len(open_txn_cases))
    col2.metric("Open ring cases", len(open_cluster_cases))
    col3.metric("Resolved this session", len(st.session_state["resolved_cases"]))

    if st.button(f"▶ Advance stream (+{STREAM_BATCH_SIZE} new transactions)"):
        try:
            base_seed = st.session_state["stream_seed"]
            st.session_state["stream_seed"] = int(sample_seed) + 1 if base_seed is None else base_seed + 1
            new_sample = fetch_sample_transactions(STREAM_BATCH_SIZE, st.session_state["stream_seed"])
            new_scores = fetch_batch_scores(new_sample, second_look_floor)
            existing_ids = {c["TransactionID"] for c in (st.session_state["txn_cases"] or [])}
            new_cases = [r for r in new_scores if r["decision"] != "pass" and r["TransactionID"] not in existing_ids]
            st.session_state["txn_cases"] = (st.session_state["txn_cases"] or []) + new_cases
            st.session_state["last_error"] = None
            st.toast(f"Stream advanced: {len(new_cases)} new case(s) arrived.")
        except requests.exceptions.RequestException as exc:
            st.session_state["last_error"] = str(exc)
        st.rerun()
    st.caption(
        "Simulates a live transaction feed by pulling the next slice of held-out data on demand -- "
        "see docs/problem-and-approach.md's 'simulated, not live' caveat. New cases are added to the "
        "queue below, existing open cases are never touched."
    )

    st.subheader("Transaction cases")
    if not open_txn_cases:
        st.info("No open transaction cases in this sample. Try a larger sample or a lower second-look floor.")
    for case in sorted(open_txn_cases, key=lambda c: c["classifier_score"], reverse=True):
        txn_id = case["TransactionID"]
        badge = "🔴 FLAGGED" if case["decision"] == "flag" else "🟡 SECOND LOOK"
        with st.container(border=True):
            st.markdown(f"**{badge} -- Transaction `{txn_id}`** (score {case['classifier_score']:.2f})")
            st.write(case["case_reason"] or "_No specific driver identified -- borderline score._")
            st.caption(f"Ring evidence: {case['ring_category']} (cluster size {case['ring_cluster_size']})")

            cols = st.columns(4)
            for action, col in zip(DECISION_ACTIONS, cols):
                if col.button(f"{DECISION_ICONS[action]} {action.title()}", key=f"txn_{txn_id}_{action}", width='stretch'):
                    post_decision({
                        "transaction_id": int(txn_id), "action": action, "analyst": analyst_name,
                        "context_score": case["classifier_score"], "context_category": case["ring_category"],
                        "context_reason": case["case_reason"],
                    })
                    st.session_state["resolved_cases"].add(f"txn:{txn_id}")
                    st.rerun()

    st.subheader("Ring cases")
    if not open_cluster_cases:
        st.info("No open ring cases at this size threshold.")
    for case in open_cluster_cases:
        cluster_id = case["cluster_id"]
        with st.container(border=True):
            st.markdown(
                f"**🔵 RING ALERT -- Cluster `{cluster_id}`** -- {case['size']} linked transactions "
                f"({case['category']}, trust weight {case['trust_weight']:.2f}x)"
            )
            with st.expander(f"Members & network ({len(case['member_transaction_ids'])} of {case['size']} shown)"):
                st.write(", ".join(str(t) for t in case["member_transaction_ids"]))

                network_key = f"network_html_{cluster_id}"
                if st.button("Load network view", key=f"loadnet_{cluster_id}"):
                    try:
                        graph_data = fetch_cluster_graph(cluster_id)
                        st.session_state[network_key] = render_cluster_network(graph_data)
                    except requests.exceptions.RequestException as exc:
                        st.warning(f"Couldn't load network view: {exc}")
                if network_key in st.session_state:
                    st.caption("Blue = shared card/address/email, orange = similar behavior, green = both.")
                    st.iframe(st.session_state[network_key], height=400)

            cols = st.columns(4)
            for action, col in zip(DECISION_ACTIONS, cols):
                if col.button(f"{DECISION_ICONS[action]} {action.title()}", key=f"cluster_{cluster_id}_{action}", width='stretch'):
                    post_decision({
                        "cluster_id": int(cluster_id), "action": action, "analyst": analyst_name,
                        "context_score": case["trust_weight"], "context_category": case["category"],
                    })
                    st.session_state["resolved_cases"].add(f"cluster:{cluster_id}")
                    st.rerun()

WEAK_LABEL_TEXT = {1: "🔴 Confirmed fraud", 0: "🟢 Dismissed (false positive)", None: "⚪ No verdict yet"}

with tab_history:
    st.subheader("Review history")
    st.caption(
        "Persisted in SQLite (`data/processed/audit_log.db`) -- survives an API restart, unlike Day 9's "
        "in-memory store. Each decision doubles as a weak label: Decline -> confirmed fraud, "
        "Approve -> dismissed as a false positive, Hold/Escalate -> no verdict yet (see src/audit_log.py)."
    )
    try:
        decisions = fetch_decisions()
    except requests.exceptions.RequestException as exc:
        st.error(f"Couldn't load decision history: {exc}")
        decisions = []

    if not decisions:
        st.info("No decisions recorded yet -- resolve a case in the Open cases tab.")
    else:
        history_df = pd.DataFrame(decisions)
        history_df["recorded_at"] = pd.to_datetime(history_df["recorded_at"])
        history_df["verdict"] = history_df["weak_label"].map(WEAK_LABEL_TEXT)

        col1, col2, col3 = st.columns(3)
        col1.metric("Confirmed fraud (weak label = 1)", int((history_df["weak_label"] == 1).sum()))
        col2.metric("Dismissed as false positive (weak label = 0)", int((history_df["weak_label"] == 0).sum()))
        col3.metric("No verdict yet (hold/escalate)", int(history_df["weak_label"].isna().sum()))

        st.subheader("Confirmed vs. dismissed over time")
        labeled = history_df[history_df["weak_label"].notna()].sort_values("recorded_at").copy()
        if labeled.empty:
            st.info("No confirmed/dismissed decisions yet -- Decline or Approve a case to see the trend.")
        else:
            labeled["Confirmed fraud"] = (labeled["weak_label"] == 1).cumsum()
            labeled["Dismissed (false positive)"] = (labeled["weak_label"] == 0).cumsum()
            trend = labeled.set_index("recorded_at")[["Confirmed fraud", "Dismissed (false positive)"]]
            st.line_chart(trend)
            st.caption(
                "Cumulative counts -- a real deployment would bucket this by day/week once decision "
                "volume is high enough for a rate to be meaningful; a single demo session is too short for that."
            )

        st.subheader("All decisions")
        display_cols = [
            "recorded_at", "transaction_id", "cluster_id", "action", "verdict",
            "analyst", "context_score", "context_category", "note",
        ]
        st.dataframe(
            history_df[display_cols].sort_values("recorded_at", ascending=False),
            width='stretch', hide_index=True,
        )
