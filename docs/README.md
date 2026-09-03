# Documentation

This folder explains **what the AI Risk Manager is and why it's built this way**, organized by concept.

## Read in this order

1. [Problem & Approach](problem-and-approach.md) — the problem we're solving, and why it's three signals instead of one
2. [Dataset](dataset.md) — what data powers the system, and its known limitations
3. [Classifier](classifier.md) — the transaction-level fraud detector (signal 1)
4. [Identity Graph](identity-graph.md) — the shared-card/address/email ring detector (signal 2)
5. [Behavioral Graph](behavioral-graph.md) — the similar-behavior ring detector (signal 3), and how it combines with the identity graph
6. [Ring Evaluation](ring-evaluation.md) — checking the blind clusters against real fraud labels for the first time
7. [Hybrid Merge](hybrid-merge.md) — combining all three signals, honestly reporting what worked and what didn't
8. [Rule-Based Flags](rule-based-flags.md) — simple deterministic rules run alongside the ML score, the way real payment-fraud systems do it
9. [Case Queue & API](case-queue-and-api.md) — the FastAPI service, plain-English case reasons, the Streamlit case queue, and the persistent audit log/weak-label loop

## Status

_Last updated: 2026-08-26._

| Piece | Status |
|---|---|
| Dataset understanding (EDA) | Done |
| Transaction classifier | **Done and locked** — XGBoost, cost-based threshold, SHAP explainability |
| Identity graph | **Built** — 803,453 edges, 41.1% of transactions connected |
| Behavioral graph | **Built and combined with identity graph, then clustered** — 265,011 Louvain clusters |
| Ring evaluation | **Done** — 1.18x fraud lift overall; behavioral-only clusters are the strongest sub-signal at 1.50x lift, identity-only clusters underperform baseline at 0.57x |
| Hybrid scoring | **Done** — six attempts tried and reported honestly (see `docs/hybrid-merge.md`): global rescale loses money; an opt-in "second look" mode is adopted as an analyst sensitivity dial (not a cost win); flat-label fusion (v3) and GNN embeddings (v5) both under-deliver for identifiable reasons; **v3.1 (ring category + graph degree/PageRank) is the best result overall (₹330,948, cheapest of all six) and is adopted as the primary classifier** |
| Rule-based flags | **Done** — 3 causal rules (velocity/new-card/amount-spike), 1.10x-1.35x lift; catch 21.9% of the classifier's remaining misses independently |
| FastAPI service | **Done** — `/score`, `/batch-score`, `/cluster-alerts` (+ category filter), `/cluster-alerts/{id}/graph`, `/decision`, `/decisions` |
| Case-queue product | **Done** — Streamlit UI, plain-English SHAP-driven reasons, sensitivity dial, on-demand network view, simulated stream |
| Audit log + weak labels | **Done** — persistent SQLite store, decisions mapped to weak labels (decline→fraud, approve→not fraud), confirmed-vs-dismissed trend |
| Deployment | **Prepared, not shipped** — Dockerfile validated locally (1.85GB image, ~1.7GB runtime memory); parked until the product is fully built, per direction |
