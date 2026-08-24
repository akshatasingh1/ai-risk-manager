# Documentation

This folder explains **what the AI Risk Manager is and why it's built this way**, organized by concept.

## Read in this order

1. [Problem & Approach](problem-and-approach.md) — the problem we're solving, and why it's three signals instead of one
2. [Dataset](dataset.md) — what data powers the system, and its known limitations
3. [Classifier](classifier.md) — the transaction-level fraud detector (signal 1)
4. [Identity Graph](identity-graph.md) — the shared-card/address/email ring detector (signal 2)
5. [Behavioral Graph](behavioral-graph.md) — the similar-behavior ring detector (signal 3), and how it combines with the identity graph

More sections will be added here as they're built: the hybrid score that combines all three signals, and the case-queue product an analyst actually uses.

## Status

_Last updated: 2026-08-25._

| Piece | Status |
|---|---|
| Dataset understanding (EDA) | Done |
| Transaction classifier | **Done and locked** — XGBoost, cost-based threshold, SHAP explainability |
| Identity graph | **Built** — 803,453 edges, 41.1% of transactions connected |
| Behavioral graph | **Built and combined with identity graph, then clustered** — 265,011 Louvain clusters; not yet evaluated against fraud labels (that's Day 7) |
| Hybrid scoring | Not started |
| Case-queue product | Not started |
