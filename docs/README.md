# Documentation

This folder explains **what the AI Risk Manager is and why it's built this way**, organized by concept.

## Read in this order

1. [Problem & Approach](problem-and-approach.md) — the problem we're solving, and why it's three signals instead of one
2. [Dataset](dataset.md) — what data powers the system, and its known limitations
3. [Classifier](classifier.md) — the transaction-level fraud detector (signal 1)

More sections will be added here as they're built: the identity graph and behavioral graph (signals 2 and 3), the hybrid score that combines all three, and the case-queue product an analyst actually uses.

## Status

_Last updated: 2026-08-24._

| Piece | Status |
|---|---|
| Dataset understanding (EDA) | Done |
| Transaction classifier | Baseline built (Logistic Regression); improved model (XGBoost + imbalance handling) in progress |
| Identity graph | Not started |
| Behavioral graph | Not started |
| Hybrid scoring | Not started |
| Case-queue product | Not started |
