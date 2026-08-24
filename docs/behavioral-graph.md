# Behavioral-Similarity Graph (Signal 3: "careful" rings)

## What it does

Links transactions that behave suspiciously alike — similar amounts, similar timing, similar velocity — **even when they share no card, address, or email at all**. This catches rings that are careful enough to rotate their identifiers per account specifically to dodge the identity graph, but can't as easily fake matching operational behavior every time.

**`isFraud` is still not used to build this.** Same discipline as the identity graph — built blind, evaluated against labels only afterward (Day 7).

## How an edge is built

- **Features**: `TransactionAmt` (log-transformed) plus `C1–C14` (Vesta's identity-count features) and `D1–D15` (time-delta/velocity features) — all already in the dataset, purpose-built by Vesta partly to capture exactly this kind of pattern (see [Dataset](dataset.md)).
- **Time bucketing**: transactions are grouped into 15-minute buckets, and similarity is only computed *within* a bucket. Comparing all 590K×590K possible pairs directly is computationally infeasible, and "suspiciously similar" is only meaningful for a burst of activity close together in time anyway.
- **Similarity metric**: cosine similarity between standardized feature vectors.
- **Threshold, chosen from the data, not guessed**: a sample of 170,526 same-bucket transaction pairs was checked first. The *median* pair has near-zero similarity (0.06) — most transactions that happen to land in the same time window look nothing alike. Only the top ~1% exceed 0.99. The threshold used is **0.98**, just above the 99th percentile — selective enough to mean something.
- **Weight = similarity × 0.2.** Similarity is a *suspicion*, not proof, unlike a shared identifier — deliberately weighted lower so it counts for less than identity evidence once combined.

## Does this actually add anything beyond the identity graph?

Checked structurally, before any label is involved:

| Check | Result |
|---|---|
| Behavioral edges with no matching identity-graph edge | **270,451 / 287,183 (94.2%)** |
| Transactions connected only through behavior (isolated in the identity graph) | **123,336** |

94% of what this layer finds, the identity graph would have missed entirely. That's the structural case for why this is a genuinely separate signal, not a redundant one — whether it's a *useful* one (i.e. whether these behaviorally-linked transactions are actually more likely to be fraud) is what Day 7 checks.

## Combining and clustering

The identity graph (Day 5) and this behavioral graph are unioned into one combined graph — where a pair of transactions has an edge in both, the weights sum, so a pair with both a shared card *and* matching behavior ends up more strongly connected than either alone. [Louvain community detection](https://en.wikipedia.org/wiki/Louvain_method) — a standard, well-benchmarked algorithm, not a custom one, run via `networkx`'s built-in implementation (avoids adding the separate `python-louvain` package for the same algorithm) — is then run on the combined graph to find the actual candidate rings.

Louvain matters here specifically because the raw identity graph (Day 5) had one enormous 49,294-transaction connected component that was mostly a chaining artifact, not one real ring. Clustering fixes that:

| Metric | Identity graph alone (Day 5) | Combined graph, after Louvain |
|---|---|---|
| Largest group size | 49,294 (raw connected component) | **4,187** (Louvain cluster) |
| Median non-trivial group size | 3 | 2 |
| Non-trivial groups | 37,813 (connected components) | 40,451 (clusters) |

The largest group shrank by over 10x — Louvain successfully broke the loosely-chained giant component into tighter, more plausible sub-communities.

## Results

| Metric | Value |
|---|---|
| Behavioral graph: nodes / edges | 590,540 / 287,183 |
| Behavioral graph: non-isolated transactions | 214,559 (36.3%) |
| Combined graph: edges | 1,073,904 |
| Total Louvain clusters | 265,011 |
| Non-trivial clusters (size > 1) | 40,451 |
| Behavioral graph build time | 13.0 seconds |
| Louvain clustering time | 61.1 seconds |

## What's next

- Day 7: the first time `isFraud` is used — fraud-rate scoring per cluster, capture rate, cluster-level precision, and specifically how many of these behavioral-only-connected clusters turn out to concentrate real fraud
