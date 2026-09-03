# Ring Evaluation (Signals 2 & 3, checked against real fraud for the first time)

## What this is

Days 5 and 6 built the identity graph and behavioral graph **without ever looking at `isFraud`**. This is the step where that discipline pays off: the label is used, for the first time, purely to check whether the blind clustering actually found anything real — never to build it. See [Identity Graph](identity-graph.md) for why that separation matters.

## Headline result: does clustering find more fraud than chance?

"Flagged" = every transaction inside a non-trivial cluster (size ≥ 2) is treated as a positive prediction.

| Metric | Value |
|---|---|
| Clusters flagged | 40,451 |
| Transactions flagged | 365,980 (62.0% of the dataset) |
| Fraud captured | 15,151 |
| Precision (fraud % within flagged clusters) | 4.14% |
| **Fraud lift vs. population baseline (3.5%)** | **1.18x** |
| Capture rate (% of all fraud found) | 73.3% |

Honest read: **this is a real signal, but a modest one on its own.** A 1.18x lift means flagged clusters are only somewhat more fraud-concentrated than random chance — nowhere near as sharp a signal as the classifier's own precision/recall. The high capture rate (73.3%) is partly an artifact of the flagged set being enormous (62% of the whole dataset) — casting that wide a net will always catch most fraud, just with low precision. This is exactly why the graph layers were never meant to stand alone: they're built to catch what the classifier misses, as a *second opinion*, not to replace it. That comparison is Day 8's job.

## Sensitivity check: does this hold at different ring-size cutoffs?

| Min cluster size | Clusters flagged | Precision | Lift | Capture rate |
|---|---|---|---|---|
| ≥ 2 | 40,451 | 4.14% | 1.18x | 73.3% |
| ≥ 3 | 17,499 | 4.18% | 1.19x | 64.7% |
| ≥ 5 | 6,629 | 4.21% | 1.20x | 57.9% |

The lift stays consistently around 1.18–1.20x regardless of the cutoff — the finding isn't a fluke of one arbitrary threshold. As expected, a stricter size cutoff trades capture rate for a slightly higher lift (fewer, larger clusters are marginally more concentrated).

## Does the behavioral layer earn its place? Yes — and more than the identity layer does alone

This is the most important result of Day 7. Every non-trivial cluster was classified by what kind of edges hold it together:

| Cluster type | Clusters | Fraud rate | Lift vs. population |
|---|---|---|---|
| **Behavioral-only** (no shared identifier at all) | 18,485 | **5.26%** | **1.50x** |
| Both (shared identifier *and* similar behavior) | 8,854 | 4.22% | 1.21x |
| **Identity-only** (shared identifier, no behavioral match) | 13,112 | 1.998% | **0.57x** |

This is a genuinely surprising, counter-intuitive finding worth stating plainly: **clusters held together only by a shared identifier (card/address/email) are actually *less* fraud-concentrated than random chance (0.57x)**, while clusters held together only by behavioral similarity — with no shared identifier at all — are the strongest signal of the three categories (1.50x). This runs against the original assumption that a shared identifier is inherently stronger evidence than behavioral similarity. The likely explanation, confirmed by the manual audit below: many "identity-only" clusters are large chaining artifacts (see [Identity Graph](identity-graph.md#results)) rather than real rings, while the transactions that survive as behaviorally-linked with *no* shared identifier are a much more deliberately-filtered, meaningful group.

**18,485 clusters were found only through behavioral similarity — and they concentrate real fraud at 1.5x the population rate.** That's the number that justifies signal 3: it isn't just structurally independent of the identity graph (Day 6's finding), it's also more predictive on its own.

## Manual false-positive audit

36,866 of the 40,451 non-trivial clusters (91.1%) contain **zero** fraud. The five largest of these were inspected by hand rather than assumed to be a system failure.

**What was actually found:** the large zero-fraud clusters are not coincidental groupings — they're the transitive-chaining artifact already flagged in [Identity Graph](identity-graph.md#results). For example, a 411-transaction cluster turned out to be linked through a moderately common `card1` value (7638) that reappears with a *different* address and email domain every few weeks, chaining together dozens of otherwise-unrelated purchases across a span of many months — not a burst, not a ring, just the same card slowly showing up in the dataset over and over. The identity graph's edges are correctly time-windowed *pairwise* (any two linked transactions are within 24 hours of each other), but a long chain of such pairs — A↔B, B↔C, C↔D — can still span months once assembled into one connected structure.

**This is the concrete mechanism behind the identity-only category's weak 0.57x lift above.** It's a genuine limitation of attribute-indexed identity graphs, not unique to this project, and stating it plainly here is more useful than a vague "some false positives exist" — it also directly motivates weighting cluster categories differently (behavioral-containing clusters are more trustworthy) rather than treating every non-trivial cluster identically, going into Day 8.

## What this changes for Day 8 (hybrid merge)

Treating every non-trivial cluster as equally "flagged" — as the headline number above does — isn't the right rule for production use, given how differently the three cluster categories actually perform. The hybrid score should weight a cluster's contribution by its composition (behavioral-containing clusters carry more real signal than identity-only ones), not by cluster membership alone.

## What's next

- Day 8: combine classifier + ring scores into one hybrid score, and find the headline finding — fraud caught by the graph layers that the classifier scored as low-risk
