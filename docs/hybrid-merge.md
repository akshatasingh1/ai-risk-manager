# Hybrid Merge (combining all three signals)

## What this is

The classifier (signal 1) and the two graph layers (signals 2 & 3) were built and evaluated separately. This is where they come together — and where the project's central bet gets tested honestly: does the graph catch real fraud the classifier misses, in a way that's actually worth acting on?

Six combination strategies were tried, in increasing sophistication. The first didn't work, and that's reported plainly rather than hidden. The second works, but not in the way you might expect — it isn't a ₹-cost win, and that's reported plainly too. The third (retraining with a flat graph-category feature) is a small, likely-noise-level win. The fourth (a Graph Neural Network on its own) doesn't beat the classifier either, even after real tuning investment. The fifth (feeding the classifier the GNN's actual learned embeddings) confirms embeddings carry far more signal than a flat label — but still loses on real ₹ cost, for a specific, explainable reason. The sixth (adding graph degree/PageRank alongside the category label) is the best result of all six, and is the version adopted as the primary classifier. All six are reported honestly rather than only keeping the ones that worked.

## The leakage fix

Day 7's category lift weights (`identity_only` 0.57x, `behavioral_only` 1.50x, `both` 1.21x) were computed using all 590,540 transactions' labels — including the 118,108-row holdout. Reusing them directly to score the holdout would be a soft leakage: the holdout's own labels partly shaped the weights being used to score it, the exact thing this project has never allowed for the classifier itself (its holdout has been locked since Day 4).

**Fix**: the category weights are recomputed here using only the 472,432-row train split, then applied to score the holdout.

| Category | Train-only weight | Day 7 full-dataset weight |
|---|---|---|
| `behavioral_only` | 1.51x | 1.50x |
| `both` | 1.21x | 1.21x |
| `identity_only` | 0.575x | 0.571x |

The two barely differ — reassuring, since it means the leakage fix was a correctness fix, not one that changes the underlying finding.

## Attempt 1 (rejected): rescale every score, then re-optimize one threshold

The obvious approach: `hybrid_score = classifier_proba × category_weight`, then sweep for the cost-minimizing threshold, exactly like Day 4 did for the classifier alone.

**Result: this loses money.** At its own best threshold (0.95), the hybrid score costs **₹392,838** — worse than the classifier alone (₹337,421). A **₹55,417 net loss**.

**Why**: `behavioral_only` and `both` together cover **56.1% of the holdout** — not a rare, selective subset, but over half the dataset. Multiplying that much of the population's score by 1.2–1.5x doesn't act as a precise "boost the risky ones" signal — it acts as a broad recalibration. Since most of what gets boosted is still legitimate (3.5% population fraud rate), the boost mostly creates new false positives across a huge swath of ordinary transactions. The cost-optimizer compensates by pushing the threshold from 0.83 up to 0.95 — which then costs real true positives among the untouched "isolated"/"neither" majority, whose scores never moved. Net: more reviews, less catch.

## Attempt 2 (adopted): an opt-in "second look," not a global rescale

Instead of touching every score, only ever *add* extra manual reviews — never remove one of the classifier's own flags. A transaction gets a "second look" if:

1. The classifier already rejected it (score < 0.83), **and**
2. It scored at least some minimum floor (the "sensitivity dial" — see below), **and**
3. Its cluster carries `behavioral_only` or `both` evidence — the two categories shown (twice now, on the full dataset and on train-only data) to genuinely concentrate fraud.

Because this only ever adds flags to the classifier's own decisions, **it cannot cause a regression** on cases the classifier already got right — no case-by-case tradeoff to audit, by construction.

### Why a floor is necessary

Without one, "classifier-missed + trusted category" is a huge population — 63,645 transactions in this holdout, with a median classifier score of just 0.07 (i.e., mostly transactions the classifier is very confident are legitimate). Reviewing all of them costs ₹6.36 million to recover ₹163,056 in fraud. The floor is what makes this a usable dial rather than a firehose.

### The sensitivity curve

| Floor | Extra reviews | Fraud caught | Precision | Lift vs. baseline | Net ₹ |
|---|---|---|---|---|---|
| 0.10 | 26,162 | 799 | 3.1% | 0.9x | -₹2,475,976 |
| 0.30 | 9,251 | 622 | 6.7% | 1.9x | -₹816,863 |
| **0.50** | **3,769** | **439** | **11.6%** | **3.3x** | -₹297,870 |
| 0.65 | 1,705 | 291 | 17.1% | 4.9x | -₹119,235 |
| 0.80 | 237 | 64 | 27.0% | 7.7x | -₹13,920 |

(Full curve at 0.05 increments in `results/hybrid_metrics.json`.)

### The honest read

**`net_rs` is negative at every single floor tested — including the most conservative one.** This is not a tuning failure; it's structural. The classifier's own 0.83 threshold was *already* set, in Day 4, at the point where expected fraud value ≈ the ₹100 review cost. Everything below that threshold is, by the classifier's own calibrated estimate, not worth a ₹100 review. For the graph signal to produce a net ₹ saving, it would need to find cases where that calibration is *wrong* by more than the fixed review cost — and while the fraud-rate lift is real (well above the 3.5% baseline at every floor, up to 7.7x at the strictest setting), it isn't large enough in dollar terms to clear that bar.

**This is presented as an optional, analyst-tunable higher-recall mode — not a cost-reducing default.** It maps directly onto the "sensitivity control" already scoped in the product design (see [Problem & Approach](problem-and-approach.md)): an analyst who'd rather trade some review budget for catching more fraud (e.g. compliance-driven, or simply risk-averse) can dial the floor down; one who wants to minimize review load can dial it up or leave the second look off entirely, running the classifier alone. At floor 0.50 (an illustrative middle setting, not a computed optimum — there isn't one to compute here, since every point is net-negative by ₹): **439 additional fraud cases caught (31% of the classifier's 1,408 misses), at a cost of 3,769 extra reviews, 11.6% precision, 3.3x the population baseline.**

## Attempt 3: feature fusion (retrain with graph features)

Both attempts above combine the classifier and the graph *after* the classifier is already trained and frozen (score-level fusion). Published approaches on this same dataset (see below) more commonly do **feature-level fusion** instead: add graph-derived columns (cluster category, cluster size) directly to the classifier's input and retrain, letting the model learn the right way to weigh them itself instead of a hand-written rule.

This is a genuinely different combination strategy from Attempts 1 and 2, so it's reported separately as its own experiment (`notebooks/08b_feature_fusion_classifier.ipynb`), producing a new classifier version (v3). **The Day 4 locked classifier result is unchanged and still stands as the project's official baseline** — v3 is an add-on comparison, not a replacement.

(Reference: a [Neo4j write-up on graph-based fraud detection using this exact IEEE-CIS dataset](https://neo4j.com/developer/industry-use-cases/finserv/retail-banking/ieee-cis-fraud-graphs/) describes building the same kind of identity graph and Louvain clustering used here, then feeding graph-derived features — community ID, centrality, embeddings — directly into an XGBoost classifier alongside transaction features, rather than combining scores after the fact.)

No new leakage risk here: unlike Day 8's category *weight* (a number derived from labels), cluster *category* is purely structural — it never looks at `isFraud` — so it's safe to use as an ordinary feature on train, holdout, or any future transaction with no train-only recomputation needed.

| Metric | v2 (locked, Day 4) | v3 (+ ring features) |
|---|---|---|
| Threshold | 0.83 | 0.87 |
| Precision | 75.8% | 83.5% |
| Recall | 65.9% | 61.4% |
| F1 | 0.71 | 0.71 |
| Total cost | ₹337,421 | **₹333,872** |

**v3 is marginally cheaper — ₹3,549 less (about 1%).** Small, and likely within the noise of retraining rather than a decisive win.

**Why so small?** Checking feature importance answers this directly: `ring_category` and `ring_cluster_size` rank **dead last** among all features the model was given — importance scores of 0.0006–0.0011, versus the top features (`C13`, `TransactionAmt`, etc.) scoring an order of magnitude higher. XGBoost essentially ignored them. The likely reason: the classifier already had access to `C1–C14` — Vesta's own identity-count/velocity features — which evidently already encode most of what the graph's structural category adds, for the purpose of scoring an individual transaction. The graph isn't wrong or useless; it's *redundant* with information already present in the existing feature set, at least for this specific per-transaction decision.

This actually clarifies something important about *where* the graph's real value lives: not in nudging the classifier's per-transaction score (which barely moves), but in the genuinely relational information a single row's features can never capture — "this specific transaction connects to these other specific transactions," which is exactly what the case-queue product needs to show an analyst (*"this account is linked to 4 others, one confirmed fraud"*), not something a feature-fusion classifier reduces to a single number well.

## Attempt 4: Graph Neural Network (GNN)

A fair question at this point: several published fraud-detection papers on this exact dataset report strong results using Graph Neural Networks (GNNs) — models trained end-to-end on the graph structure itself, rather than a classifier and a graph treated as separate pieces. Was that worth trying here?

**Yes — tried directly, in two stages, since the first attempt on its own would have been misleading.**

**Attempt 4a, naive**: a plain 2-layer GraphSAGE network, one fixed learning rate, no tuning, 15 epochs — the "just try it" version. Result: holdout PR-AUC 0.39, cost **₹579,357** — clearly worse than everything built so far. Taken alone, this could easily (and wrongly) be read as "GNNs don't work here."

**Attempt 4b, tuned**: the same underlying idea, with the actual effort a GNN needs to be fairly evaluated — the identity/behavioral edge weights (used, not discarded, via `GraphConv`), layer normalization for training stability, weight decay, a learning-rate scheduler, a short hyperparameter sweep, and 250 epochs of training with model selection on an internal validation split carved out of the *train* data (the real holdout was still touched only once, at the end — same discipline as everywhere else in this project). Result: holdout PR-AUC **0.64**, cost **₹429,607**.

| Model | Holdout PR-AUC | Cost |
|---|---|---|
| GNN, naive (Attempt 4a) | 0.39 | ₹579,357 |
| GNN, tuned (Attempt 4b) | 0.64 | ₹429,607 |
| XGBoost (locked, Day 4) | 0.76 | ₹337,421 |
| XGBoost + graph features (Attempt 3, v3) | — | ₹333,872 |

**Tuning closed most of the gap (PR-AUC 0.39 → 0.64, cost down 26%) — but not all of it.** Even the properly-tuned GNN is still **~27% more expensive than the classifier already built**. The honest conclusion: with the time actually available for this project, a GNN does not beat a well-tuned XGBoost model here. This is a real, deliberately time-boxed finding, not a dismissal — a GNN with substantially more investment (deeper architectures, attention-based layers, GPU-accelerated tuning over days rather than hours) might eventually close the remaining gap, but that's outside this project's scope and timeline.

One methodological note worth being upfront about: published papers reporting high raw *accuracy* on this dataset (e.g. "94.7% accuracy") are not directly comparable to the numbers above — with only 3.5% fraud in the data, a model that predicts "not fraud" for every transaction already scores 96.5% accuracy without learning anything. This project reports PR-AUC and ₹ cost specifically to avoid that trap (see [Classifier](classifier.md)).

See `notebooks/08c_gnn_experiment.ipynb` for the full code (both attempts) and `results/gnn_experiment_metrics.json` for the saved numbers. Given the clearly negative result, the notebook was not re-executed end-to-end with saved cell outputs after the numbers above were confirmed (CPU-only training takes ~25-30 minutes) — the code is exact and reproducible, but the time was spent on the core product instead of a polish pass with no new information.

## Attempt 5: real GNN embeddings, not a flat category label

Attempt 3's feature-fusion used a single 4-value category label (`isolated`/`identity_only`/`behavioral_only`/`both`) as the classifier's only graph-derived input — and it barely helped, since a flat label carries very little information. The GNN research (Section "Attempt 4") suggested the production-standard fix: feed the classifier the GNN's actual 128-dimensional learned embedding per transaction — a dense summary of that transaction and its graph neighborhood — instead of one coarse bucket.

Tried directly: the already-trained tuned GNN (Attempt 4b) was re-run once to also save its weights and extract a 128-number embedding for all 590,540 transactions (`notebooks/08d_gnn_embedding_fusion.ipynb`), which were then added as 128 ordinary numeric columns and the classifier retrained (v5).

**The embeddings clearly mattered to the model** — confirming the theory:

| Check | Result |
|---|---|
| Combined importance of all 128 embedding dims | **58.7%** of total feature importance |
| Rank of the single most important feature overall | **#1**, an embedding dimension |
| Embedding dims in the top 20 features | **15 of 20** |

For comparison, Attempt 3's flat category label had essentially zero importance and ranked last. This is a real, measured confirmation that a dense embedding carries far more signal than a coarse label — exactly what the research predicted.

**But real ₹ cost got *worse*, not better:**

| Model | Cost |
|---|---|
| v2 (locked, no graph features) | ₹337,421 |
| v3 (flat category label) | ₹333,872 |
| **v5 (real GNN embeddings)** | **₹374,903** |

**Why the model can lean heavily on something that hurts it — a known, explainable pitfall, not an execution mistake.** The GNN was trained *transductively*: during its own training, a training-set transaction's embedding was shaped directly by gradient descent against its own label. A holdout transaction's embedding never got that treatment — it only ever came from a forward pass through the trained network (structure + features), never from its own label. That makes training-row embeddings subtly more "label-informed" than holdout-row embeddings in a way the classifier can't tell apart. XGBoost, trained only on the training rows, learns to trust the embeddings more than it should — a trust that doesn't transfer cleanly to the honestly-generic embeddings the holdout rows actually have. This is a documented failure mode of using transductive GNN embeddings as downstream features, not something specific to this implementation; fixing it properly would need an *inductive* GNN setup (embeddings computed the same way for every node, train or not) — a larger redesign, out of scope here.

**Verdict: v5 is not adopted.** Worse than both v2 and v3 in real ₹ terms, despite the model visibly trusting the embeddings the most of anything it was given. Reported in full because the *mechanism* is a genuinely useful, correct finding — confirming the research's prediction about embeddings carrying more signal than a flat label, while also surfacing a real, well-known limitation of applying that idea naively with a transductively-trained GNN.

## Attempt 6: v3.1 — richer structural graph features

Attempt 3 (v3) used only the flat category label. Real production systems (per the Neo4j reference above) also add **degree centrality and PageRank**, not just a community/cluster ID — richer, continuous structural signals, still computed purely from graph structure (no leakage risk, same as `ring_category`, unlike the GNN embeddings in Attempt 5).

Added to v3's features: **graph degree** (how many direct connections a transaction has), **weighted degree** (sum of edge weights, distinguishing a rare-shared-card connection from a common one), and **PageRank** (how "central" a transaction is within its cluster's structure). Retrained as v3.1 (`notebooks/08e_v3_1_structural_features.ipynb`).

| Metric | v2 (locked) | v3 (flat label) | **v3.1 (+ degree/PageRank)** |
|---|---|---|---|
| Threshold | 0.83 | 0.87 | 0.84 |
| Precision | 75.8% | 83.5% | 79.7% |
| Recall | 65.9% | 61.4% | 64.2% |
| F1 | 0.705 | 0.707 | **0.711** |
| PR-AUC | 0.757 | 0.757 | 0.755 |
| Total cost | ₹337,421 | ₹333,872 | **₹330,948** |

**v3.1 is the cheapest of everything tried so far** — ₹6,473 (1.9%) less than v2, ₹2,924 less than v3. Worth being precise about what this is and isn't: PR-AUC is marginally *lower* than both v2 and v3 (0.755 vs 0.757), and the new structural features still rank near the bottom of feature importance (0.0006–0.0010, the same order of magnitude as the flat label). So this isn't the model discovering a powerful new signal — it's a real but small win, most likely from the extra features nudging where the cost-optimal threshold lands rather than genuinely better fraud-vs-non-fraud discrimination. The same honest caveat as v3 applies: on a different holdout sample, this ranking could plausibly shift.

**Adopted as the primary classifier**, given it's the best result across every version tried, is architecturally closer to how production systems (Stripe, PayPal, and the Neo4j reference) actually combine graph and tabular signals, and carries the same leakage-safety guarantee as v3 (no transductive-embedding risk, unlike v5). The margin over v2 is small enough that it should be treated as a genuine but modest improvement, not a transformative one — the honest story for the pitch is "we tried five other combination strategies first, several of which failed or underperformed, and this is the one that survived scrutiny."

## Example: a fraud case only the second look catches

At floor 0.50, several real fraud transactions the classifier scored well below 0.83 — and would otherwise have missed entirely — are caught because their cluster shows genuine behavioral similarity to other transactions, at least one of which is confirmed fraud. See `results/hybrid_metrics.json` and `notebooks/08_hybrid_merge.ipynb` for specific `TransactionID`s and their cluster context.

## Honest limitations

- **Category-level, not cluster-level, granularity.** Every transaction in a `behavioral_only` cluster is treated identically, regardless of that specific cluster's actual size or edge strength. This is a deliberate choice for statistical stability (an individual cluster's own fraud rate is a much noisier, more leakage-prone estimate than a category-wide aggregate — see [Ring Evaluation](ring-evaluation.md)), at the cost of nuance a finer-grained model could capture.
- **No single "optimal" second-look floor** — every point on the curve costs more than it saves in ₹ terms. The floor is a policy choice about acceptable review burden, not a number this analysis can pick for you.
- **Both attempts, and the classifier itself, tune/evaluate on the same holdout.** Consistent with the precedent set since Day 4, but worth naming: none of these thresholds come from a genuinely independent third split.
- **Currency caveat inherited from the classifier** (see [Classifier](classifier.md)): `TransactionAmt` is treated as ₹ at risk for this exercise; the original dataset denominates it in USD.
- **The residual gap stands**: none of this catches a ring that rotates identifiers *and* deliberately varies its behavior at the same time — see [Problem & Approach](problem-and-approach.md).

## What's next

**Done, Days 9–11**: the FastAPI service exposes this second-look floor as a live parameter on `/batch-score`, and the Streamlit case queue turns it into a sensitivity slider an analyst actually operates — with the real ₹ trade-off from the table above shown next to it, not hidden behind a single score. See [Case Queue & API](case-queue-and-api.md).
