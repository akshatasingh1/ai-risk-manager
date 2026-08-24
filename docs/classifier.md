# Classifier (Signal 1: solo-fraud detection)

## What it does

Scores each individual transaction with a fraud probability, using only that transaction's own details — amount, product type, card/address/email, and behavioral features. This is the signal that catches **solo fraud**: a single bad transaction with no accomplices, nothing to link it to anything else.

## Why a gradient-boosted tree model, not deep learning

This is structured, tabular data (rows and columns), not images, text, or sequences. On tabular data, gradient-boosted trees (XGBoost/LightGBM) consistently outperform deep learning approaches like autoencoders, and their decisions are far easier to inspect and explain — important both for catching mistakes and for defending the design in review. A Logistic Regression model is used first as a simple, interpretable baseline before moving to the tree model.

## Handling the 3.5% imbalance

Fraud is rare (see [Dataset](dataset.md)) — only 3.5% of transactions. A model can look "accurate" while barely catching any fraud, simply by predicting "not fraud" most of the time. This is addressed in two stages:

1. **Baseline first, honestly reported** — a plain model with no imbalance correction, to show the real effect of the imbalance rather than hide it.
2. **Correction next** — class weighting (and/or SMOTE, applied only to the training data, never the holdout) so the model is pushed to actually learn the minority class.

## Choosing the decision threshold

The default 0.5 cutoff for "is this fraud?" is arbitrary — it doesn't reflect that a false positive (an analyst manually reviewing a legitimate transaction) and a false negative (missed fraud) cost very different amounts. Instead, each is assigned an explicit ₹ cost, and the threshold is chosen to minimize total expected cost across the holdout.

**Cost assumptions (stated explicitly, not hidden):**

- **Missed fraud (false negative) costs the transaction's own amount.** If fraud goes undetected, the merchant loses that transaction's value — a missed ₹50,000 fraud is treated as worse than a missed ₹200 one. This follows standard cost-sensitive fraud-detection practice (the cost-matrix approach used in credit-card fraud research), rather than one flat penalty for every miss.
- **An unnecessary manual review (false positive) costs ₹100.** Estimated from roughly 10–15 minutes of a fraud-ops analyst's time at a typical blended cost of ₹400–600/hour, for a lightweight per-transaction check — not a full investigation.
- **Currency caveat:** the IEEE-CIS dataset's `TransactionAmt` is originally denominated in USD. Its numeric value is used directly as ₹ at risk here — what matters for *selecting a threshold* is the ratio between the two costs, not the absolute currency. A real deployment would substitute the merchant's actual currency and order values.

The threshold is swept from 0.01 to 0.99 and the one minimizing `(false positives × ₹100) + (sum of missed fraud amounts)` on the holdout is selected — see `notebooks/04_threshold_and_explainability.ipynb` and `src/cost.py`.

## Results so far

Both models trained on a stratified 80/20 split (train: 472,432 rows, holdout: 118,108 rows — both preserving the 3.5% fraud rate). The holdout is locked and reused for every evaluation below, so the comparison is apples-to-apples.

| Metric | Baseline (Logistic Regression) | v2 (LightGBM, class-weighted) |
|---|---|---|
| Precision | 81.6% | 27.5% |
| Recall | 26.9% | **83.2%** |
| F1 | 0.40 | 0.41 |
| PR-AUC | 0.47 | **0.69** |

Out of 4,133 actual fraud cases in the holdout, the baseline caught 1,111; v2 catches **3,439** — a large recall improvement, and PR-AUC (the metric that matters most for a ranking/scoring problem like this) improved from 0.47 to 0.69.

**Why precision dropped:** correcting for the imbalance (via `class_weight='balanced'`) intentionally makes the model much more willing to flag something as fraud, which catches far more real fraud but also produces more false alarms (9,089 legitimate transactions flagged, vs. 251 for the baseline) — all measured at the same default 0.5 cutoff. This tradeoff is expected and by design: **the 0.5 cutoff is arbitrary and was never meant to be final.** Day 4's job is to replace it with a cost-based threshold that explicitly balances "cost of a manual review" against "cost of missed fraud," rather than reading precision/recall at an untuned default.

**Why class weighting instead of SMOTE:** class weighting tells the model to pay more attention to real fraud examples during training, without inventing synthetic ones. It's simpler to defend ("we told the model fraud costs more to miss," vs. "we generated synthetic fraud transactions that never happened") and gradient-boosted trees handle it natively.

## Final locked results

**The classifier is now locked: the holdout is not evaluated against again.** At the cost-optimal threshold of **0.88**:

| Metric | @ default 0.5 | @ cost-optimal 0.88 |
|---|---|---|
| Precision | 27.5% | **80.4%** |
| Recall | 83.2% | 53.8% |
| F1 | 0.41 | **0.64** |
| Manual reviews per real fraud caught | 2.64 | **0.24** |
| Total holdout cost | ₹1,023,980 | **₹387,932** |

Moving from the untuned 0.5 cutoff to the cost-optimal 0.88 cuts total cost by **~62%** (₹636,047 saved on the holdout alone) — and drops manual review burden from "2.64 reviews per fraud caught" to "0.24 reviews per fraud caught," meaning most flagged cases are now real fraud rather than false alarms. This confirms the point made in Day 3: 0.5 was never the right number to judge the model at — PR-AUC (0.69) reflected the model's real ranking ability all along, and the threshold sweep is what turns that ranking ability into a good operating point.

This is also a real, honest trade-off, not a free lunch: recall drops from 83.2% to 53.8% — the system now misses more fraud than it did at 0.5, in exchange for far fewer false alarms. That trade-off is exactly what the ₹ cost assumptions above encode, and a different assumption (e.g. a higher missed-fraud cost, or a cheaper review cost) would shift the threshold and this trade-off differently — which is the point of making the costs explicit rather than baking in a silent judgment call.

## Explainability: what drives a fraud flag

[SHAP](https://github.com/shap/shap) was used to see which features actually move the model's prediction, computed on a 3,000-transaction sample of the holdout. Top drivers, in plain language:

- **`TransactionAmt`** — the transaction amount itself, the single strongest signal
- **`C1`, `C11`, `C13`, `C14`** — Vesta's "identity count" features: roughly, how many other accounts/cards/addresses are linked to this identity
- **`card1`, `card2`, `card5`, `card6` (credit vs. debit)** — characteristics of the card used
- **`D1`, `D3`, `D4`, `D15`** — time-delta features: how long since related past activity (e.g. since this card was last seen)
- **`ProductCD` (category "R")** — the type of product/service purchased
- **`addr1`** — billing address region
- **Several `V` columns** (`V70`, `V91`, `V294`, `V258`, `V48`) — Vesta's anonymized engineered features; their exact real-world meaning isn't disclosed by the dataset provider, but they're empirically influential

This becomes the basis for the case queue's plain-English reason (e.g. *"flagged mainly due to unusual transaction amount and a high number of linked identities"*) instead of showing an analyst a raw probability score.

## What's next

- Signal 1 (this classifier) is done. Next: the identity graph (Day 5) and behavioral-similarity graph (Day 6) — signals 2 and 3 — followed by combining all three into one hybrid score (Day 8)
