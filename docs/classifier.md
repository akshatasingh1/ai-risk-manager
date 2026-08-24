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

The default 0.5 cutoff for "is this fraud?" is arbitrary — it doesn't reflect that a false positive (an analyst manually reviewing a legitimate transaction) and a false negative (missed fraud) cost very different amounts. The plan is to assign each an explicit ₹ cost and pick the threshold that minimizes total expected cost, rather than using 0.5 by default. This is a design choice still to be finalized.

## Results so far

**Current status: baseline only.** Trained on a stratified 80/20 split (train: 472,432 rows, holdout: 118,108 rows — both preserving the 3.5% fraud rate). The holdout is locked and reused for every later evaluation, so results stay comparable across model versions.

| Metric | Value |
|---|---|
| Precision | 81.6% |
| Recall | 26.9% |
| F1 | 0.40 |
| PR-AUC | 0.47 |

Out of 4,133 actual fraud cases in the holdout, this baseline catches only 1,111 and misses 3,022. Precision is reasonably good — when it does flag something as fraud, it's usually right — but recall is weak, exactly as expected from an imbalanced dataset with no correction applied yet. This number is the "before" picture; the next step (an improved model with imbalance handling) is expected to raise recall substantially, and that comparison will be reported honestly here once available.

## What's next

- Replace/compare against XGBoost with class-weight or SMOTE correction
- SHAP-based feature importance, so "why was this flagged" can be explained in plain language for the case-queue product
- Cost-based threshold selection, replacing the default 0.5 cutoff used above
