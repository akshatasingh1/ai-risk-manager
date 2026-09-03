# Rule-Based Flags (alongside the ML score, not blended into it)

## What this is

Three simple, deterministic, explainable rules that run **next to** the classifier and graph signals — not fed into them. This follows how real payment-fraud systems actually combine rules and ML: as independent, parallel signals that both feed a final decision, rather than one blended number. Each rule is cheap to compute, easy to explain to an analyst, and easy to update without retraining anything.

**Built with the same discipline as the graph work**: the rules are computed without ever looking at `isFraud`. The label is only used afterward, to check whether these blind rules actually catch real fraud.

## The three rules

All three are computed **causally** — each transaction is only ever compared against transactions that happened *before* it (by `TransactionDT`), for the same card (`card1`+`card2`). This matters: a rule that "peeks" at later transactions relative to the one being scored could never actually fire that way in production, since the future hasn't happened yet at scoring time.

1. **High card velocity** — 3 or more transactions from the same card in the preceding hour.
2. **New card, high value** — this is the card's first-ever transaction in the data, and the amount is at or above the 90th percentile (₹275.99, computed from train-only data, same discipline as every other threshold in this project).
3. **Amount spike** — the transaction amount is more than 5x this card's own median amount from its prior transactions. Never fires on a card's first transaction (no history to compare against yet).

## Results

| Rule | Transactions flagged | Fraud rate among flagged | Lift vs. 3.5% baseline | % of all fraud captured |
|---|---|---|---|---|
| High card velocity | 87,176 | 3.84% | 1.10x | 16.2% |
| New card, high value | 1,460 | 4.59% | 1.31x | 0.3% |
| **Amount spike** | 30,670 | **4.71%** | **1.35x** | 7.0% |
| Any rule triggered | 114,457 | 4.06% | 1.16x | 22.5% |

Modest individually — similar order of magnitude to the identity/behavioral graph lifts (1.18x–1.50x), not a standalone detector. `amount_spike` is the strongest single rule; `high_velocity` is the weakest (fires on 14.8% of the entire dataset, too broad to be very selective on its own).

## Does this add coverage beyond the classifier and graph?

Checked against the locked v3.1 classifier's holdout misses: of the **1,481 real fraud cases v3.1 misses**, **324 (21.9%) trigger at least one rule flag**. Real, independent additional coverage — comparable in scale to the graph-based "second look" mode's coverage of classifier misses (31% at floor 0.50) — reached through a completely different mechanism (deterministic thresholds, not learned patterns).

## Why these stay separate from the classifier, not fed in as features

Unlike the graph-structural features (Attempt 6, `docs/hybrid-merge.md`), these rules are deliberately **not** added as classifier inputs. Two reasons:

1. **This is how the industry actually does it.** The research is explicit that rule-based and ML components are kept as separate contributors to a final decision, not merged into one score — partly because rules need to stay individually auditable and instantly updatable (a compliance/support person can understand and change "flag if 3+ transactions in an hour" without retraining a model) in a way a blended feature can't.
2. **It fits the case-queue product directly.** Each rule already comes with a ready-made plain-English reason (*"3+ transactions from this card in the last hour"*) — exactly what an analyst needs to see, with no translation step required, unlike a raw model score or SHAP value.

## What's next

**Done, Day 10**: these flags are surfaced directly in the case queue, folded into each case's plain-English reason (`src/case_reason.py`) as a labeled "Rule flags: ..." clause alongside the classifier-driven and ring-evidence parts of the sentence — not merged into the score itself. See [Case Queue & API](case-queue-and-api.md).
