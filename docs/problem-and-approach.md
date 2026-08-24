# Problem & Approach

## The problem

Online merchants lose money to fraud. Someone reviews suspicious transactions today, but usually one at a time, in a spreadsheet, with no context on *why* a transaction looks risky and no memory of past decisions. That's slow, inconsistent, and easy for fraud to slip through.

## Why one detector isn't enough

There isn't just one kind of fraud:

- **Solo fraud** — a single stolen card used once. A transaction-level classifier that has seen thousands of fraud examples can catch this from the transaction's own details (amount, product, timing).
- **"Sloppy" rings** — a group of fraudulent accounts that share infrastructure, like the same card number, address, or email. These accounts are *linked* to each other, which a single-transaction classifier can't see, but a network of shared connections can.
- **"Careful" rings** — a group that knows sharing infrastructure gets caught, so every account uses a different card, address, and email. But if they still behave the same way — same amounts, same timing patterns, same velocity of transactions — that behavioral fingerprint gives them away even when nothing else matches.

A system built for only one of these misses the other two. So instead of picking one, this project combines three complementary signals into a single hybrid risk score:

1. **Transaction classifier** — scores each transaction individually (catches solo fraud)
2. **Identity graph** — links accounts that share a card, address, or email (catches sloppy rings)
3. **Behavioral-similarity graph** — links accounts with suspiciously similar behavior even when nothing is shared (catches careful rings)

All three feed into one score. This is one detector with three inputs, not three separate projects — and it produces the project's strongest finding: fraud that the graph layers catch which the classifier alone scored as low-risk.

## Who actually uses this

The model is the engine, not the product. The product is what a human does with its output.

Picture a risk analyst — call her Priya — whose job today is reviewing flagged transactions one at a time with no context. What she needs isn't a chart of precision and recall; she needs a **queue of cases**, each with:

- A plain-English reason it was flagged (*"this account shares a card with 3 others, one of which was confirmed fraud"* — not a raw model score)
- An action to take: Approve, Hold for review, Decline, or Escalate
- A record of her own past decisions, so the tool visibly earns her trust over time
- A way for her decisions to feed back into the system, so it improves from her judgment rather than just logging it

That queue — not a metrics dashboard — is the actual deliverable. The demo is Priya opening her queue, seeing a solo-fraud case, a ring case, and a behavioral-ring case, and resolving all three.

## What this system deliberately does not do

- **It does not claim to catch everything.** A ring that changes both its identifiers *and* its behavior at the same time can still slip past both graph layers. This gap is stated openly rather than hidden.
- **It is strictly defense-only.** Nothing it outputs explains how to evade detection — no exposed thresholds or rules that would teach someone how to get around it.
- **It does not pretend to be real-time when it isn't.** Any "live" demo replays historical data through the system as a simulated stream — this is stated honestly, not implied to be a production live feed.
