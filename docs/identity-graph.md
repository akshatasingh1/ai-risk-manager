# Identity Graph (Signal 2: "sloppy" rings)

## What it does

Links transactions that share a card, billing address, or email domain — catching fraud **rings**: groups of accounts that share infrastructure, which a single-transaction classifier can't see because it only ever looks at one transaction at a time.

**`isFraud` is never used to build this graph.** It's built purely from shared attributes; whether the resulting structure actually concentrates real fraud is checked afterward (Day 7), using the label only to *evaluate* the already-built graph, never to build it. This is what makes that later evaluation genuine rather than circular.

## Which attributes actually carry identity signal

The obvious approach is "link on any shared card/address field" — but not every field is equally meaningful. Before building anything, each candidate column's value distribution was checked:

| Field | Distinct values | Most common value covers | Verdict |
|---|---|---|---|
| `card1` | 13,553 | 2.5% of transactions | **Used** — enough distinct values to plausibly mean "the same card" |
| `card2` | 500 | 8.3% | **Used** |
| `addr1` | 332 | 7.8% | **Used** |
| `P_emaildomain` | 59 | 38.7% (mostly Gmail/Yahoo/Hotmail) | **Used**, with a cap (below) |
| `card3` | 114 | **88.3%** | Excluded |
| `card4` | 4 | 65.2% (card network, e.g. "Visa") | Excluded |
| `card5` | 119 | 50.2% | Excluded |
| `card6` | 4 | 74.5% (debit vs. credit) | Excluded |
| `addr2` | 74 | **88.2%** | Excluded |

`card3`, `card4`, `card5`, `card6`, and `addr2` aren't missing data (they're 99–100% populated) — they're simply too coarse. Sharing "Visa" or "credit card" with someone tells you nothing about being the same person; it just means two unrelated people both used a Visa card. Building edges from them would connect huge swaths of the dataset with zero real meaning. This is a refinement on the plan's original "card1–card6" shorthand, based on actually checking the data rather than assuming every card-related column is equally useful.

## How an edge is built

- **One node per transaction** (the dataset has no persistent account ID, so a transaction is the practical unit — the same approach used in published graph-based fraud research on this dataset).
- **An edge connects two transactions if they share a `card1`, `card2`, `addr1`, or `P_emaildomain` value *and* fall within 24 hours of each other.** The time window matters: real rings operate in bursts, not spread evenly across the dataset's full 6-month span.
- **Edge weight = 1 / (how many transactions share that value).** A rare shared email domain counts as much stronger evidence than a common one. A pair sharing multiple attributes (e.g. same card *and* same email) gets a summed weight — more shared evidence, stronger edge.
- **A scale guard skips attribute values shared by more than 1,000 transactions** (e.g. "gmail.com"). Building edges for something that common would be computationally wasteful and contribute a near-zero weight anyway, so it's skipped outright rather than processed and discarded.

Built attribute-indexed (grouped by value first, connections only made within each group) rather than comparing all 590K×590K possible transaction pairs, which would be computationally infeasible.

## Results

| Metric | Value |
|---|---|
| Nodes (transactions) | 590,540 |
| Edges | 803,453 |
| Transactions connected to at least one other | 242,644 (41.1%) |
| Build time | 4.7 seconds |
| Connected components (size > 1) | 37,813 |
| Median non-trivial component size | 3 |
| Largest component size | 49,294 |

The median group size of 3 is a healthy sign — most connected clusters are small, consistent with genuine small-scale sharing (e.g. a household using the same card). But the *largest* component — 49,294 transactions, 8.3% of the entire dataset — is not a real fraud ring; it's an artifact of transitive chaining: transaction A links to B via a shared card, B links to C via a shared email, C links to D via a shared address, and so on. Each individual link is meaningful, but the resulting chain can sprawl across thousands of loosely related transactions that don't actually belong together.

**This is expected, not a bug, and it's exactly why Day 6 uses Louvain community detection rather than treating raw connected components as the rings.** Louvain looks for *densely* connected sub-groups within a component, using edge weights, rather than just "is there any path at all between these two nodes." The connected-component numbers above describe the raw graph; the actual candidate rings come from clustering it, next.

## What's next

- Day 6: build the behavioral-similarity graph (signal 3), combine it with this identity graph, and run Louvain clustering on the combined graph
- Day 7: evaluate the resulting clusters against `isFraud` for the first time — fraud lift, capture rate, cluster precision
