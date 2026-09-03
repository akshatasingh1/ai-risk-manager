# Case Queue & API (Days 9–11)

## What this is

Everything through Day 8 is the engine — the classifier, the two graphs, the hybrid merge, the rules. This is where it becomes the actual product: a FastAPI service that serves the locked v3.1 model, and a Streamlit case queue Priya can click through, per the persona in [Problem & Approach](problem-and-approach.md).

## The serving layer (`src/serving.py`)

`ScoringContext` loads every artifact the model needs exactly once at startup — the v3.1 pipeline, the three graphs, the cluster partition, the category weights, the SHAP explainer — and reuses them across every request. The one thing worth understanding here: `score_batch()` doesn't reimplement feature engineering, it *reproduces* the exact pipeline `notebooks/08e_v3_1_structural_features.ipynb` trained on (graph degree/weighted-degree/PageRank, ring category, rule flags), so a live request and a training-time row get identical treatment. This was checked, not assumed: `score_batch()` output was diffed against the notebook's own computation on 200 real holdout rows and matched to `0.0` — no drift.

### A real bug this parity check caught

`/score` (one transaction) and `/batch-score` (many) initially returned **different scores for the identical transaction**. Root cause: a DataFrame built from a single JSON payload infers `object` dtype (holding Python `None`) for any column whose only value is missing, instead of a proper `float64` NaN or missing-string — and SimpleImputer/OneHotEncoder handle a raw `None` differently than they handle a real NaN. A DataFrame built from many rows never hits this, since pandas infers the real dtype from hundreds of values in the same column — so the bug was invisible until the API scored one transaction at a time, which is exactly `/score`'s job. Fixed with explicit numeric coercion (`pd.to_numeric`) and categorical None-normalization before scoring; locked in with a regression test (`tests/test_serving.py`).

## Plain-English case reasons (`src/case_reason.py`)

Section 2a's product framing calls for *"a plain-English reason, not a raw SHAP plot."* This module builds that sentence from three sources per case:

1. **SHAP** (`shap.TreeExplainer`, built once at startup) — the top features actually pushing this specific prediction toward fraud, mapped to the same plain-language categories already documented in `docs/classifier.md` (e.g. `C1–C14` → "a high number of linked identities/cards/addresses")
2. **Ring evidence** — the transaction's `ring_category` and cluster size, phrased per category (`behavioral_only` → "shares suspicious behavioral patterns with N others")
3. **Rule flags** — already plain English from `src/rules.py`, appended as-is

Computed only for cases that actually get surfaced (`flag`/`second_look`, not `pass`) — SHAP is real per-row compute, and a "pass" case never reaches an analyst, so there's no reason to pay for its explanation.

## The FastAPI service (`src/api.py`)

| Route | Purpose |
|---|---|
| `POST /score` | Score one transaction (optionally with `history` — that card's own recent transactions, for the velocity/spike rules) |
| `POST /batch-score` | Score many, with an optional `second_look_floor` |
| `GET /cluster-alerts` | Ranked ring alerts, optionally filtered to specific categories (see below) |
| `GET /cluster-alerts/{id}/graph` | Nodes + typed edges for one cluster, for the network view |
| `POST /decision` / `GET /decisions` | Record and read analyst decisions |
| `GET /dev/sample-transactions` | Dev/demo convenience — stands in for a real transaction stream until Day 11's replay is a genuine live feed |

## The case queue (`app.py`, Streamlit)

A thin HTTP client of the API above — it never touches the model, graphs, or database directly. Sidebar controls: sample size and seed, the second-look sensitivity dial (with the *real* offline-evaluated ₹ trade-off shown next to it, pulled from `results/hybrid_metrics.json` — never a live number from a small session sample, since that would be misleading), and ring-alert filters. Main view: transaction and ring cases with their plain-English reason, Approve/Hold/Decline/Escalate buttons, an on-demand pyvis network view per ring (blue = identity edge, orange = behavioral, green = both), and a **"▶ Advance stream"** button that simulates new transactions arriving by pulling and appending a new batch — explicitly labeled as simulated, not live, per the project's own stated principle.

### A leakage bug caught before it shipped

The first cut of `/dev/sample-transactions` returned `isFraud` in the payload — harmless to the classifier (it's not a trained feature) but a real discipline violation: this endpoint stands in for a genuinely unresolved incoming transaction, and a live feed would never carry the ground-truth label. Fixed to strip it, matching the same discipline already held everywhere else in this project (blind clustering, train-only category weights).

## The ranking-vs-coverage tension in ring alerts

`get_cluster_alerts()` ranks all qualifying clusters together by trust weight, then size. Since `identity_only` scores **0.58x** — genuinely below the population baseline, the Day 7 finding that identity evidence is weaker than behavioral evidence — and there are far more `behavioral_only`/`both` clusters in absolute count, an `identity_only` cluster essentially never reaches the top of that combined ranking. Confirmed empirically: a dry run of the full three-case demo flow (solo-fraud transaction, behavioral ring, identity ring) found **zero** `identity_only` cases in the default top-5 ring alerts.

This isn't a bug — the ranking is doing exactly what it should (surface what's most worth an analyst's attention first) — but it means the demo needs a way to deliberately inspect a specific category rather than only ever seeing the highest-priority ones. Fixed by adding a `category` filter to `/cluster-alerts` (repeatable query param) and a matching multiselect in the sidebar. Re-ran the full three-case dry run afterward: solo-fraud, behavioral-ring, and identity-ring cases all surfaced and resolved cleanly, with correct weak labels recorded.

Separately, solo-fraud cases (`isolated` + `flag`) are genuinely rare — only 10 of 2,000 demo-pool transactions (0.5%) — which tracks with the project's own design: a transaction that's isolated *and* still gets flagged is exactly the "solo fraud, no accomplices" case the classifier alone exists to catch, and that pattern is intrinsically less common than fraud that leaves some kind of ring trace. Not a bug either — just a reason the demo needs a large enough sample (close to the full 500-transaction UI max) to reliably show one.

## Persistent audit log + weak labels (`src/audit_log.py`)

Day 9's in-memory `/decision` store didn't survive an API restart. Replaced with a SQLite-backed one (plain `sqlite3`, no new dependency — the whole "database" is one file, `data/processed/audit_log.db`, gitignored like every other artifact). Verified by killing and restarting the API mid-session: all recorded decisions were still there afterward.

Each decision also becomes a weak label — the raw material a future retraining pass would use:

| Action | Meaning | Weak label |
|---|---|---|
| Decline | analyst agrees it's fraud | 1 |
| Approve | analyst overrides the flag; false positive | 0 |
| Escalate | not a verdict — passed to someone else | none |
| Hold | not a verdict — deferred | none |

The review-history tab shows these as counts and a cumulative confirmed-vs-dismissed trend — deliberately cumulative rather than a rate-over-time, since a single demo session doesn't have enough decisions for a bucketed rate to mean anything.

## Deployment: prepared, not shipped

A `Dockerfile` + slim `requirements-api.txt` were built and validated locally (image builds, container runs, every route works). Two real findings from that work:

- XGBoost's Linux wheel pulls in `nvidia-nccl-cu12` (~340MB) for GPU/distributed training a CPU-only inference server never uses — stripped via `pip install --no-deps`, cutting the image from 2.82GB to 1.85GB.
- The backend's baseline memory footprint is **~1.7GB idle** — the three in-memory networkx graphs (~1.09M edges) plus several parallel per-transaction dicts (degree, weighted-degree, PageRank, cluster partition) for all 590K transactions. A first cut of the demo-sample endpoint made this worse (4.2GB, from loading the entire merged dataset for a demo that only needs a couple thousand rows) — fixed with a pre-generated 2,000-row sample file.

Net result: this doesn't fit Render's free tier (512MB) or Fly.io's smallest free machines. Deployment is deliberately parked until the product is fully built, per direction — nothing has been pushed anywhere or cost anything.

## What's next

- Day 12: full integration polish (the three-case dry run above already covers most of this), finalize this doc set
- Day 13: README pass, architecture diagram at the top level
- Day 14: pitch script + rehearsal
