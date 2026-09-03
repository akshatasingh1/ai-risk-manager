"""FastAPI service -- Day 9.

Exposes the v3.1 hybrid scoring pipeline (src/serving.py) over HTTP:
`/score`, `/batch-score`, `/cluster-alerts`, `/decision`. The case-queue UI
(Day 10) and the simulated transaction stream (Day 11) are both just clients
of this API -- neither touches the model, graphs, or raw dataset directly.

Run with: uvicorn src.api:app --reload
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.audit_log import fetch_decisions, init_db, insert_decision
from src.serving import CLASSIFIER_THRESHOLD, ScoringContext, get_cluster_alerts, get_cluster_graph, score_batch, score_one

load_dotenv()
DB_STRING = os.environ.get("DB_STRING")

DEMO_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "demo_sample.parquet"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DB_STRING:
        raise RuntimeError(
            "DB_STRING is not set. This API is Postgres-only (no local SQLite fallback) -- "
            "set DB_STRING in a .env file at the repo root, e.g. DB_STRING=postgresql://user:pass@host/db"
        )
    app.state.ctx = ScoringContext()
    init_db(db_string=DB_STRING)
    app.state.demo_sample_df = None  # loaded lazily, only if /dev/sample-transactions is used
    yield


app = FastAPI(title="AI Risk Manager", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TransactionIn(BaseModel):
    """A raw transaction, same schema as the merged IEEE-CIS data. Only the
    three fields actually used outside the classifier (graph lookup, rules,
    cost) are required -- every other real column (card1, V1..V339, C1-C14,
    D1-D15, ...) is accepted as an extra field and simply passed through;
    any the classifier expects but the caller omitted are treated as missing
    (NaN), same as a genuinely sparse field in the source data.
    """
    model_config = ConfigDict(extra="allow")

    TransactionID: int
    TransactionDT: int
    TransactionAmt: float


class ScoreOut(BaseModel):
    TransactionID: int
    classifier_score: float
    decision: Literal["pass", "second_look", "flag"]
    ring_category: str
    ring_cluster_size: int
    rule_reasons: str
    any_rule_triggered: bool
    second_look: bool
    case_reason: str


class BatchScoreRequest(BaseModel):
    transactions: list[TransactionIn]
    second_look_floor: float | None = Field(
        default=None, description="Optional sensitivity dial -- see docs/hybrid-merge.md."
    )


class ScoreRequest(BaseModel):
    transaction: TransactionIn
    history: list[TransactionIn] = Field(
        default_factory=list,
        description="This card's own recent transactions, strictly before `transaction`, "
        "so the velocity/amount-spike rules have context to fire against.",
    )
    second_look_floor: float | None = None


class ClusterAlert(BaseModel):
    cluster_id: int
    size: int
    category: str
    trust_weight: float
    member_transaction_ids: list[int]


class ClusterGraphEdge(BaseModel):
    source: int
    target: int
    type: Literal["identity", "behavioral", "both"]
    weight: float


class ClusterGraphOut(BaseModel):
    cluster_id: int
    total_size: int
    nodes: list[int]
    edges: list[ClusterGraphEdge]


class DecisionIn(BaseModel):
    transaction_id: int | None = None
    cluster_id: int | None = None
    action: Literal["approve", "hold", "decline", "escalate"]
    analyst: str | None = None
    note: str | None = None
    # A snapshot of the case's own context at decision time (the caller
    # already has this from /score, /batch-score, or /cluster-alerts) --
    # stored alongside the decision so the history view and any future
    # retraining pass don't need to re-fetch or re-score the transaction.
    context_score: float | None = Field(
        default=None, description="classifier_score for a transaction case, or trust_weight for a ring case"
    )
    context_category: str | None = None
    context_reason: str | None = None


class DecisionRecord(DecisionIn):
    id: int
    weak_label: int | None = Field(
        description="1 = analyst confirmed fraud (decline), 0 = analyst dismissed as false positive (approve), "
        "None = no verdict yet (hold/escalate)"
    )
    recorded_at: datetime


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "classifier_threshold": CLASSIFIER_THRESHOLD}


def _to_dataframe(transactions: list[TransactionIn]) -> pd.DataFrame:
    return pd.DataFrame([t.model_dump() for t in transactions])


@app.post("/score", response_model=ScoreOut)
def score(request: ScoreRequest):
    ctx: ScoringContext = app.state.ctx
    history_df = _to_dataframe(request.history) if request.history else None
    return score_one(
        request.transaction.model_dump(), ctx, history=history_df, second_look_floor=request.second_look_floor
    )


@app.post("/batch-score", response_model=list[ScoreOut])
def batch_score(request: BatchScoreRequest):
    if not request.transactions:
        raise HTTPException(status_code=400, detail="transactions must be non-empty")

    ctx: ScoringContext = app.state.ctx
    df = _to_dataframe(request.transactions)
    result = score_batch(df, ctx, second_look_floor=request.second_look_floor)
    return result.to_dict(orient="records")


@app.get("/cluster-alerts", response_model=list[ClusterAlert])
def cluster_alerts(
    min_size: int = Query(default=2, ge=2),
    limit: int = Query(default=20, ge=1, le=200),
    category: list[Literal["identity_only", "behavioral_only", "both"]] | None = Query(
        default=None,
        description="Restrict to specific ring evidence categories (repeatable, e.g. "
        "?category=identity_only&category=both). Omit to rank all three together by trust "
        "weight -- since identity_only scores below the population baseline (see "
        "docs/ring-evaluation.md), it rarely reaches the top of that combined ranking, so this "
        "filter is how to deliberately look at one category rather than only the highest-priority ones.",
    ),
):
    ctx: ScoringContext = app.state.ctx
    kwargs = {"min_size": min_size, "limit": limit}
    if category:
        kwargs["categories"] = tuple(category)
    return get_cluster_alerts(ctx, **kwargs)


@app.get("/cluster-alerts/{cluster_id}/graph", response_model=ClusterGraphOut)
def cluster_graph(cluster_id: int, max_nodes: int = Query(default=30, ge=2, le=100)):
    ctx: ScoringContext = app.state.ctx
    result = get_cluster_graph(ctx, cluster_id, max_nodes=max_nodes)
    if not result["nodes"]:
        raise HTTPException(status_code=404, detail=f"cluster {cluster_id} not found")
    return result


@app.get("/dev/sample-transactions")
def sample_transactions(n: int = Query(default=150, ge=1, le=2000), seed: int = Query(default=42)):
    """Dev/demo convenience only -- NOT part of the real product surface.
    Stands in for the real transaction stream until Day 11's simulated-
    stream replay, so the case-queue UI can pull sample transactions without
    ever needing direct access to the (large, gitignored) raw dataset --
    keeping the UI a true thin client of this API, which matters once the UI
    and API are deployed as separate services.

    Reads from a small pre-generated 2,000-row sample of the holdout
    (data/processed/demo_sample.parquet, `isFraud` already stripped at
    generation time -- a real live feed would never carry the ground-truth
    label, same discipline as everywhere else in this project), not the
    full 590K-row/434-column merged dataset: loading the full frame here
    was measured to push the process to 4GB+ RAM for a demo convenience
    that only ever needs a couple thousand rows, which would force a far
    more expensive deploy tier for no real benefit.
    """
    if app.state.demo_sample_df is None:
        app.state.demo_sample_df = pd.read_parquet(DEMO_SAMPLE_PATH)

    pool = app.state.demo_sample_df
    sample = pool.sample(n=min(n, len(pool)), random_state=seed)
    return [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in sample.to_dict(orient="records")
    ]


@app.post("/decision", response_model=DecisionRecord)
def record_decision(decision: DecisionIn):
    if decision.transaction_id is None and decision.cluster_id is None:
        raise HTTPException(status_code=400, detail="one of transaction_id or cluster_id is required")

    return insert_decision(db_string=DB_STRING, **decision.model_dump())


@app.get("/decisions", response_model=list[DecisionRecord])
def list_decisions():
    return fetch_decisions(db_string=DB_STRING)
