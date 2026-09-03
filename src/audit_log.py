"""Persistent audit log + weak-label capture -- Day 11.

Postgres only, raw SQL via psycopg2 -- no ORM, no local SQLite fallback.
`db_string` is a required, explicit parameter on every function (never read
from the environment inside this module): src/api.py is the only place that
reads `DB_STRING` (from `.env` via python-dotenv) and passes it through, so
this module stays a pure, fully-testable unit with no import-time or
call-time environment side effects.

Every decision also gets translated into a weak label -- the raw material a
future retraining cycle would use, not just an audit trail. Weak-label
mapping (a real judgment call, confirmed with the user, not derivable from
the code alone):
  - decline  -> 1 (analyst agrees this is fraud)
  - approve  -> 0 (analyst overrides the flag; this is a false positive)
  - escalate -> None (not a verdict -- passed to someone else)
  - hold     -> None (not a verdict -- deferred)
"""

from datetime import datetime, timezone

import psycopg2

WEAK_LABEL_BY_ACTION = {"decline": 1, "approve": 0}

_COLUMNS = [
    "id", "transaction_id", "cluster_id", "action", "analyst", "note",
    "weak_label", "context_score", "context_category", "context_reason", "recorded_at",
]

_DDL = """
    CREATE TABLE IF NOT EXISTS decisions (
        id SERIAL PRIMARY KEY,
        transaction_id BIGINT,
        cluster_id BIGINT,
        action TEXT NOT NULL,
        analyst TEXT,
        note TEXT,
        weak_label INTEGER,
        context_score DOUBLE PRECISION,
        context_category TEXT,
        context_reason TEXT,
        recorded_at TEXT NOT NULL
    )
"""


def weak_label_for_action(action: str) -> int | None:
    return WEAK_LABEL_BY_ACTION.get(action)


def init_db(db_string: str) -> None:
    conn = psycopg2.connect(db_string)
    try:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
    finally:
        conn.close()


def insert_decision(
    db_string: str,
    *,
    transaction_id: int | None = None,
    cluster_id: int | None = None,
    action: str,
    analyst: str | None = None,
    note: str | None = None,
    context_score: float | None = None,
    context_category: str | None = None,
    context_reason: str | None = None,
) -> dict:
    """Insert one decision, deriving its weak label from `action`, and
    return the full stored record (including the generated id/timestamp).
    """
    weak_label = weak_label_for_action(action)
    recorded_at = datetime.now(timezone.utc).isoformat()
    values = (transaction_id, cluster_id, action, analyst, note,
              weak_label, context_score, context_category, context_reason, recorded_at)

    sql = """
        INSERT INTO decisions (
            transaction_id, cluster_id, action, analyst, note,
            weak_label, context_score, context_category, context_reason, recorded_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    conn = psycopg2.connect(db_string)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, values)
            row_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()

    return {
        "id": row_id,
        "transaction_id": transaction_id,
        "cluster_id": cluster_id,
        "action": action,
        "analyst": analyst,
        "note": note,
        "weak_label": weak_label,
        "context_score": context_score,
        "context_category": context_category,
        "context_reason": context_reason,
        "recorded_at": recorded_at,
    }


def fetch_decisions(db_string: str) -> list[dict]:
    """All recorded decisions, oldest first."""
    conn = psycopg2.connect(db_string)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM decisions ORDER BY id ASC")
            rows = cur.fetchall()
    finally:
        conn.close()
    return [dict(zip(_COLUMNS, row)) for row in rows]
