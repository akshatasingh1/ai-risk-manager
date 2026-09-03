"""Persistent audit log + weak-label capture -- Day 11.

Replaces Day 9's in-memory `/decision` store with a SQLite-backed one, so an
analyst's decisions survive an API restart (see docs/problem-and-approach.md,
Section 2a: "a review history view ... so the tool visibly earns trust over
time"). Every decision also gets translated into a weak label -- the raw
material a future retraining cycle would use, not just an audit trail.

Weak-label mapping (a real judgment call, confirmed with the user, not
derivable from the code alone):
  - decline  -> 1 (analyst agrees this is fraud)
  - approve  -> 0 (analyst overrides the flag; this is a false positive)
  - escalate -> None (not a verdict -- passed to someone else)
  - hold     -> None (not a verdict -- deferred)

Uses the plain `sqlite3` standard-library module -- no new dependency, and a
new connection per call (SQLite handles many short-lived connections fine at
this scale) rather than one long-lived shared connection, so there's no
cross-thread/cross-request locking to reason about.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "audit_log.db"

WEAK_LABEL_BY_ACTION = {"decline": 1, "approve": 0}


def weak_label_for_action(action: str) -> int | None:
    return WEAK_LABEL_BY_ACTION.get(action)


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER,
                cluster_id INTEGER,
                action TEXT NOT NULL,
                analyst TEXT,
                note TEXT,
                weak_label INTEGER,
                context_score REAL,
                context_category TEXT,
                context_reason TEXT,
                recorded_at TEXT NOT NULL
            )
        """)


def insert_decision(
    db_path: Path = DB_PATH,
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

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO decisions (
                transaction_id, cluster_id, action, analyst, note,
                weak_label, context_score, context_category, context_reason, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (transaction_id, cluster_id, action, analyst, note,
             weak_label, context_score, context_category, context_reason, recorded_at),
        )
        row_id = cursor.lastrowid

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


def fetch_decisions(db_path: Path = DB_PATH) -> list[dict]:
    """All recorded decisions, oldest first."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM decisions ORDER BY id ASC").fetchall()
    return [dict(row) for row in rows]
