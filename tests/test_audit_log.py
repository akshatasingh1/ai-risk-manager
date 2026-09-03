import os

import pytest
from dotenv import load_dotenv

from src.audit_log import fetch_decisions, init_db, insert_decision, weak_label_for_action

load_dotenv()
DB_STRING = os.environ.get("DB_STRING")
requires_postgres = pytest.mark.skipif(
    not DB_STRING, reason="DB_STRING not set -- these tests hit the real Postgres instance directly"
)


def test_weak_label_mapping():
    assert weak_label_for_action("decline") == 1
    assert weak_label_for_action("approve") == 0
    assert weak_label_for_action("escalate") is None
    assert weak_label_for_action("hold") is None


# --- Everything below hits the real, configured Postgres database directly --
# there's no local/offline backend anymore (Postgres-only, per direction).
# Every test uses clearly-sentinel negative transaction_ids so they can never
# collide with a real one, and cleans up after itself even if an assertion
# fails, so the live database is never left with test noise.

@pytest.fixture
def cleanup():
    sentinel_ids: list[int] = []
    yield sentinel_ids
    if sentinel_ids:
        import psycopg2
        conn = psycopg2.connect(DB_STRING)
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM decisions WHERE transaction_id = ANY(%s)", (sentinel_ids,))
            conn.commit()
        finally:
            conn.close()


@requires_postgres
def test_init_db_is_idempotent():
    init_db(DB_STRING)
    init_db(DB_STRING)  # re-running init must not error or wipe existing data


@requires_postgres
def test_insert_and_fetch_roundtrip(cleanup):
    init_db(DB_STRING)
    sentinel_id = -900101
    cleanup.append(sentinel_id)

    record = insert_decision(
        DB_STRING, transaction_id=sentinel_id, action="decline", analyst="priya", note="looks bad",
        context_score=0.91, context_category="both", context_reason="driven mainly by X",
    )

    assert record["weak_label"] == 1
    assert record["transaction_id"] == sentinel_id
    assert record["recorded_at"]  # a timestamp was generated

    fetched = [d for d in fetch_decisions(DB_STRING) if d["transaction_id"] == sentinel_id]
    assert len(fetched) == 1
    assert fetched[0]["weak_label"] == 1
    assert fetched[0]["context_score"] == pytest.approx(0.91)


@requires_postgres
def test_fetch_decisions_returns_oldest_first(cleanup):
    init_db(DB_STRING)
    sentinel_approve, sentinel_decline, sentinel_escalate = -900102, -900103, -900104
    cleanup.extend([sentinel_approve, sentinel_decline, sentinel_escalate])

    r1 = insert_decision(DB_STRING, transaction_id=sentinel_approve, action="approve")
    r2 = insert_decision(DB_STRING, transaction_id=sentinel_decline, action="decline")
    r3 = insert_decision(DB_STRING, cluster_id=99, transaction_id=sentinel_escalate, action="escalate")

    fetched = {d["transaction_id"]: d for d in fetch_decisions(DB_STRING)}
    # ids must increase in insertion order (oldest first), regardless of
    # whatever other rows already exist in the shared live table.
    assert r1["id"] < r2["id"] < r3["id"]
    assert fetched[sentinel_approve]["weak_label"] == 0
    assert fetched[sentinel_decline]["weak_label"] == 1
    assert fetched[sentinel_escalate]["weak_label"] is None
    assert fetched[sentinel_escalate]["cluster_id"] == 99


@requires_postgres
def test_weak_label_mapping_against_live_db(cleanup):
    sentinel_approve, sentinel_escalate = -900105, -900106
    cleanup.extend([sentinel_approve, sentinel_escalate])

    insert_decision(DB_STRING, transaction_id=sentinel_approve, action="approve")
    insert_decision(DB_STRING, transaction_id=sentinel_escalate, action="escalate")

    fetched = {d["transaction_id"]: d for d in fetch_decisions(DB_STRING)}
    assert fetched[sentinel_approve]["weak_label"] == 0
    assert fetched[sentinel_escalate]["weak_label"] is None
