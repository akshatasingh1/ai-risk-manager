from src.audit_log import fetch_decisions, init_db, insert_decision, weak_label_for_action


def test_weak_label_mapping():
    assert weak_label_for_action("decline") == 1
    assert weak_label_for_action("approve") == 0
    assert weak_label_for_action("escalate") is None
    assert weak_label_for_action("hold") is None


def test_insert_and_fetch_roundtrip(tmp_path):
    db_path = tmp_path / "audit.db"
    init_db(db_path)

    record = insert_decision(
        db_path, transaction_id=123, action="decline", analyst="priya", note="looks bad",
        context_score=0.91, context_category="both", context_reason="driven mainly by X",
    )

    assert record["weak_label"] == 1
    assert record["id"] == 1
    assert record["transaction_id"] == 123
    assert record["recorded_at"]  # a timestamp was generated

    fetched = fetch_decisions(db_path)
    assert len(fetched) == 1
    assert fetched[0]["transaction_id"] == 123
    assert fetched[0]["weak_label"] == 1
    assert fetched[0]["context_score"] == 0.91


def test_fetch_decisions_returns_oldest_first(tmp_path):
    db_path = tmp_path / "audit.db"
    init_db(db_path)

    insert_decision(db_path, transaction_id=1, action="approve")
    insert_decision(db_path, transaction_id=2, action="decline")
    insert_decision(db_path, cluster_id=99, action="escalate")

    fetched = fetch_decisions(db_path)
    assert [d["id"] for d in fetched] == [1, 2, 3]
    assert fetched[0]["weak_label"] == 0
    assert fetched[1]["weak_label"] == 1
    assert fetched[2]["weak_label"] is None
    assert fetched[2]["cluster_id"] == 99
    assert fetched[2]["transaction_id"] is None


def test_init_db_is_idempotent(tmp_path):
    db_path = tmp_path / "audit.db"
    init_db(db_path)
    insert_decision(db_path, transaction_id=1, action="hold")
    init_db(db_path)  # re-running init must not wipe existing data

    assert len(fetch_decisions(db_path)) == 1
