import pandas as pd

from src.data import load_merged_train


def _write_synthetic_raw(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)

    transactions = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "isFraud": [0, 1, 0],
        "TransactionAmt": [100.0, 250.5, 40.0],
        "card1": [1001, 1002, 1003],
    })
    transactions.to_csv(raw_dir / "train_transaction.csv", index=False)

    # Only transactions 1 and 3 have a matching identity row — mirrors the
    # real dataset, where most transactions have none (see Day 1 EDA: ~24%).
    identity = pd.DataFrame({
        "TransactionID": [1, 3],
        "DeviceType": ["mobile", "desktop"],
    })
    identity.to_csv(raw_dir / "train_identity.csv", index=False)


def test_load_merged_train_left_joins_on_transaction_id(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_synthetic_raw(raw_dir)

    merged = load_merged_train(use_cache=False, raw_dir=raw_dir, processed_dir=processed_dir)

    assert len(merged) == 3
    assert "DeviceType" in merged.columns
    # Transaction 2 has no identity row -> DeviceType must be missing, not dropped.
    row_2 = merged.loc[merged["TransactionID"] == 2, "DeviceType"]
    assert row_2.isna().all()
    row_1 = merged.loc[merged["TransactionID"] == 1, "DeviceType"]
    assert row_1.item() == "mobile"


def test_load_merged_train_caches_to_parquet(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_synthetic_raw(raw_dir)

    first = load_merged_train(use_cache=True, raw_dir=raw_dir, processed_dir=processed_dir)
    assert (processed_dir / "train_merged.parquet").exists()

    # Delete the raw CSVs; a second call must still succeed by reading the
    # cache instead of re-reading (now-missing) raw files.
    (raw_dir / "train_transaction.csv").unlink()
    (raw_dir / "train_identity.csv").unlink()

    second = load_merged_train(use_cache=True, raw_dir=raw_dir, processed_dir=processed_dir)
    pd.testing.assert_frame_equal(first, second)
