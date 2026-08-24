"""Shared data loading for the IEEE-CIS fraud dataset."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_merged_train(use_cache: bool = True, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load train_transaction + train_identity, merged on TransactionID.

    Caches the merged frame as parquet on first call so later notebooks
    don't have to re-read ~1.3GB of CSV every time.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = processed_dir / "train_merged.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    txn = pd.read_csv(raw_dir / "train_transaction.csv")
    identity = pd.read_csv(raw_dir / "train_identity.csv")
    merged = txn.merge(identity, on="TransactionID", how="left")

    merged.to_parquet(cache_path, index=False)
    return merged
