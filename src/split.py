"""Reproducible train/holdout split.

The split is locked once (Day 2, via train_test_split with a fixed random
state) and its TransactionIDs saved to disk. Every model from Day 3 onward
must evaluate against that same holdout so results stay comparable.
"""

import json
from pathlib import Path

import pandas as pd

SPLIT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "split_ids.json"


def load_split_ids(split_path: Path = SPLIT_PATH) -> dict:
    with open(split_path) as f:
        return json.load(f)


def apply_locked_split(df: pd.DataFrame, split_path: Path = SPLIT_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `df` into (train_df, holdout_df) using the TransactionIDs locked in Day 2."""
    split_ids = load_split_ids(split_path)
    train_df = df[df["TransactionID"].isin(split_ids["train_ids"])]
    holdout_df = df[df["TransactionID"].isin(split_ids["holdout_ids"])]
    return train_df, holdout_df
