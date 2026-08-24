import json

import pandas as pd

from src.split import apply_locked_split


def test_apply_locked_split_matches_saved_ids(tmp_path):
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "isFraud": [0, 1, 0, 0, 1],
    })

    split_path = tmp_path / "split_ids.json"
    split_path.write_text(json.dumps({
        "train_ids": [1, 2, 3],
        "holdout_ids": [4, 5],
    }))

    train_df, holdout_df = apply_locked_split(df, split_path=split_path)

    assert sorted(train_df["TransactionID"]) == [1, 2, 3]
    assert sorted(holdout_df["TransactionID"]) == [4, 5]
    # No overlap and nothing dropped.
    assert set(train_df["TransactionID"]) & set(holdout_df["TransactionID"]) == set()
    assert len(train_df) + len(holdout_df) == len(df)
