import numpy as np
import pandas as pd

from src.features import build_preprocessor, get_feature_lists


def test_get_feature_lists_splits_numeric_and_categorical():
    df = pd.DataFrame({
        "TransactionAmt": [1.0],
        "card1": [100],
        "C3": [2],
        "D5": [3],
        "V100": [0.5],   # V-columns detected by prefix, not an explicit list
        "id_05": [1.0],
        "ProductCD": ["W"],
        "M4": ["T"],
        "id_31": ["chrome"],   # categorical id_ column (>=12) — not treated as numeric
        "SomeUnrelatedColumn": [1],
    })

    numeric, categorical = get_feature_lists(df)

    for col in ["TransactionAmt", "card1", "C3", "D5", "V100", "id_05"]:
        assert col in numeric
    for col in ["ProductCD", "M4"]:
        assert col in categorical

    assert "id_31" not in numeric
    assert "id_31" not in categorical
    assert "SomeUnrelatedColumn" not in numeric
    assert "SomeUnrelatedColumn" not in categorical


def test_get_feature_lists_only_returns_present_columns():
    df = pd.DataFrame({"TransactionAmt": [1.0], "ProductCD": ["W"]})
    numeric, categorical = get_feature_lists(df)
    assert numeric == ["TransactionAmt"]
    assert categorical == ["ProductCD"]


def test_build_preprocessor_handles_missing_values_and_unseen_categories():
    train_df = pd.DataFrame({
        "TransactionAmt": [10.0, 20.0, np.nan],
        "ProductCD": ["W", "C", "W"],
    })
    test_df = pd.DataFrame({
        "TransactionAmt": [np.nan],
        "ProductCD": ["never_seen_in_train"],
    })

    preprocessor = build_preprocessor(["TransactionAmt"], ["ProductCD"])
    preprocessor.fit(train_df)

    transformed = preprocessor.transform(test_df)

    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == 1
