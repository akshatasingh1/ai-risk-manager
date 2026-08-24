"""Feature definitions and the shared preprocessing pipeline for the classifier.

Column groupings follow the IEEE-CIS data dictionary. Kept in one place so the
baseline notebook, later model notebooks, and tests all agree on what a
"feature" is.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_NUMERIC = [f"id_{i:02d}" for i in range(1, 12)]
C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_COLS = [f"M{i}" for i in range(1, 10)]

BASE_NUMERIC = [
    "TransactionAmt", "card1", "card2", "card3", "card5",
    "addr1", "addr2", "dist1", "dist2",
]
BASE_CATEGORICAL = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType"]


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split a dataframe's columns into (numeric_features, categorical_features).

    Only columns actually present in `df` are returned, so this works on
    both the full merged dataset and smaller/test frames.
    """
    v_cols = [c for c in df.columns if c.startswith("V")]
    numeric = BASE_NUMERIC + C_COLS + D_COLS + v_cols + ID_NUMERIC
    categorical = BASE_CATEGORICAL + M_COLS

    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]
    return numeric, categorical


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Median-impute + scale numeric features; missing-fill + one-hot categorical features."""
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])
