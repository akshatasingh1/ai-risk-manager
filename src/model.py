"""Shared model-building code for the classifier, so every notebook trains the same way."""

import lightgbm as lgb
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor


def build_lgbm_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
    n_estimators: int = 300,
) -> Pipeline:
    """LightGBM classifier with class weighting for the fraud imbalance.

    See docs/classifier.md for why class weighting was chosen over SMOTE.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline([
        ("prep", preprocessor),
        ("clf", lgb.LGBMClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1,
        )),
    ])
