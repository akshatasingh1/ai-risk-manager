"""Shared model-building code for the classifier, so every notebook trains the same way."""

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor


def build_xgb_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
    n_estimators: int = 300,
) -> Pipeline:
    """XGBoost classifier with class-imbalance correction via scale_pos_weight.

    `scale_pos_weight` should be set to (count of negatives / count of
    positives) in the training data — see docs/classifier.md for why
    weighting was chosen over SMOTE.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=n_estimators,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="aucpr",
            n_jobs=-1,
        )),
    ])
