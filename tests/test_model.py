import numpy as np
import pandas as pd

from src.model import build_xgb_pipeline


def test_build_xgb_pipeline_fits_and_predicts():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "TransactionAmt": rng.uniform(10, 500, n),
        "ProductCD": rng.choice(["W", "C", "R"], n),
    })
    y = pd.Series(rng.choice([0, 1], n, p=[0.9, 0.1]))

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    pipeline = build_xgb_pipeline(["TransactionAmt"], ["ProductCD"], scale_pos_weight=scale_pos_weight, n_estimators=10)
    pipeline.fit(df, y)

    proba = pipeline.predict_proba(df)
    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
