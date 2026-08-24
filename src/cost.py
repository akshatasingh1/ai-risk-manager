"""Cost-based threshold selection.

Cost model follows standard cost-sensitive fraud-detection practice: a missed
fraud (false negative) costs the actual transaction amount lost; a false
alarm (false positive) costs a fixed manual-review overhead. See
docs/classifier.md for the reasoning behind the specific review-cost figure.
"""

import numpy as np
import pandas as pd

DEFAULT_REVIEW_COST = 100.0  # cost of one manual review (a false alarm)


def total_cost(y_true, y_proba, amounts, threshold: float, review_cost: float = DEFAULT_REVIEW_COST) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    amounts = np.asarray(amounts)

    y_pred = (y_proba >= threshold).astype(int)

    false_positives = (y_pred == 1) & (y_true == 0)
    false_negatives = (y_pred == 0) & (y_true == 1)

    fp_cost = false_positives.sum() * review_cost
    fn_cost = amounts[false_negatives].sum()
    return float(fp_cost + fn_cost)


def find_optimal_threshold(
    y_true, y_proba, amounts,
    review_cost: float = DEFAULT_REVIEW_COST,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float, pd.DataFrame]:
    """Return (best_threshold, best_cost, cost_curve_df) minimizing total cost."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    costs = np.array([total_cost(y_true, y_proba, amounts, t, review_cost) for t in thresholds])
    best_idx = costs.argmin()
    curve = pd.DataFrame({"threshold": thresholds, "total_cost": costs})
    return float(thresholds[best_idx]), float(costs[best_idx]), curve
