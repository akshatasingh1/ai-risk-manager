from src.cost import find_optimal_threshold, total_cost

Y_TRUE = [1, 1, 0, 0]
Y_PROBA = [0.9, 0.4, 0.6, 0.1]
AMOUNTS = [500, 300, 50, 20]
REVIEW_COST = 100.0


def test_total_cost_at_threshold_half():
    # pred = [1, 0, 1, 0] -> 1 false positive (idx 2), 1 false negative (idx 1, amount 300)
    cost = total_cost(Y_TRUE, Y_PROBA, AMOUNTS, threshold=0.5, review_cost=REVIEW_COST)
    assert cost == 100.0 + 300.0


def test_total_cost_flagging_everything_as_fraud():
    # threshold 0.05 -> pred = [1,1,1,1] -> 2 false positives, 0 false negatives
    cost = total_cost(Y_TRUE, Y_PROBA, AMOUNTS, threshold=0.05, review_cost=REVIEW_COST)
    assert cost == 2 * REVIEW_COST


def test_total_cost_flagging_nothing_as_fraud():
    # threshold 0.95 -> pred = [0,0,0,0] -> 0 false positives, both frauds missed
    cost = total_cost(Y_TRUE, Y_PROBA, AMOUNTS, threshold=0.95, review_cost=REVIEW_COST)
    assert cost == 500.0 + 300.0


def test_find_optimal_threshold_picks_cheapest_option():
    # Missed-fraud amounts (500, 300) far exceed the review cost (100), so
    # flagging everything (threshold 0.05) should be cheapest of the three.
    best_threshold, best_cost, curve = find_optimal_threshold(
        Y_TRUE, Y_PROBA, AMOUNTS, review_cost=REVIEW_COST,
        thresholds=[0.05, 0.5, 0.95],
    )
    assert best_threshold == 0.05
    assert best_cost == 200.0
    assert len(curve) == 3
