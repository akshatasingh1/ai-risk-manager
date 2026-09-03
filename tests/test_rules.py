import pandas as pd

from src.rules import (
    apply_all_rules,
    flag_amount_spike,
    flag_high_card_velocity,
    flag_new_card_high_value,
)


def test_flag_high_card_velocity_triggers_on_burst():
    df = pd.DataFrame({
        "card1": [100, 100, 100, 100, 200],
        "card2": [1, 1, 1, 1, 1],
        "TransactionDT": [0, 600, 1200, 1800, 0],  # card 100: 4 txns within 30 min
    })
    flags = flag_high_card_velocity(df, min_count=3)

    # 4th transaction (index 3) has 3 prior transactions (0, 600, 1200) within the last hour -> True
    assert flags.iloc[3]
    # 1st transaction has 0 prior -> False
    assert not flags.iloc[0]
    # card 200's only transaction -> False
    assert not flags.iloc[4]


def test_flag_high_card_velocity_ignores_transactions_outside_window():
    df = pd.DataFrame({
        "card1": [100, 100, 100, 100],
        "card2": [1, 1, 1, 1],
        "TransactionDT": [0, 10000, 20000, 20100],  # first two are >1h apart from the rest
    })
    flags = flag_high_card_velocity(df, min_count=2)
    # Last transaction (20100) has only one prior within the window (20000, 100s ago) -> not >= 2
    assert not flags.iloc[3]


def test_flag_new_card_high_value():
    df = pd.DataFrame({
        "card1": [100, 100, 200],
        "card2": [1, 1, 1],
        "TransactionDT": [0, 100, 0],
        "TransactionAmt": [500.0, 10.0, 5.0],
    })
    flags = flag_new_card_high_value(df, amount_threshold=100.0)

    assert flags.iloc[0]  # card 100's first transaction, amount 500 >= threshold
    assert not flags.iloc[1]  # card 100's second transaction -- not "first" anymore
    assert not flags.iloc[2]  # card 200's first transaction, but amount too low


def test_flag_amount_spike_needs_prior_history():
    df = pd.DataFrame({
        "card1": [100, 100, 100],
        "card2": [1, 1, 1],
        "TransactionDT": [0, 100, 200],
        "TransactionAmt": [50.0, 50.0, 1000.0],  # median of prior (50, 50) = 50; 1000 > 5x50
    })
    flags = flag_amount_spike(df, multiplier=5.0)

    assert not flags.iloc[0]  # first transaction ever for this card -- no prior history, never flagged
    assert not flags.iloc[1]  # 50 is not > 5x the prior median (50)
    assert flags.iloc[2]  # 1000 > 5 * 50


def test_apply_all_rules_combines_flags_and_reasons():
    df = pd.DataFrame({
        "card1": [100],
        "card2": [1],
        "TransactionDT": [0],
        "TransactionAmt": [1000.0],
    })
    result = apply_all_rules(df, amount_threshold_for_new_card=500.0)

    assert result["rule_new_card_high_value"].iloc[0]
    assert result["any_rule_triggered"].iloc[0]
    assert "unusually high value" in result["rule_reasons"].iloc[0]
