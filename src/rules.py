"""Rule-based fraud flags, run alongside the ML score -- not blended into it.

Real payment-fraud systems layer simple, deterministic, explainable rules next
to the ML score rather than folding them into one number (see
docs/hybrid-merge.md for the research this follows). Each rule here produces
its own boolean flag and a plain-English reason, exactly what the case queue
needs to show an analyst -- independent of, and complementary to, the
classifier and graph signals.

All rules are computed causally: each transaction is only compared against
transactions that happened strictly *before* it (by TransactionDT), for the
same card. This matters because a rule using "future" transactions relative
to the one being scored could never actually fire that way in production --
at scoring time, the future hasn't happened yet.
"""

import numpy as np
import pandas as pd

DEFAULT_VELOCITY_WINDOW_SECONDS = 60 * 60  # 1 hour
DEFAULT_VELOCITY_THRESHOLD = 3  # 3+ prior transactions in the window triggers the flag
DEFAULT_AMOUNT_SPIKE_MULTIPLIER = 5.0  # current amount > 5x this card's own causal median


def _card_key(df: pd.DataFrame) -> pd.Series:
    return df["card1"].astype(str) + "_" + df["card2"].astype(str)


def flag_high_card_velocity(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    window: int = DEFAULT_VELOCITY_WINDOW_SECONDS,
    min_count: int = DEFAULT_VELOCITY_THRESHOLD,
) -> pd.Series:
    """True if this card had >= min_count other transactions in the preceding `window` seconds."""
    work = pd.DataFrame({
        "card_key": _card_key(df),
        "time": df[time_col].to_numpy(),
    }, index=df.index)

    prior_count = pd.Series(0, index=df.index, dtype=int)
    for _, group in work.groupby("card_key"):
        times = group["time"].sort_values()
        idx = times.index.to_numpy()
        t = times.to_numpy()
        # For each transaction, count prior transactions (strictly before it)
        # within the window -- searchsorted finds the window's lower/upper bounds.
        lower = np.searchsorted(t, t - window, side="left")
        upper = np.searchsorted(t, t, side="left")  # exclusive of the transaction itself
        counts = upper - lower
        prior_count.loc[idx] = counts

    return prior_count >= min_count


def flag_new_card_high_value(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
    amount_threshold: float = None,
) -> pd.Series:
    """True if this is the card's first-ever transaction (in this data) and the
    amount exceeds `amount_threshold` (caller should pass a threshold computed
    from train-only data, e.g. a percentile -- see notebook for the value used).
    """
    if amount_threshold is None:
        raise ValueError("amount_threshold must be provided (compute from train-only data)")

    work = pd.DataFrame({
        "card_key": _card_key(df),
        "time": df[time_col].to_numpy(),
    }, index=df.index)
    first_seen_time = work.groupby("card_key")["time"].transform("min")
    is_first = work["time"] == first_seen_time

    return is_first & (df[amount_col] >= amount_threshold)


def flag_amount_spike(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
    multiplier: float = DEFAULT_AMOUNT_SPIKE_MULTIPLIER,
) -> pd.Series:
    """True if this transaction's amount is > multiplier times the card's own
    median amount from strictly-prior transactions. False (never flagged) for
    a card's first transaction, since there's no prior history to compare to.
    """
    work = pd.DataFrame({
        "card_key": _card_key(df),
        "time": df[time_col].to_numpy(),
        "amount": df[amount_col].to_numpy(),
    }, index=df.index)

    flagged = pd.Series(False, index=df.index)
    for _, group in work.groupby("card_key"):
        ordered = group.sort_values("time")
        # expanding median of all amounts *before* the current row (shift(1) excludes it)
        causal_median = ordered["amount"].shift(1).expanding().median()
        spike = ordered["amount"] > (causal_median * multiplier)
        flagged.loc[ordered.index] = spike.fillna(False)

    return flagged


def apply_all_rules(df: pd.DataFrame, amount_threshold_for_new_card: float) -> pd.DataFrame:
    """Run all rules; return a DataFrame of boolean flag columns plus a combined
    `any_rule_triggered` column and a plain-English `rule_reasons` column.
    """
    result = pd.DataFrame(index=df.index)
    result["rule_high_velocity"] = flag_high_card_velocity(df)
    result["rule_new_card_high_value"] = flag_new_card_high_value(df, amount_threshold=amount_threshold_for_new_card)
    result["rule_amount_spike"] = flag_amount_spike(df)

    reasons = {
        "rule_high_velocity": f"3+ transactions from this card in the last hour",
        "rule_new_card_high_value": "first transaction from this card, and unusually high value",
        "rule_amount_spike": "amount is far above this card's typical spend",
    }

    def build_reason(row):
        triggered = [reasons[col] for col in reasons if row[col]]
        return "; ".join(triggered) if triggered else ""

    result["rule_reasons"] = result.apply(build_reason, axis=1)
    result["any_rule_triggered"] = result[list(reasons.keys())].any(axis=1)
    return result
