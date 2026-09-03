from src.case_reason import combine_case_reason, describe_feature, ring_phrase


def test_describe_feature_groups_known_columns():
    assert describe_feature("C13") == describe_feature("C1")  # same category
    assert "linked identities" in describe_feature("C13")
    assert "time since" in describe_feature("D4")
    assert describe_feature("TransactionAmt") == "an unusual transaction amount"
    assert describe_feature("V70") == "an anonymized behavioral signal (V70)"


def test_describe_feature_falls_back_to_raw_name_for_unknown_columns():
    assert describe_feature("some_future_column") == "some_future_column"


def test_ring_phrase_counts_other_members_not_total_size():
    assert "3 other transactions" in ring_phrase("behavioral_only", cluster_size=4)
    assert "1 other transaction" in ring_phrase("identity_only", cluster_size=2)  # singular, not "1 transactions"


def test_ring_phrase_empty_for_non_ring_categories():
    assert ring_phrase("isolated", cluster_size=1) == ""
    assert ring_phrase("neither", cluster_size=5) == ""


def test_combine_case_reason_capitalizes_only_first_letter():
    # str.capitalize() would wrongly lowercase 'V312' -- must not do that.
    reason = combine_case_reason("driven mainly by an anonymized behavioral signal (V312)", "", "")
    assert reason == "Driven mainly by an anonymized behavioral signal (V312)."


def test_combine_case_reason_handles_any_subset_of_evidence():
    assert combine_case_reason("", "", "") == ""
    assert combine_case_reason("", "shares a card with 2 other transactions", "") == "Shares a card with 2 other transactions."
    assert combine_case_reason("", "", "amount is far above typical spend") == "Rule flags: amount is far above typical spend."

    full = combine_case_reason("driven mainly by X", "shares a card with 2 other transactions", "amount spike")
    assert full == "Driven mainly by X. Also, this transaction shares a card with 2 other transactions. Rule flags: amount spike."
