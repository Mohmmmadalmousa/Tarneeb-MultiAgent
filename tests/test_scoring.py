"""Hand-written truth table for score_deal under tarneeb-classic-v1.

Every expected value below was written by hand from the frozen rule table
(C20-C24, C26), not produced by running ``score_deal``.

Rule recap used to build the table, from the bidding team's view. Below,
``bidding_team_tricks`` is that team's trick count and
``defending_team_tricks = 13 - bidding_team_tricks``:

* bid 13 made (bidding_team_tricks = 13)   -> (+26, 0)                            C23
* bid 13 failed                            -> (-16, 2 * defending_team_tricks)    C24
* bid < 13, bidding_team_tricks = 13       -> (+16, 0)                            C22 bidder kaboot
* bid < 13, bidding_team_tricks >= bid     -> (+bidding_team_tricks, 0)           C20
* bid < 13, defending_team_tricks = 13     -> (-bid, 16)                          C26 defender kaboot
* bid < 13, otherwise failed               -> (-bid, defending_team_tricks)       C21
"""

from __future__ import annotations

import pytest

from tarneeb.core.rules import RulesConfig
from tarneeb.core.scoring import score_deal

# (bid, tricks_bidding_team) -> (delta_bidding_team, delta_defending_team)
# 7 bids x 14 trick counts = 98 rows. No row is duplicated.
TRUTH_TABLE: dict[tuple[int, int], tuple[int, int]] = {
    # --- bid 7 --------------------------------------------------------------
    (7, 0): (-7, 16),  # C26 defender kaboot
    (7, 1): (-7, 12),
    (7, 2): (-7, 11),
    (7, 3): (-7, 10),
    (7, 4): (-7, 9),
    (7, 5): (-7, 8),
    (7, 6): (-7, 7),
    (7, 7): (7, 0),  # made exactly
    (7, 8): (8, 0),
    (7, 9): (9, 0),
    (7, 10): (10, 0),
    (7, 11): (11, 0),
    (7, 12): (12, 0),
    (7, 13): (16, 0),  # C22 bidder kaboot
    # --- bid 8 --------------------------------------------------------------
    (8, 0): (-8, 16),  # C26 defender kaboot
    (8, 1): (-8, 12),
    (8, 2): (-8, 11),
    (8, 3): (-8, 10),
    (8, 4): (-8, 9),
    (8, 5): (-8, 8),
    (8, 6): (-8, 7),
    (8, 7): (-8, 6),
    (8, 8): (8, 0),  # made exactly
    (8, 9): (9, 0),
    (8, 10): (10, 0),
    (8, 11): (11, 0),
    (8, 12): (12, 0),
    (8, 13): (16, 0),  # C22 bidder kaboot
    # --- bid 9 --------------------------------------------------------------
    (9, 0): (-9, 16),  # C26 defender kaboot
    (9, 1): (-9, 12),
    (9, 2): (-9, 11),
    (9, 3): (-9, 10),
    (9, 4): (-9, 9),
    (9, 5): (-9, 8),
    (9, 6): (-9, 7),
    (9, 7): (-9, 6),
    (9, 8): (-9, 5),
    (9, 9): (9, 0),  # made exactly
    (9, 10): (10, 0),
    (9, 11): (11, 0),
    (9, 12): (12, 0),
    (9, 13): (16, 0),  # C22 bidder kaboot
    # --- bid 10 -------------------------------------------------------------
    (10, 0): (-10, 16),  # C26 defender kaboot
    (10, 1): (-10, 12),
    (10, 2): (-10, 11),
    (10, 3): (-10, 10),
    (10, 4): (-10, 9),
    (10, 5): (-10, 8),
    (10, 6): (-10, 7),
    (10, 7): (-10, 6),
    (10, 8): (-10, 5),
    (10, 9): (-10, 4),
    (10, 10): (10, 0),  # made exactly
    (10, 11): (11, 0),
    (10, 12): (12, 0),
    (10, 13): (16, 0),  # C22 bidder kaboot
    # --- bid 11 -------------------------------------------------------------
    (11, 0): (-11, 16),  # C26 defender kaboot
    (11, 1): (-11, 12),
    (11, 2): (-11, 11),
    (11, 3): (-11, 10),
    (11, 4): (-11, 9),
    (11, 5): (-11, 8),
    (11, 6): (-11, 7),
    (11, 7): (-11, 6),
    (11, 8): (-11, 5),
    (11, 9): (-11, 4),
    (11, 10): (-11, 3),
    (11, 11): (11, 0),  # made exactly
    (11, 12): (12, 0),
    (11, 13): (16, 0),  # C22 bidder kaboot
    # --- bid 12 -------------------------------------------------------------
    (12, 0): (-12, 16),  # C26 defender kaboot
    (12, 1): (-12, 12),
    (12, 2): (-12, 11),
    (12, 3): (-12, 10),
    (12, 4): (-12, 9),
    (12, 5): (-12, 8),
    (12, 6): (-12, 7),
    (12, 7): (-12, 6),
    (12, 8): (-12, 5),
    (12, 9): (-12, 4),
    (12, 10): (-12, 3),
    (12, 11): (-12, 2),
    (12, 12): (12, 0),  # made exactly
    (12, 13): (16, 0),  # C22 bidder kaboot
    # --- bid 13 (C23 / C24; defenders score double, C26 does not apply) ------
    (13, 0): (-16, 26),  # failed, defenders take all 13 -> the -42 extreme
    (13, 1): (-16, 24),
    (13, 2): (-16, 22),
    (13, 3): (-16, 20),
    (13, 4): (-16, 18),
    (13, 5): (-16, 16),
    (13, 6): (-16, 14),
    (13, 7): (-16, 12),
    (13, 8): (-16, 10),
    (13, 9): (-16, 8),
    (13, 10): (-16, 6),
    (13, 11): (-16, 4),
    (13, 12): (-16, 2),
    (13, 13): (26, 0),  # C23 bid 13 made -> the +26 extreme
}


def test_truth_table_covers_the_whole_domain() -> None:
    assert len(TRUTH_TABLE) == 98
    assert set(TRUTH_TABLE) == {
        (bid, bidding_team_tricks) for bid in range(7, 14) for bidding_team_tricks in range(14)
    }


@pytest.mark.parametrize(("scoring_inputs", "expected_score_changes"), sorted(TRUTH_TABLE.items()))
def test_score_deal_matches_truth_table(
    scoring_inputs: tuple[int, int], expected_score_changes: tuple[int, int]
) -> None:
    bid, bidding_team_tricks = scoring_inputs
    assert score_deal(bid, bidding_team_tricks, RulesConfig()) == expected_score_changes


# --- score-difference bounds ------------------------------------------------


def _score_differences_from_truth_table() -> list[int]:
    """The bidding team's score change minus the defending team's, per row."""
    return [
        bidding_team_score_change - defending_team_score_change
        for (bidding_team_score_change, defending_team_score_change) in TRUTH_TABLE.values()
    ]


def _rows_with_score_difference(target_difference: int) -> list[tuple[int, int]]:
    return [
        scoring_inputs
        for scoring_inputs, (
            bidding_team_score_change,
            defending_team_score_change,
        ) in TRUTH_TABLE.items()
        if bidding_team_score_change - defending_team_score_change == target_difference
    ]


def test_score_difference_maximum_is_26() -> None:
    maximum_score_difference = max(_score_differences_from_truth_table())
    assert maximum_score_difference == 26
    assert _rows_with_score_difference(26) == [(13, 13)]


def test_score_difference_minimum_is_minus_42() -> None:
    minimum_score_difference = min(_score_differences_from_truth_table())
    assert minimum_score_difference == -42
    assert _rows_with_score_difference(-42) == [(13, 0)]


def test_score_difference_bounds_hold_for_the_implementation() -> None:
    rules_config = RulesConfig()
    score_differences = []
    for bid in range(rules_config.min_bid, rules_config.max_bid + 1):
        for bidding_team_tricks in range(14):
            bidding_team_score_change, defending_team_score_change = score_deal(
                bid, bidding_team_tricks, rules_config
            )
            score_differences.append(bidding_team_score_change - defending_team_score_change)

    assert min(score_differences) == -42
    assert max(score_differences) == 26


def test_defenders_score_zero_whenever_the_contract_is_made() -> None:
    for scoring_inputs, score_changes in TRUTH_TABLE.items():
        bid, bidding_team_tricks = scoring_inputs
        _, defending_team_score_change = score_changes
        if bidding_team_tricks >= bid:
            assert defending_team_score_change == 0, scoring_inputs


# --- invalid inputs ---------------------------------------------------------


@pytest.mark.parametrize("bid", [-1, 0, 6, 14, 26])
def test_bid_out_of_range_raises(bid: int) -> None:
    with pytest.raises(ValueError, match="bid must be in"):
        score_deal(bid, 7, RulesConfig())


@pytest.mark.parametrize("bidding_team_tricks", [-1, 14, 52])
def test_trick_count_out_of_range_raises(bidding_team_tricks: int) -> None:
    with pytest.raises(ValueError, match="tricks_bidding_team must be in"):
        score_deal(10, bidding_team_tricks, RulesConfig())
