"""Duplicate-deal pairing correctness.

These test the construction of a pair, not aggregate means: both orientations
must come from the same underlying deal, the partnerships must actually be
swapped, and the sign convention must be the documented one.
"""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.core.cards import NUM_SEATS
from tarneeb.core.state import team_of_seat
from tarneeb.evaluation.arena import (
    Arena,
    build_seat_assignment,
    evaluation_match_seed,
)
from tarneeb.evaluation.metrics import DealOutcome, PairedOutcome


@pytest.fixture
def arena() -> Arena:
    return Arena(replay_sample_rate=0.0)


# --- seat assignment --------------------------------------------------------


def test_seat_assignment_places_partnerships_correctly() -> None:
    policy_a = StrongHeuristicAgent()
    policy_b = BasicHeuristicAgent()

    assignment = build_seat_assignment(policy_a, policy_b, policy_a_team=0)
    assert assignment.agent_for(0) is policy_a
    assert assignment.agent_for(2) is policy_a
    assert assignment.agent_for(1) is policy_b
    assert assignment.agent_for(3) is policy_b

    swapped = build_seat_assignment(policy_a, policy_b, policy_a_team=1)
    assert swapped.agent_for(1) is policy_a
    assert swapped.agent_for(3) is policy_a
    assert swapped.agent_for(0) is policy_b
    assert swapped.agent_for(2) is policy_b


def test_seat_assignment_rejects_a_bad_team() -> None:
    with pytest.raises(ValueError, match="policy_a_team must be"):
        build_seat_assignment(RandomAgent(), RandomAgent(), policy_a_team=2)


def test_partners_are_always_opposite_seats() -> None:
    assignment = build_seat_assignment(StrongHeuristicAgent(), BasicHeuristicAgent(), 0)
    for seat in range(NUM_SEATS):
        partner_seat = (seat + 2) % NUM_SEATS
        assert assignment.agent_for(seat) is assignment.agent_for(partner_seat)


# --- both halves of a pair share the same cards -----------------------------


def test_both_runs_of_a_pair_use_the_same_dealt_cards(arena: Arena) -> None:
    """The deal is a pure function of (match_seed, deal_index), so both match."""
    match_seed = evaluation_match_seed(evaluation_seed=1, pair_index=0)

    first_state = arena.engine.deal(arena.engine.new_deal(0, deal_index=0, match_seed=match_seed))
    second_state = arena.engine.deal(arena.engine.new_deal(0, deal_index=0, match_seed=match_seed))
    assert first_state.original_hands == second_state.original_hands


def test_a_pair_records_both_orientations(arena: Arena) -> None:
    pair = arena.play_duplicate_pair(
        StrongHeuristicAgent(),
        BasicHeuristicAgent(),
        match_seed=evaluation_match_seed(7, 0),
        dealer_seat=0,
    )
    assert pair.first_run.policy_a_team == 0
    assert pair.second_run.policy_a_team == 1
    assert pair.first_run.match_seed == pair.second_run.match_seed
    assert pair.first_run.deal_index == pair.second_run.deal_index
    assert pair.first_run.dealer_seat == pair.second_run.dealer_seat


def test_identical_policies_produce_an_exactly_zero_paired_difference(arena: Arena) -> None:
    """A symmetry check on the pairing itself.

    With two behaviourally identical policies both orientations are the same
    game, so policy A's margin in one run is the exact negative of the other and
    the paired statistic must be exactly zero. A non-zero result here would mean
    the two halves of a pair are not really the same deal.
    """
    for pair_index in range(6):
        pair = arena.play_duplicate_pair(
            BasicHeuristicAgent(),
            BasicHeuristicAgent(),
            match_seed=evaluation_match_seed(3, pair_index),
            dealer_seat=pair_index % NUM_SEATS,
        )
        assert pair.paired_score_difference == 0.0


def test_swapping_the_policy_labels_flips_the_sign(arena: Arena) -> None:
    match_seed = evaluation_match_seed(11, 0)
    forward = arena.play_duplicate_pair(
        StrongHeuristicAgent(), BasicHeuristicAgent(), match_seed=match_seed, dealer_seat=0
    )
    reversed_pair = arena.play_duplicate_pair(
        BasicHeuristicAgent(), StrongHeuristicAgent(), match_seed=match_seed, dealer_seat=0
    )
    assert forward.paired_score_difference == pytest.approx(-reversed_pair.paired_score_difference)


# --- the sign convention ----------------------------------------------------


def _make_outcome(policy_a_team: int, score_by_team: tuple[int, int]) -> DealOutcome:
    return DealOutcome(
        match_seed=1,
        deal_index=0,
        dealer_seat=0,
        policy_a_team=policy_a_team,
        score_change_by_team=score_by_team,
        declarer_seat=0,
        winning_bid=9,
        tricks_won_by_team=(9, 4),
        contract_made=True,
        passed_out=False,
        decision_count=57,
        bid_action_count=1,
        pass_action_count=3,
    )


def test_paired_difference_is_positive_when_policy_a_scores_more() -> None:
    """Hand-built: A takes +9 as team 0, then +9 as team 1. Difference = +9."""
    pair = PairedOutcome(
        match_seed=1,
        deal_index=0,
        first_run=_make_outcome(policy_a_team=0, score_by_team=(9, 0)),
        second_run=_make_outcome(policy_a_team=1, score_by_team=(0, 9)),
    )
    assert pair.paired_score_difference == pytest.approx(9.0)
    assert pair.policy_a_wins


def test_paired_difference_is_negative_when_policy_b_scores_more() -> None:
    pair = PairedOutcome(
        match_seed=1,
        deal_index=0,
        first_run=_make_outcome(policy_a_team=0, score_by_team=(0, 9)),
        second_run=_make_outcome(policy_a_team=1, score_by_team=(9, 0)),
    )
    assert pair.paired_score_difference == pytest.approx(-9.0)
    assert not pair.policy_a_wins


def test_paired_difference_averages_the_two_orientations() -> None:
    """A wins big in one seating and loses a little in the other."""
    pair = PairedOutcome(
        match_seed=1,
        deal_index=0,
        first_run=_make_outcome(policy_a_team=0, score_by_team=(20, 0)),
        second_run=_make_outcome(policy_a_team=1, score_by_team=(6, -4)),
    )
    # first margin +20, second margin (-4) - 6 = -10, mean = +5
    assert pair.paired_score_difference == pytest.approx(5.0)


def test_policy_scores_follow_the_team_assignment() -> None:
    outcome = _make_outcome(policy_a_team=1, score_by_team=(3, 11))
    assert outcome.policy_a_score == 11
    assert outcome.policy_b_score == 3


def test_a_tie_is_neither_a_win_nor_a_loss() -> None:
    pair = PairedOutcome(
        match_seed=1,
        deal_index=0,
        first_run=_make_outcome(policy_a_team=0, score_by_team=(7, 7)),
        second_run=_make_outcome(policy_a_team=1, score_by_team=(7, 7)),
    )
    assert pair.is_tie
    assert not pair.policy_a_wins


# --- rotations --------------------------------------------------------------


def test_evaluation_rotates_the_dealer_across_pairs(arena: Arena) -> None:
    result = arena.evaluate(
        StrongHeuristicAgent(),
        BasicHeuristicAgent(),
        paired_deal_count=8,
        evaluation_seed=5,
        bootstrap_resamples=200,
        replay_sample_rate=0.0,
    )
    dealers = [pair.first_run.dealer_seat for pair in result.paired_outcomes]
    assert dealers == [0, 1, 2, 3, 0, 1, 2, 3]
    assert set(dealers) == set(range(NUM_SEATS))


def test_each_policy_holds_each_team_equally_often(arena: Arena) -> None:
    result = arena.evaluate(
        StrongHeuristicAgent(),
        BasicHeuristicAgent(),
        paired_deal_count=10,
        evaluation_seed=6,
        bootstrap_resamples=200,
        replay_sample_rate=0.0,
    )
    team_counts = {0: 0, 1: 0}
    for pair in result.paired_outcomes:
        team_counts[pair.first_run.policy_a_team] += 1
        team_counts[pair.second_run.policy_a_team] += 1
    assert team_counts[0] == team_counts[1] == 10


def test_each_pair_uses_a_distinct_match_seed(arena: Arena) -> None:
    result = arena.evaluate(
        RandomAgent(seed=1),
        RandomAgent(seed=2),
        paired_deal_count=25,
        evaluation_seed=9,
        bootstrap_resamples=200,
        replay_sample_rate=0.0,
    )
    seeds = [pair.match_seed for pair in result.paired_outcomes]
    assert len(set(seeds)) == 25


def test_evaluation_is_reproducible(arena: Arena) -> None:
    first = arena.evaluate(
        StrongHeuristicAgent(),
        BasicHeuristicAgent(),
        paired_deal_count=12,
        evaluation_seed=77,
        bootstrap_resamples=500,
        replay_sample_rate=0.0,
    )
    second = Arena(replay_sample_rate=0.0).evaluate(
        StrongHeuristicAgent(),
        BasicHeuristicAgent(),
        paired_deal_count=12,
        evaluation_seed=77,
        bootstrap_resamples=500,
        replay_sample_rate=0.0,
    )
    first_differences = [pair.paired_score_difference for pair in first.paired_outcomes]
    second_differences = [pair.paired_score_difference for pair in second.paired_outcomes]
    assert first_differences == second_differences
    assert first.metrics["mean_paired_score_difference"] == pytest.approx(
        second.metrics["mean_paired_score_difference"]
    )
    assert (
        first.metrics["paired_score_difference_ci"] == second.metrics["paired_score_difference_ci"]
    )


# --- the arena implements no rules ------------------------------------------


def test_illegal_action_rate_is_zero(arena: Arena) -> None:
    result = arena.evaluate(
        RandomAgent(seed=3),
        StrongHeuristicAgent(),
        paired_deal_count=15,
        evaluation_seed=13,
        bootstrap_resamples=200,
        replay_sample_rate=0.0,
    )
    assert result.metrics["illegal_action_rate"] == 0.0
    assert result.metrics["no_trump_action_rate"] == 0.0
    assert arena.illegal_action_count == 0


def test_replay_verification_passes_on_the_sampled_deals() -> None:
    arena = Arena(replay_sample_rate=1.0)
    result = arena.evaluate(
        StrongHeuristicAgent(),
        BasicHeuristicAgent(),
        paired_deal_count=6,
        evaluation_seed=17,
        bootstrap_resamples=200,
    )
    assert result.metrics["replay_integrity_rate"] == 1.0
    assert result.metrics["replay_verified_count"] == 12


def test_passed_out_deals_score_zero_for_both_teams(arena: Arena) -> None:
    """A hand nobody bids on is worth nothing to either side."""

    class AlwaysPassAgent(RandomAgent):
        def act(self, view):  # type: ignore[override]
            from tarneeb.core.actions import PASS_ACTION

            if view.legal_action_mask[PASS_ACTION]:
                return PASS_ACTION
            return int(np.flatnonzero(view.legal_action_mask)[0])

    outcome = arena.play_deal(
        build_seat_assignment(AlwaysPassAgent(), AlwaysPassAgent(), policy_a_team=0),
        match_seed=evaluation_match_seed(1, 0),
        dealer_seat=0,
    )
    assert outcome.passed_out
    assert outcome.score_change_by_team == (0, 0)
    assert outcome.declarer_seat is None
    assert outcome.winning_bid is None
    assert outcome.contract_made is None


def test_deal_outcome_roles_follow_the_declarer() -> None:
    outcome = _make_outcome(policy_a_team=0, score_by_team=(9, 0))  # declarer seat 0
    assert outcome.role_of_seat(0) == "declarer"
    assert outcome.role_of_seat(2) == "partner"
    assert outcome.role_of_seat(1) == "defender"
    assert outcome.role_of_seat(3) == "defender"
    assert team_of_seat(0) == team_of_seat(2)
