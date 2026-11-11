"""Deal reward tests.

Expected values come from the frozen scoring rules in docs/RULES.md, divided by
the proved bound of 42. They are written out by hand; none is produced by the
reward code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.core.actions import PASS_ACTION, action_for_bid, action_for_trump_suit
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import DealResult, Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.core.scoring import score_deal
from tarneeb.core.state import NUM_SEATS, team_of_seat
from tarneeb.env.reward import (
    deal_rewards_by_seat,
    deal_rewards_by_team,
    deal_rewards_for_state,
    normalized_team_reward,
    team_score_changes,
    zero_rewards_by_seat,
)
from tarneeb.env.spaces import MAX_ABSOLUTE_SCORE_DIFFERENCE

RULES = RulesConfig()


@pytest.fixture
def engine() -> TarneebEngine:
    return TarneebEngine(RULES)


# --- the formula ------------------------------------------------------------


def test_normalisation_constant_is_the_frozen_bound() -> None:
    assert MAX_ABSOLUTE_SCORE_DIFFERENCE == 42


@pytest.mark.parametrize(
    ("team_score_change", "opponent_score_change", "expected_reward"),
    [
        (26, 0, 26 / 42),  # bid 13 made: the maximum
        (-16, 26, -1.0),  # bid 13 failed, defenders take all 13: the minimum
        (26, -16, 1.0),  # the defenders' view of that same deal
        (9, 0, 9 / 42),  # a made bid of 9
        (-9, 4, -13 / 42),  # a failed bid of 9, defenders take 4
        (16, 0, 16 / 42),  # kaboot
        (0, 0, 0.0),
    ],
)
def test_normalized_team_reward(
    team_score_change: int, opponent_score_change: int, expected_reward: float
) -> None:
    assert normalized_team_reward(team_score_change, opponent_score_change) == pytest.approx(
        expected_reward
    )


def test_reward_rejects_an_impossible_score_difference() -> None:
    """A difference beyond 42 means the rules changed; fail loudly."""
    with pytest.raises(ValueError, match="exceeds the frozen bound"):
        normalized_team_reward(100, 0)


# --- range over the whole frozen scoring domain -----------------------------


def test_reward_range_over_all_98_scoring_outcomes() -> None:
    observed_rewards = []
    for bid in range(7, 14):
        for bidding_team_tricks in range(14):
            bidding_change, defending_change = score_deal(bid, bidding_team_tricks, RULES)
            observed_rewards.append(normalized_team_reward(bidding_change, defending_change))
            observed_rewards.append(normalized_team_reward(defending_change, bidding_change))

    assert min(observed_rewards) == pytest.approx(-1.0)
    assert max(observed_rewards) == pytest.approx(1.0)
    assert all(-1.0 <= reward <= 1.0 for reward in observed_rewards)


def test_the_extremes_come_from_the_bid_13_rows() -> None:
    worst_bidding, worst_defending = score_deal(13, 0, RULES)
    assert (worst_bidding, worst_defending) == (-16, 26)
    assert normalized_team_reward(worst_bidding, worst_defending) == pytest.approx(-1.0)

    best_bidding, best_defending = score_deal(13, 13, RULES)
    assert (best_bidding, best_defending) == (26, 0)
    assert normalized_team_reward(best_bidding, best_defending) == pytest.approx(26 / 42)


# --- team mapping -----------------------------------------------------------


def test_team_score_changes_follow_the_declaring_team() -> None:
    deal_result = DealResult(
        bid=9, tricks_bidding_team=9, score_delta_bidding_team=9, score_delta_defending_team=0
    )
    assert team_score_changes(deal_result, declaring_team=0) == (9, 0)
    assert team_score_changes(deal_result, declaring_team=1) == (0, 9)


@pytest.mark.parametrize("declaring_team", [-1, 2])
def test_team_score_changes_rejects_a_bad_team(declaring_team: int) -> None:
    deal_result = DealResult(9, 9, 9, 0)
    with pytest.raises(ValueError, match="declaring_team must be"):
        team_score_changes(deal_result, declaring_team)


def test_rewards_by_team_are_antisymmetric() -> None:
    deal_result = DealResult(
        bid=11, tricks_bidding_team=8, score_delta_bidding_team=-11, score_delta_defending_team=5
    )
    rewards = deal_rewards_by_team(deal_result, declaring_team=0)
    assert rewards.shape == (2,)
    assert rewards.dtype == np.float32
    assert rewards[0] == pytest.approx(-16 / 42)
    assert rewards[1] == pytest.approx(16 / 42)
    assert rewards[0] == pytest.approx(-rewards[1])


# --- per-seat rewards -------------------------------------------------------


def test_partners_receive_equal_rewards_and_opponents_the_negative() -> None:
    deal_result = DealResult(
        bid=9, tricks_bidding_team=9, score_delta_bidding_team=9, score_delta_defending_team=0
    )
    for declaring_team in (0, 1):
        rewards = deal_rewards_by_seat(deal_result, declaring_team)
        assert rewards.shape == (4,)
        assert rewards.dtype == np.float32
        assert rewards[0] == rewards[2]  # partners
        assert rewards[1] == rewards[3]  # partners
        assert rewards[0] == pytest.approx(-rewards[1])
        assert rewards.sum() == pytest.approx(0.0)


def test_the_declaring_team_seats_get_the_bidding_reward() -> None:
    deal_result = DealResult(
        bid=13, tricks_bidding_team=13, score_delta_bidding_team=26, score_delta_defending_team=0
    )
    rewards = deal_rewards_by_seat(deal_result, declaring_team=1)
    assert rewards[1] == pytest.approx(26 / 42)
    assert rewards[3] == pytest.approx(26 / 42)
    assert rewards[0] == pytest.approx(-26 / 42)
    assert rewards[2] == pytest.approx(-26 / 42)


def test_zero_rewards_shape() -> None:
    zero_rewards = zero_rewards_by_seat()
    assert zero_rewards.shape == (4,)
    assert zero_rewards.dtype == np.float32
    assert not zero_rewards.any()


# --- integration with a really played deal ----------------------------------


def test_rewards_for_a_played_out_deal_match_the_engine_result(engine: TarneebEngine) -> None:
    state = engine.deal(engine.new_deal(0, match_seed=42))
    engine.step(state, action_for_bid(9))
    for _ in range(NUM_SEATS - 1):
        engine.step(state, PASS_ACTION)
    engine.step(state, action_for_trump_suit(2))

    rng = np.random.default_rng(0)
    while state.phase is Phase.PLAY:
        engine.step(state, int(rng.choice(np.flatnonzero(engine.legal_actions(state)))))

    rewards = deal_rewards_for_state(state)
    declaring_team = state.declaring_team
    expected_declaring_reward = normalized_team_reward(
        state.deal_result.score_delta_bidding_team,
        state.deal_result.score_delta_defending_team,
    )
    for seat in range(NUM_SEATS):
        expected = (
            expected_declaring_reward
            if team_of_seat(seat) == declaring_team
            else -expected_declaring_reward
        )
        assert rewards[seat] == pytest.approx(expected)


def test_rewards_require_a_scored_deal(engine: TarneebEngine) -> None:
    state = engine.deal(engine.new_deal(0, match_seed=1))
    with pytest.raises(ValueError, match="has not been scored"):
        deal_rewards_for_state(state)


def test_many_played_deals_stay_in_range_and_sum_to_zero(engine: TarneebEngine) -> None:
    for match_seed in range(25):
        state = engine.deal(engine.new_deal(match_seed % 4, match_seed=match_seed))
        engine.step(state, action_for_bid(7 + match_seed % 7))
        for _ in range(NUM_SEATS - 1):
            engine.step(state, PASS_ACTION)
        engine.step(state, action_for_trump_suit(match_seed % 4))

        rng = np.random.default_rng(match_seed)
        while state.phase is Phase.PLAY:
            engine.step(state, int(rng.choice(np.flatnonzero(engine.legal_actions(state)))))

        rewards = deal_rewards_for_state(state)
        assert rewards.min() >= -1.0 and rewards.max() <= 1.0
        assert rewards.sum() == pytest.approx(0.0)
        assert rewards[0] == rewards[2]
        assert rewards[1] == rewards[3]
