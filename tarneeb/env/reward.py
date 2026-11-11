"""Deal reward for the primary training configuration.

One episode is one deal. The reward is the normalised team score difference:

``reward(team) = (team_score_change - opponent_score_change) / 42``

The score changes come from the engine's :class:`~tarneeb.core.events.DealResult`,
which is produced by the frozen Phase 1 ``score_deal``. Nothing here reimplements
scoring, and 42 is the proved ``|score difference|`` bound from docs/RULES.md
section 5.3, so the reward always lies in ``[-1, +1]``.

Both partners receive exactly the same reward and the opponents receive its
negative. There is no intermediate reward in the primary configuration, and no
penalty for illegal actions -- illegality is prevented by masking.
"""

from __future__ import annotations

import numpy as np

from tarneeb.core.events import DealResult
from tarneeb.core.state import NUM_SEATS, TarneebState, team_of_seat
from tarneeb.env.spaces import MAX_ABSOLUTE_SCORE_DIFFERENCE

NUM_TEAMS = 2


def team_score_changes(deal_result: DealResult, declaring_team: int) -> tuple[int, int]:
    """Return ``(change_for_team_0, change_for_team_1)`` from a scored deal."""
    if declaring_team not in (0, 1):
        raise ValueError(f"declaring_team must be 0 or 1, got {declaring_team}")
    if declaring_team == 0:
        return (
            deal_result.score_delta_bidding_team,
            deal_result.score_delta_defending_team,
        )
    return (
        deal_result.score_delta_defending_team,
        deal_result.score_delta_bidding_team,
    )


def normalized_team_reward(team_score_change: int, opponent_score_change: int) -> float:
    """Return the normalised score difference, guaranteed to lie in ``[-1, +1]``."""
    score_difference = team_score_change - opponent_score_change
    if abs(score_difference) > MAX_ABSOLUTE_SCORE_DIFFERENCE:
        raise ValueError(
            f"score difference {score_difference} exceeds the frozen bound "
            f"{MAX_ABSOLUTE_SCORE_DIFFERENCE}; the rules or scoring changed"
        )
    return score_difference / MAX_ABSOLUTE_SCORE_DIFFERENCE


def deal_rewards_by_team(deal_result: DealResult, declaring_team: int) -> np.ndarray:
    """Return the ``(2,)`` per-team rewards for a completed deal."""
    change_for_team_0, change_for_team_1 = team_score_changes(deal_result, declaring_team)
    return np.array(
        [
            normalized_team_reward(change_for_team_0, change_for_team_1),
            normalized_team_reward(change_for_team_1, change_for_team_0),
        ],
        dtype=np.float32,
    )


def deal_rewards_by_seat(deal_result: DealResult, declaring_team: int) -> np.ndarray:
    """Return the ``(4,)`` per-seat rewards for a completed deal.

    Partners share a seat-independent team reward, so seats 0 and 2 always match,
    as do seats 1 and 3, and the two pairs are exact negatives of each other.
    """
    rewards_by_team = deal_rewards_by_team(deal_result, declaring_team)
    return np.array(
        [rewards_by_team[team_of_seat(seat)] for seat in range(NUM_SEATS)],
        dtype=np.float32,
    )


def zero_rewards_by_seat() -> np.ndarray:
    """Return the ``(4,)`` all-zero reward emitted by every non-terminal step."""
    return np.zeros(NUM_SEATS, dtype=np.float32)


def deal_rewards_for_state(state: TarneebState) -> np.ndarray:
    """Return the ``(4,)`` per-seat rewards for a scored deal state."""
    if state.deal_result is None or state.declaring_team is None:
        raise ValueError("the deal has not been scored yet")
    return deal_rewards_by_seat(state.deal_result, state.declaring_team)
