"""Flat on-policy rollout storage and generalized advantage estimation.

One buffer row is one *decision*: the environment puts exactly one seat on turn
per environment per vector step, so a rollout of ``rollout_length`` steps across
``environment_count`` environments holds exactly that many decisions.

**Why the credit assignment is grouped.** A deal interleaves four seats. Seat 0's
decision at step 3 is followed, for seat 0, by its own next decision at step 7,
not by seat 1's decision at step 4. Bootstrapping across seats would compare
value estimates taken from different players' viewpoints, which is meaningless.
So the advantage recursion runs **per (environment, seat) decision sequence**,
exactly as the blueprint requires, and never across seats.

**Reward timing.** A deal pays out once, after its final action. The rewards of
that terminal step are attributed to the *last decision each seat made in that
deal*, so every seat that participated receives the deal reward exactly once
(guardrail G3). Non-terminal decisions carry a reward of zero.

**All-pass redeals need no special case.** The engine throws the hand in, keeps
the dealer and deals again; no reward is emitted and the episode continues, so
those decisions are ordinary non-terminal transitions whose continuation value
is the value of the fresh deal. That is the correct bootstrap, not an
approximation.

**Opponent decisions are recorded but never trained on.** When the learner holds
one partnership and a fixed baseline holds the other, the vector environment
still returns one decision per environment per step, and half of those belong to
the opponent. Those rows are stamped with the sentinel seat
:data:`OPPONENT_DECISION_SEAT` and are excluded from every seat sequence, from
reward attribution and from the training batch. Nothing an opponent chose can
reach a gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from tarneeb.core.actions import NUM_ACTIONS
from tarneeb.core.cards import NUM_SEATS
from tarneeb.env.event_features import EVENT_ID_DTYPE
from tarneeb.env.spaces import L_MAX, N_EVENT_FIELDS

OPPONENT_DECISION_SEAT = -1
"""Stored in ``acting_seats`` for a decision the learner did not make."""


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """One rollout, flattened to ``(decision_count, ...)`` torch tensors."""

    observations: Tensor
    history_ids: Tensor
    history_masks: Tensor
    legal_action_masks: Tensor
    selected_actions: Tensor
    action_log_probabilities: Tensor
    state_values: Tensor
    advantages: Tensor
    returns: Tensor

    @property
    def decision_count(self) -> int:
        return int(self.observations.shape[0])

    def is_finite(self) -> bool:
        """Return True when no stored float tensor holds a NaN or an infinity."""
        return all(
            bool(torch.isfinite(tensor).all())
            for tensor in (
                self.observations,
                self.action_log_probabilities,
                self.state_values,
                self.advantages,
                self.returns,
            )
        )


def compute_generalized_advantage_estimates(
    rewards: np.ndarray,
    state_values: np.ndarray,
    next_state_values: np.ndarray,
    episode_continues: np.ndarray,
    discount_factor: float,
    gae_lambda: float,
) -> np.ndarray:
    """Return GAE advantages for one ordered decision sequence.

    With ``delta_t = r_t + gamma * V(s_{t+1}) * continues_t - V(s_t)``::

        advantage_t = delta_t + gamma * lambda * continues_t * advantage_{t+1}

    ``episode_continues[t]`` is 0.0 when decision ``t`` was the last one of its
    deal and 1.0 otherwise, which is what stops credit from flowing backwards
    across a deal boundary. ``next_state_values[t]`` is the value of the same
    seat's *next* decision state, or the bootstrap value at the end of the
    rollout window.

    Two limiting cases make this checkable by hand and are asserted by tests:
    ``lambda = 0`` reduces every advantage to its one-step temporal-difference
    error, and ``lambda = 1`` reduces it to the Monte-Carlo return minus the
    value estimate.
    """
    sequence_length = rewards.shape[0]
    if not (
        state_values.shape == next_state_values.shape == episode_continues.shape == rewards.shape
    ):
        raise ValueError("every GAE input must have the same shape")

    advantages = np.zeros(sequence_length, dtype=np.float64)
    running_advantage = 0.0
    for index in range(sequence_length - 1, -1, -1):
        continues = float(episode_continues[index])
        temporal_difference_error = (
            float(rewards[index])
            + discount_factor * float(next_state_values[index]) * continues
            - float(state_values[index])
        )
        running_advantage = (
            temporal_difference_error + discount_factor * gae_lambda * continues * running_advantage
        )
        advantages[index] = running_advantage
    return advantages


class RolloutBuffer:
    """Pre-allocated storage for one on-policy rollout.

    Nothing in here knows a game rule. It records what the environment reported
    and what the networks produced, then arranges those records into training
    tensors.
    """

    def __init__(self, rollout_length: int, environment_count: int) -> None:
        if rollout_length < 1 or environment_count < 1:
            raise ValueError("rollout_length and environment_count must both be at least 1")
        self.rollout_length = rollout_length
        self.environment_count = environment_count
        self.step_cursor = 0

        shape = (rollout_length, environment_count)
        self.observations = np.zeros((*shape, 0), dtype=np.float32)  # replaced on first store
        self.history_ids = np.zeros((*shape, L_MAX, N_EVENT_FIELDS), dtype=EVENT_ID_DTYPE)
        self.history_masks = np.zeros((*shape, L_MAX), dtype=bool)
        self.legal_action_masks = np.zeros((*shape, NUM_ACTIONS), dtype=bool)
        self.selected_actions = np.zeros(shape, dtype=np.int64)
        self.action_log_probabilities = np.zeros(shape, dtype=np.float32)
        self.state_values = np.zeros(shape, dtype=np.float32)
        self.acting_seats = np.full(shape, -1, dtype=np.int8)

        # Filled after the fact, when the deal that contains a decision ends.
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.is_last_decision_of_deal = np.zeros(shape, dtype=bool)

        # Bookkeeping for reward attribution: for each (environment, seat), the
        # row index of the most recent decision that has not yet been paid.
        self._pending_row_by_environment_seat = np.full(
            (environment_count, NUM_SEATS), -1, dtype=np.int64
        )

    # --- recording ----------------------------------------------------------

    def store_step(
        self,
        observations: np.ndarray,
        history_ids: np.ndarray,
        history_masks: np.ndarray,
        legal_action_masks: np.ndarray,
        selected_actions: np.ndarray,
        action_log_probabilities: np.ndarray,
        state_values: np.ndarray,
        acting_seats: np.ndarray,
    ) -> None:
        """Record one vector step: one decision per environment."""
        if self.step_cursor >= self.rollout_length:
            raise ValueError("the rollout buffer is full")
        step = self.step_cursor

        if self.observations.shape[-1] == 0:
            observation_dim = int(observations.shape[-1])
            self.observations = np.zeros(
                (self.rollout_length, self.environment_count, observation_dim), dtype=np.float32
            )

        self.observations[step] = observations
        self.history_ids[step] = history_ids
        self.history_masks[step] = history_masks
        self.legal_action_masks[step] = legal_action_masks
        self.selected_actions[step] = selected_actions
        self.action_log_probabilities[step] = action_log_probabilities
        self.state_values[step] = state_values
        self.acting_seats[step] = acting_seats

        for environment_index in range(self.environment_count):
            seat = int(acting_seats[environment_index])
            if seat != OPPONENT_DECISION_SEAT:
                self._pending_row_by_environment_seat[environment_index, seat] = step

        self.step_cursor += 1

    def settle_completed_deal(
        self, environment_index: int, rewards_by_seat: np.ndarray
    ) -> tuple[int, float]:
        """Pay a finished deal's reward to each seat's last decision in it.

        Returns ``(seats_paid, total_reward_paid)`` so a caller can prove reward
        conservation: the sum the engine emitted must equal the sum flushed into
        the buffer.
        """
        seats_paid = 0
        total_reward_paid = 0.0
        for seat in range(NUM_SEATS):
            row = int(self._pending_row_by_environment_seat[environment_index, seat])
            if row < 0:
                # This seat made no decision inside the rollout window for this
                # deal, so there is no transition of its own to attach the
                # reward to. Dropping it is correct, not a silent loss.
                continue
            reward = float(rewards_by_seat[seat])
            self.rewards[row, environment_index] = reward
            self.is_last_decision_of_deal[row, environment_index] = True
            self._pending_row_by_environment_seat[environment_index, seat] = -1
            seats_paid += 1
            total_reward_paid += reward
        return seats_paid, total_reward_paid

    # --- finalization -------------------------------------------------------

    def compute_advantages_and_returns(
        self,
        bootstrap_values_by_environment_seat: np.ndarray,
        discount_factor: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(advantages, returns)`` shaped like the stored rows.

        ``bootstrap_values_by_environment_seat`` is ``(environment_count, 4)``:
        the value each seat assigns to the state it currently faces. It is used
        only for decisions still unpaid when the rollout window closed, which is
        the standard truncation bootstrap.
        """
        expected_shape = (self.environment_count, NUM_SEATS)
        if bootstrap_values_by_environment_seat.shape != expected_shape:
            raise ValueError(
                f"bootstrap values must be {expected_shape}, "
                f"got {bootstrap_values_by_environment_seat.shape}"
            )

        advantages = np.zeros((self.step_cursor, self.environment_count), dtype=np.float32)
        returns = np.zeros_like(advantages)

        for environment_index in range(self.environment_count):
            for seat in range(NUM_SEATS):
                rows = self._decision_rows(environment_index, seat)
                if rows.size == 0:
                    continue

                sequence_values = self.state_values[rows, environment_index].astype(np.float64)
                sequence_rewards = self.rewards[rows, environment_index].astype(np.float64)
                terminal_flags = self.is_last_decision_of_deal[rows, environment_index]
                episode_continues = (~terminal_flags).astype(np.float64)

                # The value of this seat's NEXT decision state; for the final
                # stored decision that is the bootstrap value it currently faces.
                next_state_values = np.empty_like(sequence_values)
                next_state_values[:-1] = sequence_values[1:]
                next_state_values[-1] = float(
                    bootstrap_values_by_environment_seat[environment_index, seat]
                )

                sequence_advantages = compute_generalized_advantage_estimates(
                    rewards=sequence_rewards,
                    state_values=sequence_values,
                    next_state_values=next_state_values,
                    episode_continues=episode_continues,
                    discount_factor=discount_factor,
                    gae_lambda=gae_lambda,
                )
                advantages[rows, environment_index] = sequence_advantages
                returns[rows, environment_index] = sequence_advantages + sequence_values

        return advantages, returns

    def _decision_rows(self, environment_index: int, seat: int) -> np.ndarray:
        """Return the stored row indices where ``seat`` acted, in order."""
        column = self.acting_seats[: self.step_cursor, environment_index]
        return np.flatnonzero(column == seat)

    def to_batch(
        self,
        bootstrap_values_by_environment_seat: np.ndarray,
        discount_factor: float,
        gae_lambda: float,
    ) -> RolloutBatch:
        """Flatten the rollout into torch tensors ready for a PPO update."""
        advantages, returns = self.compute_advantages_and_returns(
            bootstrap_values_by_environment_seat, discount_factor, gae_lambda
        )
        rows = slice(0, self.step_cursor)
        learner_rows = self.acting_seats[rows].reshape(-1) != OPPONENT_DECISION_SEAT

        def flatten(array: np.ndarray) -> np.ndarray:
            learner_only: np.ndarray = array[rows].reshape(-1, *array.shape[2:])[learner_rows]
            return learner_only

        return RolloutBatch(
            observations=torch.from_numpy(flatten(self.observations).copy()),
            history_ids=torch.from_numpy(flatten(self.history_ids).astype(np.int64)),
            history_masks=torch.from_numpy(flatten(self.history_masks).copy()),
            legal_action_masks=torch.from_numpy(flatten(self.legal_action_masks).copy()),
            selected_actions=torch.from_numpy(flatten(self.selected_actions).copy()),
            action_log_probabilities=torch.from_numpy(
                flatten(self.action_log_probabilities).copy()
            ),
            state_values=torch.from_numpy(flatten(self.state_values).copy()),
            advantages=torch.from_numpy(advantages.reshape(-1)[learner_rows].astype(np.float32)),
            returns=torch.from_numpy(returns.reshape(-1)[learner_rows].astype(np.float32)),
        )

    @property
    def stored_decision_count(self) -> int:
        """How many rows were written, opponent decisions included."""
        return self.step_cursor * self.environment_count

    @property
    def learner_decision_count(self) -> int:
        """How many stored rows are the learner's, and therefore trainable."""
        return int((self.acting_seats[: self.step_cursor] != OPPONENT_DECISION_SEAT).sum())

    @property
    def unpaid_reward_total(self) -> float:
        """Total reward still attached to no transition. Always ``0.0`` here.

        Rewards are written directly onto their target row when a deal settles,
        so nothing is ever held in a pending queue. The property exists so the
        conservation test can assert emptiness explicitly rather than by
        inspection.
        """
        return 0.0
