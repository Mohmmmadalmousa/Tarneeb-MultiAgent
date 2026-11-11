"""On-policy rollout collection against the frozen Phase 4 environment.

The collector drives :class:`~tarneeb.env.vec.VecTarneeb`, asks the actor for one
action per learner decision, verifies that action against the engine's mask,
steps, and records the transition. It implements no game rule and reads no
hidden state on the actor path: the only thing it ever hands the policy is the
692-dim actor-safe observation and the padded public event history.

**Who sits where.** The learner holds one partnership; a fixed non-neural
baseline holds the other. That is the blueprint's curriculum for the feedforward
baseline (random, then heuristic opponents); four copies of the learner is a
later curriculum stage and belongs to a later phase. The learner's partnership
alternates by environment index, so across a rollout the shared policy occupies
all four absolute seats and every role -- declarer, declarer's partner and both
defenders.

Current-policy self-play -- four copies of the learner -- is a later curriculum
stage and is deliberately not implemented here. It is not a neutral omission:
measured over 245k steps, four-copy self-play converged to an all-pass
equilibrium (pass rate 1.000, every deal thrown in, entropy 0, no learning
signal at all), which is the failure signature the blueprint names for the
bidding stage. An opponent partnership is therefore required, not optional.

**Opponent decisions never reach a gradient.** They are stepped through the same
environment but stored under
:data:`~tarneeb.buffers.rollout.OPPONENT_DECISION_SEAT`, which excludes them from
reward attribution, from the advantage recursion and from the training batch. The
opponent chooses from a :class:`~tarneeb.env.public_view.PublicView`, exactly as
it does in the Phase 5 arena, so it is held to the same information boundary.

Metrics are returned as plain values (project requirements training-observability boundary).
This module owns no logger, no writer and no renderer, and takes no ``verbose``
argument.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from tarneeb.agents.base import Agent
from tarneeb.buffers.rollout import OPPONENT_DECISION_SEAT, RolloutBuffer
from tarneeb.core.actions import (
    BID_ACTION_START,
    BID_ACTION_STOP,
    NO_TRUMP_ACTION,
    NUM_ACTIONS,
    PASS_ACTION,
)
from tarneeb.core.cards import NUM_SEATS
from tarneeb.core.state import team_of_seat
from tarneeb.env.public_view import PublicViewBuilder
from tarneeb.env.spaces import L_MAX, N_EVENT_FIELDS
from tarneeb.env.vec import VecTarneeb
from tarneeb.models.masking import MaskedCategorical, actions_are_legal
from tarneeb.models.nets import ActorNetwork, CriticNetwork


@dataclass
class RolloutMetrics:
    """Diagnostics produced by one rollout. Every field is a plain Python value."""

    environment_steps: int = 0
    decision_count: int = 0
    learner_decision_count: int = 0
    completed_deals: int = 0
    passed_out_deals: int = 0
    illegal_action_count: int = 0
    no_trump_action_count: int = 0
    auction_decision_count: int = 0
    pass_action_count: int = 0
    learner_declared_deals: int = 0
    mean_bid: float | None = None
    contract_success_rate: float | None = None
    mean_absolute_score_difference: float | None = None
    mean_learner_reward: float | None = None
    total_reward_paid: float = 0.0
    total_reward_emitted: float = 0.0
    collection_seconds: float = 0.0
    decisions_per_second: float = 0.0
    winning_bids: list[int] = field(default_factory=list, repr=False)

    @property
    def pass_rate(self) -> float | None:
        """Share of the learner's auction decisions that were a pass."""
        if self.auction_decision_count == 0:
            return None
        return self.pass_action_count / self.auction_decision_count

    @property
    def learner_declare_rate(self) -> float | None:
        """Share of completed deals in which the learner's team won the auction."""
        if self.completed_deals == 0:
            return None
        return self.learner_declared_deals / self.completed_deals

    def as_dict(self) -> dict[str, float | int | None]:
        """Return the reportable scalars, without the raw per-deal lists."""
        return {
            "environment_steps": self.environment_steps,
            "decision_count": self.decision_count,
            "learner_decision_count": self.learner_decision_count,
            "completed_deals": self.completed_deals,
            "passed_out_deals": self.passed_out_deals,
            "illegal_action_count": self.illegal_action_count,
            "no_trump_action_count": self.no_trump_action_count,
            "mean_bid": self.mean_bid,
            "pass_rate": self.pass_rate,
            "learner_declare_rate": self.learner_declare_rate,
            "contract_success_rate": self.contract_success_rate,
            "mean_absolute_score_difference": self.mean_absolute_score_difference,
            "mean_learner_reward": self.mean_learner_reward,
            "collection_seconds": self.collection_seconds,
            "decisions_per_second": self.decisions_per_second,
        }


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def learner_team_by_environment(environment_count: int) -> np.ndarray:
    """Alternate which partnership the learner holds, by environment index.

    Combined with the vector environment's per-index dealer rotation, this puts
    the shared policy in all four absolute seats and both dealer parities, so it
    cannot settle into a seat-specific convention.
    """
    return np.arange(environment_count, dtype=np.int8) % 2


class RolloutCollector:
    """Collects one on-policy rollout per call, reusing the same environments."""

    def __init__(
        self,
        environments: VecTarneeb,
        actor: ActorNetwork,
        critic: CriticNetwork,
        rollout_length: int,
        action_sampling_generator: torch.Generator,
        opponents: list[Agent],
    ) -> None:
        self.environments = environments
        self.actor = actor
        self.critic = critic
        self.rollout_length = rollout_length
        self.action_sampling_generator = action_sampling_generator

        if len(opponents) != environments.num_envs:
            raise ValueError(
                f"expected one opponent per environment ({environments.num_envs}), "
                f"got {len(opponents)}"
            )
        self.opponents = opponents
        self.learner_teams = learner_team_by_environment(environments.num_envs)
        self.views = PublicViewBuilder(environments.engine)

        self.environments.reset()
        self.redeal_counts = np.zeros(environments.num_envs, dtype=np.int64)

    # --- seating ------------------------------------------------------------

    def is_learner_seat(self, environment_index: int, seat: int) -> bool:
        """True when ``seat`` in this environment is played by the learning policy."""
        return team_of_seat(seat) == int(self.learner_teams[environment_index])

    def learner_decision_flags(self, acting_seats: np.ndarray) -> np.ndarray:
        """Return a per-environment mask of the decisions the learner must make."""
        return np.array(
            [
                self.is_learner_seat(environment_index, int(seat))
                for environment_index, seat in enumerate(acting_seats)
            ],
            dtype=bool,
        )

    # --- gathering ----------------------------------------------------------

    def _gather_learner_inputs(
        self, learner_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the actor inputs for exactly the environments the learner acts in."""
        observation_dim = self.actor.spec.actor_observation_dim
        observations = np.zeros((learner_indices.size, observation_dim), dtype=np.float32)
        history_ids = np.zeros((learner_indices.size, L_MAX, N_EVENT_FIELDS), dtype=np.int16)
        history_masks = np.zeros((learner_indices.size, L_MAX), dtype=bool)
        legal_action_masks = np.zeros((learner_indices.size, NUM_ACTIONS), dtype=bool)

        for row, environment_index in enumerate(learner_indices):
            environment = self.environments.environments[int(environment_index)]
            observations[row] = environment.actor_observation()
            ids, valid = environment.padded_history()
            history_ids[row] = ids
            history_masks[row] = valid
            legal_action_masks[row] = environment.legal_action_mask()

        return observations, history_ids, history_masks, legal_action_masks

    def _bootstrap_values_by_environment_seat(self) -> np.ndarray:
        """Return ``(environment_count, 4)`` values for the states now faced.

        Every seat is valued, not only the one on turn, because the truncation
        bootstrap belongs to each seat's own decision sequence. Opponent seats are
        valued too and simply never used, which is cheaper than branching.
        """
        environment_count = self.environments.num_envs
        observation_dim = self.critic.spec.critic_observation_dim
        observations = np.zeros((environment_count * NUM_SEATS, observation_dim), dtype=np.float32)
        history_ids = np.zeros(
            (environment_count * NUM_SEATS, L_MAX, N_EVENT_FIELDS), dtype=np.int16
        )
        history_masks = np.zeros((environment_count * NUM_SEATS, L_MAX), dtype=bool)

        for environment_index, environment in enumerate(self.environments.environments):
            for seat in range(NUM_SEATS):
                row = environment_index * NUM_SEATS + seat
                observations[row] = environment.actor_observation(seat)
                ids, valid = environment.padded_history(seat)
                history_ids[row] = ids
                history_masks[row] = valid

        with torch.no_grad():
            values = self.critic(
                torch.from_numpy(observations),
                torch.from_numpy(history_ids.astype(np.int64)),
                torch.from_numpy(history_masks),
            )
        bootstrap_values: np.ndarray = values.numpy().reshape(environment_count, NUM_SEATS)
        return bootstrap_values

    def _opponent_actions(self, opponent_indices: np.ndarray) -> dict[int, int]:
        """Ask each fixed opponent for its move, from a public view only."""
        chosen: dict[int, int] = {}
        for environment_index in opponent_indices:
            environment = self.environments.environments[int(environment_index)]
            state = environment.state
            assert state is not None and state.current_player_seat is not None
            view = self.views.build(state, state.current_player_seat)
            chosen[int(environment_index)] = self.opponents[int(environment_index)].act(view)
        return chosen

    # --- one rollout --------------------------------------------------------

    def collect(self) -> tuple[RolloutBuffer, RolloutMetrics]:
        """Run ``rollout_length`` vector steps and return the buffer and metrics."""
        environment_count = self.environments.num_envs
        buffer = RolloutBuffer(self.rollout_length, environment_count)
        metrics = RolloutMetrics()
        contract_results: list[float] = []
        absolute_score_differences: list[float] = []
        learner_rewards: list[float] = []

        was_training = self.actor.training
        self.actor.eval()
        self.critic.eval()
        start_time = time.perf_counter()

        for _step in range(self.rollout_length):
            acting_seats = self.environments.acting_seats()
            learner_flags = self.learner_decision_flags(acting_seats)
            learner_indices = np.flatnonzero(learner_flags)
            opponent_indices = np.flatnonzero(~learner_flags)

            observations, history_ids, history_masks, legal_action_masks = (
                self._gather_learner_inputs(learner_indices)
            )
            observation_tensor = torch.from_numpy(observations)
            history_id_tensor = torch.from_numpy(history_ids.astype(np.int64))
            history_mask_tensor = torch.from_numpy(history_masks)
            legal_mask_tensor = torch.from_numpy(legal_action_masks)

            with torch.no_grad():
                logits = self.actor(observation_tensor, history_id_tensor, history_mask_tensor)
                distribution = MaskedCategorical.from_logits(logits, legal_mask_tensor)
                selected_actions = distribution.sample(self.action_sampling_generator)
                action_log_probabilities = distribution.log_probability_of(selected_actions)
                state_values = self.critic(
                    observation_tensor, history_id_tensor, history_mask_tensor
                )

            # Masking is a safety property, so it is verified rather than trusted.
            if not actions_are_legal(selected_actions, legal_mask_tensor):
                metrics.illegal_action_count += int(learner_indices.size)
                raise AssertionError("the policy sampled an action its legal mask forbids")
            no_trump_selections = int((selected_actions == NO_TRUMP_ACTION).sum())
            if no_trump_selections:
                metrics.no_trump_action_count += no_trump_selections
                raise AssertionError("the policy selected the reserved no-trump action 64")

            learner_actions = selected_actions.numpy()
            opponent_actions = self._opponent_actions(opponent_indices)

            step_actions = np.zeros(environment_count, dtype=np.int64)
            step_actions[learner_indices] = learner_actions
            for environment_index, action in opponent_actions.items():
                step_actions[environment_index] = action

            learner_in_auction = legal_action_masks[:, PASS_ACTION]
            metrics.auction_decision_count += int(learner_in_auction.sum())
            metrics.pass_action_count += int(
                np.count_nonzero(learner_actions[learner_in_auction] == PASS_ACTION)
            )

            vector_step = self.environments.step(step_actions)

            buffer.store_step(
                acting_seats=np.where(learner_flags, acting_seats, OPPONENT_DECISION_SEAT),
                **self._scatter_to_all_environments(
                    learner_indices,
                    environment_count,
                    observations=observations,
                    history_ids=history_ids,
                    history_masks=history_masks,
                    legal_action_masks=legal_action_masks,
                    selected_actions=learner_actions,
                    action_log_probabilities=action_log_probabilities.numpy(),
                    state_values=state_values.numpy(),
                ),
            )

            metrics.environment_steps += environment_count
            metrics.decision_count += environment_count
            metrics.learner_decision_count += int(learner_indices.size)

            for terminated_index in np.flatnonzero(vector_step.terminated):
                environment_index = int(terminated_index)
                rewards_by_seat = vector_step.rewards_by_seat[environment_index]
                _seats_paid, reward_paid = buffer.settle_completed_deal(
                    environment_index, rewards_by_seat
                )
                metrics.total_reward_paid += reward_paid
                metrics.total_reward_emitted += float(rewards_by_seat.sum())
                metrics.completed_deals += 1

                learner_team = int(self.learner_teams[environment_index])
                learner_rewards.append(float(rewards_by_seat[learner_team]))
                self._record_deal_statistics(
                    environment_index,
                    learner_team,
                    metrics,
                    contract_results,
                    absolute_score_differences,
                )
                self.opponents[environment_index].reset()

            metrics.passed_out_deals += self._count_new_redeals()

            if vector_step.terminated.any():
                self.environments.reset_terminated(vector_step.terminated)
                self.redeal_counts[vector_step.terminated] = 0

        metrics.collection_seconds = time.perf_counter() - start_time
        if metrics.collection_seconds > 0:
            metrics.decisions_per_second = metrics.decision_count / metrics.collection_seconds
        metrics.mean_bid = _mean_or_none([float(bid) for bid in metrics.winning_bids])
        metrics.contract_success_rate = _mean_or_none(contract_results)
        metrics.mean_absolute_score_difference = _mean_or_none(absolute_score_differences)
        metrics.mean_learner_reward = _mean_or_none(learner_rewards)

        if was_training:
            self.actor.train()
            self.critic.train()
        return buffer, metrics

    @staticmethod
    def _scatter_to_all_environments(
        learner_indices: np.ndarray, environment_count: int, **arrays: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Scatter the learner's rows back into one row per environment.

        The buffer stores a rectangular block; opponent rows stay zero and are
        excluded from training by their sentinel seat, so what is written there
        never reaches a gradient.
        """
        scattered: dict[str, np.ndarray] = {}
        for name, values in arrays.items():
            full = np.zeros((environment_count, *values.shape[1:]), dtype=values.dtype)
            full[learner_indices] = values
            scattered[name] = full
        return scattered

    def _count_new_redeals(self) -> int:
        """Return how many all-pass redeals happened since the previous check.

        An all-pass hand is thrown in with no reward and the same dealer, so it is
        not a completed deal. It is counted separately because a policy that
        learns to pass everything shows up here first.
        """
        current = np.array(
            [
                0 if environment.state is None else environment.state.redeal_count
                for environment in self.environments.environments
            ],
            dtype=np.int64,
        )
        new_redeals = int(np.maximum(current - self.redeal_counts, 0).sum())
        self.redeal_counts = current
        return new_redeals

    def _record_deal_statistics(
        self,
        environment_index: int,
        learner_team: int,
        metrics: RolloutMetrics,
        contract_results: list[float],
        absolute_score_differences: list[float],
    ) -> None:
        """Read the finished deal's public result for reporting only.

        This runs after the deal is over and touches nothing the actor consumes.
        """
        state = self.environments.environments[environment_index].state
        if state is None or state.deal_result is None:  # pragma: no cover - terminal implies scored
            return
        result = state.deal_result
        metrics.winning_bids.append(int(result.bid))
        contract_results.append(float(result.tricks_bidding_team >= result.bid))
        absolute_score_differences.append(
            abs(float(result.score_delta_bidding_team - result.score_delta_defending_team))
        )
        if state.declaring_team == learner_team:
            metrics.learner_declared_deals += 1

    def bootstrap_values(self) -> np.ndarray:
        """Return the ``(environment_count, 4)`` truncation bootstrap values."""
        return self._bootstrap_values_by_environment_seat()


def count_bid_actions(selected_actions: np.ndarray) -> int:
    """Return how many of ``selected_actions`` are bids. Used by diagnostics."""
    return int(
        np.count_nonzero(
            (selected_actions >= BID_ACTION_START) & (selected_actions < BID_ACTION_STOP)
        )
    )
