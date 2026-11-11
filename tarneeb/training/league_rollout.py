"""G6-safe recurrent collection with deal-scoped league opponents.

This is an additive extension of the Phase 10 centralized collector. It keeps
the frozen environment, reward, mask, buffer, and complete-deal behavior while
grouping neural requests by ``policy_id``. Historical actors are obtained from
the Phase 11 actor cache; only the learner's seats are written to the PPO buffer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch import Tensor

from tarneeb.agents.base import Agent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.buffers.recurrent import DEFAULT_CHUNK_LENGTH, RecurrentRolloutBuffer
from tarneeb.core.actions import NO_TRUMP_ACTION, PASS_ACTION
from tarneeb.core.cards import NUM_SEATS
from tarneeb.env.public_view import PublicView
from tarneeb.env.spaces import N_EVENT_FIELDS
from tarneeb.league.cache import HistoricalActorCache
from tarneeb.league.sampling import OpponentProvider
from tarneeb.models.masking import MaskedCategorical, actions_are_legal
from tarneeb.models.recurrent import RecurrentActorNetwork, RecurrentCriticNetwork
from tarneeb.training.centralized_rollout import CentralizedCriticCollector
from tarneeb.training.recurrent_rollout import (
    RecurrentRolloutMetrics,
    fresh_hand_state_is_zero,
    relativized_event_ids,
)
from tarneeb.training.recurrent_state import LEARNER_POLICY_ID, RecurrentStateStore

NeuralRequest = tuple[int, int]
"""One acting ``(environment_index, seat)`` request for an actor."""


class _BatchedOpponentAgent(Agent):
    """Placeholder satisfying the frozen collector's reset hook for neural seats."""

    name = "batched_neural_opponent"

    def act(self, view: PublicView) -> int:
        raise AssertionError("historical neural actions must use grouped inference")


class LeagueRecurrentCollector(CentralizedCriticCollector):
    """One G6-safe collector shared by latest-only and PFSP opponent providers.

    ``learner_team_by_environment`` remains the seating convention: each
    environment assigns the learner one complete partnership, alternating the
    absolute partnership across environment indices. One selected opponent
    policy controls both opposing seats until that dealt hand ends.
    """

    def __init__(
        self,
        environments: Any,
        actor: RecurrentActorNetwork,
        critic: RecurrentCriticNetwork,
        rollout_length: int,
        action_sampling_generator: torch.Generator,
        opponent_provider: OpponentProvider,
        actor_cache: HistoricalActorCache,
        chunk_length: int = DEFAULT_CHUNK_LENGTH,
        policy_id: str = LEARNER_POLICY_ID,
        opponent_seed: int = 0,
        opponent_agents_by_policy: dict[str, list[Agent]] | None = None,
    ) -> None:
        self.opponent_provider = opponent_provider
        self.actor_cache = actor_cache
        self.opponent_seed = opponent_seed

        self.historical_policy_ids = tuple(
            policy_id_value
            for policy_id_value, record in actor_cache.registry.records.items()
            if record.checkpoint_path is not None and policy_id_value != policy_id
        )
        if policy_id in self.historical_policy_ids:
            raise ValueError("learner policy_id cannot also be a historical actor-cache entry")
        self.neural_policy_ids = (policy_id, *self.historical_policy_ids)

        if opponent_agents_by_policy is None:
            opponent_agents_by_policy = self._build_heuristic_opponents(
                actor_cache, environments.num_envs, opponent_seed
            )
        self.opponent_agents_by_policy = opponent_agents_by_policy
        for heuristic_policy_id, agents in self.opponent_agents_by_policy.items():
            if len(agents) != environments.num_envs:
                raise ValueError(
                    f"policy {heuristic_policy_id!r} needs one heuristic agent per environment"
                )
            if len({id(agent) for agent in agents}) != len(agents):
                raise ValueError("each environment must have its own heuristic agent instance")

        super().__init__(
            environments=environments,
            actor=actor,
            critic=critic,
            rollout_length=rollout_length,
            action_sampling_generator=action_sampling_generator,
            opponents=[_BatchedOpponentAgent() for _ in range(environments.num_envs)],
            chunk_length=chunk_length,
            policy_id=policy_id,
        )
        self.state = RecurrentStateStore(
            environments.num_envs, actor.spec, policy_ids=self.neural_policy_ids
        )
        self.opponent_policy_ids: list[str | None] = [None] * environments.num_envs
        self.opponent_assignment_generation = [0] * environments.num_envs
        self._historical_actors: dict[str, RecurrentActorNetwork] = {}
        self.sampled_opponent_policy_ids: list[str] = []
        self.completed_deal_outcomes: list[tuple[str, float]] = []
        self.policy_assignment_history: list[tuple[int, int, str]] = []
        self.decision_policy_history: list[tuple[int, int, int, str]] = []
        self.action_history: list[tuple[int, int, str, int, np.ndarray]] = []
        self.neural_decision_count = 0
        self.opponent_neural_decision_count = 0
        self.neural_policy_group_count = 0
        self.neural_policy_group_sizes: list[tuple[tuple[str, int], ...]] = []
        self.non_finite_count = 0
        self.cache_load_count = 0
        self.cache_hit_count = 0
        self.last_collection_statistics: dict[str, Any] = {}

    @staticmethod
    def _build_heuristic_opponents(
        actor_cache: HistoricalActorCache, environment_count: int, opponent_seed: int
    ) -> dict[str, list[Agent]]:
        agents_by_policy: dict[str, list[Agent]] = {}
        for policy_id, record in actor_cache.registry.records.items():
            if record.checkpoint_path is not None:
                continue
            if record.category == "strong":
                agents_by_policy[policy_id] = [
                    StrongHeuristicAgent() for _environment_index in range(environment_count)
                ]
            elif record.category == "random":
                agents_by_policy[policy_id] = [
                    RandomAgent(seed=opponent_seed + environment_index)
                    for environment_index in range(environment_count)
                ]
        return agents_by_policy

    def policy_id_for_seat(self, environment_index: int, seat: int) -> str:
        """Return the fixed policy assignment for this seat's current deal."""
        if self.is_learner_seat(environment_index, seat):
            return self.policy_id
        opponent_policy_id = self.opponent_policy_ids[environment_index]
        if opponent_policy_id is None:
            raise AssertionError(f"environment {environment_index} has no opponent assignment")
        return opponent_policy_id

    def collect(self) -> tuple[RecurrentRolloutBuffer, RecurrentRolloutMetrics]:
        """Select deal opponents at the fresh boundary, then use the frozen G6 drain."""
        for environment_index, environment in enumerate(self.environments.environments):
            if environment.state is None or environment.state.events:
                raise AssertionError("league seating may only be selected at a fresh deal boundary")
            if not fresh_hand_state_is_zero(self.state, environment_index):
                raise AssertionError("recurrent state must be zero before assigning a new deal")

        self.sampled_opponent_policy_ids = []
        self.completed_deal_outcomes = []
        self.policy_assignment_history = []
        self.decision_policy_history = []
        self.action_history = []
        self.opponent_assignment_generation = [0] * self.environments.num_envs
        self.neural_decision_count = 0
        self.opponent_neural_decision_count = 0
        self.neural_policy_group_count = 0
        self.neural_policy_group_sizes = []
        self.non_finite_count = 0
        cache_before = self.actor_cache.statistics()
        for environment_index in range(self.environments.num_envs):
            self._assign_opponent_policy(environment_index)

        buffer, metrics = super().collect()
        cache_after = self.actor_cache.statistics()
        self.cache_load_count = cache_after["loads"] - cache_before["loads"]
        self.cache_hit_count = cache_after["hits"] - cache_before["hits"]
        self.last_collection_statistics = {
            "decision_count": metrics.decision_count,
            "learner_decision_count": metrics.learner_decision_count,
            "opponent_neural_decision_count": self.opponent_neural_decision_count,
            "neural_decision_count": self.neural_decision_count,
            "neural_policy_group_count": self.neural_policy_group_count,
            "neural_policy_group_sizes": tuple(self.neural_policy_group_sizes),
            "completed_deals": metrics.completed_deals,
            "passed_out_deals": metrics.passed_out_deals,
            "illegal_action_count": metrics.illegal_action_count,
            "no_trump_action_count": metrics.no_trump_action_count,
            "non_finite_count": self.non_finite_count,
            "cache_load_count": self.cache_load_count,
            "cache_hit_count": self.cache_hit_count,
            "unique_opponent_policy_ids": tuple(dict.fromkeys(self.sampled_opponent_policy_ids)),
            "completed_deal_outcomes": tuple(self.completed_deal_outcomes),
            "collection_seconds": metrics.collection_seconds,
            "decisions_per_second": metrics.decisions_per_second,
            "last_collection_steps": self.last_collection_steps,
            "no_unfinished_learner_sequence": self.no_unfinished_learner_sequence(buffer),
            "recurrent_state_is_zero": self.recurrent_state_is_zero(),
        }
        return buffer, metrics

    def _assign_opponent_policy(self, environment_index: int) -> None:
        """Assign one coherent opponent at a fresh hand boundary only."""
        if not fresh_hand_state_is_zero(self.state, environment_index):
            raise AssertionError("opponent policy cannot switch while recurrent state is live")
        selected_policy_id = self.opponent_provider.sample()
        if selected_policy_id == self.policy_id:
            self.opponents[environment_index] = _BatchedOpponentAgent()
        else:
            record = self.actor_cache.registry.records.get(selected_policy_id)
            if record is None or not record.eligible:
                raise ValueError(
                    f"opponent provider selected ineligible policy {selected_policy_id!r}"
                )
            if record.checkpoint_path is not None:
                actor = self.actor_cache.get(selected_policy_id)
                self._assert_actor_compatible(actor, selected_policy_id)
                self._historical_actors[selected_policy_id] = actor
                self.opponents[environment_index] = _BatchedOpponentAgent()
            else:
                try:
                    self.opponents[environment_index] = self.opponent_agents_by_policy[
                        selected_policy_id
                    ][environment_index]
                except KeyError as error:
                    raise ValueError(
                        "no existing heuristic agent is configured for policy "
                        f"{selected_policy_id!r}"
                    ) from error

        self.opponent_policy_ids[environment_index] = selected_policy_id
        self.opponent_assignment_generation[environment_index] += 1
        self.policy_assignment_history.append(
            (
                environment_index,
                self.opponent_assignment_generation[environment_index],
                selected_policy_id,
            )
        )
        self.sampled_opponent_policy_ids.append(selected_policy_id)

    def refresh_registered_policies(self) -> None:
        """Refresh policy-specific recurrent slots at a safe G6 boundary.

        Phase 11 may register a newly frozen learner snapshot after an optimizer
        update. The collector can adopt that snapshot only when every recurrent
        slot is already zero; no live deal or hidden state is migrated.
        """
        if not self.recurrent_state_is_zero():
            raise AssertionError("registered policies may refresh only at a zero-state boundary")
        historical_policy_ids = tuple(
            policy_id
            for policy_id, record in self.actor_cache.registry.records.items()
            if record.checkpoint_path is not None and policy_id != self.policy_id
        )
        if self.policy_id in historical_policy_ids:
            raise ValueError("learner policy_id cannot also be a historical actor-cache entry")

        self.historical_policy_ids = historical_policy_ids
        self.neural_policy_ids = (self.policy_id, *historical_policy_ids)
        self.state = RecurrentStateStore(
            self.environments.num_envs, self.actor.spec, policy_ids=self.neural_policy_ids
        )
        self._historical_actors = {
            policy_id: actor
            for policy_id, actor in self._historical_actors.items()
            if policy_id in historical_policy_ids
        }
        self.opponent_policy_ids = [None] * self.environments.num_envs

    def _assert_actor_compatible(self, actor: RecurrentActorNetwork, policy_id: str) -> None:
        learner_spec = self.actor.spec
        historical_spec = actor.spec
        actor_contract_fields = (
            "actor_observation_dim",
            "action_count",
            "event_field_count",
            "event_embedding_dims",
            "event_projection_dim",
            "hidden_size",
            "gru_layer_count",
            "static_hidden_dim",
            "max_segment_length",
            "history_encoder",
            "gru_reset",
        )
        mismatches = [
            field
            for field in actor_contract_fields
            if getattr(learner_spec, field) != getattr(historical_spec, field)
        ]
        if mismatches:
            raise ValueError(
                f"historical actor {policy_id!r} is incompatible with the learner actor: "
                f"{', '.join(mismatches)}"
            )

    def _one_step(self, buffer: RecurrentRolloutBuffer, metrics: RecurrentRolloutMetrics) -> None:
        environment_count = self.environments.num_envs
        acting_seats = self.environments.acting_seats()
        step_actions = np.zeros(environment_count, dtype=np.int64)
        neural_requests: dict[str, list[NeuralRequest]] = defaultdict(list)
        heuristic_actions: dict[NeuralRequest, int] = {}
        observations: dict[NeuralRequest, np.ndarray] = {}
        legal_action_masks: dict[NeuralRequest, np.ndarray] = {}

        for environment_index in range(environment_count):
            seat = int(acting_seats[environment_index])
            if seat < 0:
                continue
            request = (environment_index, seat)
            policy_id = self.policy_id_for_seat(environment_index, seat)
            self.decision_policy_history.append(
                (
                    environment_index,
                    self.opponent_assignment_generation[environment_index],
                    seat,
                    policy_id,
                )
            )
            metrics.decision_count += 1
            if self._is_neural_policy(policy_id):
                neural_requests[policy_id].append(request)
                environment = self.environments.environments[environment_index]
                observations[request] = environment.actor_observation(seat)
                legal_action_masks[request] = environment.legal_action_mask()
            else:
                heuristic_actions[request] = self._opponent_action(environment_index)

        logits_by_request: dict[NeuralRequest, Tensor] = {}
        for policy_id, requests in neural_requests.items():
            actor = (
                self.actor if policy_id == self.policy_id else self._historical_actors[policy_id]
            )
            logits_by_request.update(
                self._actor_decision_outputs(policy_id, actor, requests, observations)
            )
        self.neural_policy_group_count += len(neural_requests)
        self.neural_policy_group_sizes.append(
            tuple((policy_id, len(requests)) for policy_id, requests in neural_requests.items())
        )
        self.neural_decision_count += sum(len(requests) for requests in neural_requests.values())
        self.opponent_neural_decision_count += sum(
            not self.is_learner_seat(*request)
            for requests in neural_requests.values()
            for request in requests
        )

        learner_requests = [
            request
            for requests in neural_requests.values()
            for request in requests
            if self.is_learner_seat(*request)
        ]
        critic_values = self._learner_critic_values(learner_requests)

        # Draw from the shared Torch generator in environment order, independent
        # of which actor group produced the logits.
        for environment_index in range(environment_count):
            seat = int(acting_seats[environment_index])
            if seat < 0:
                continue
            request = (environment_index, seat)
            if request in heuristic_actions:
                action = heuristic_actions[request]
            else:
                policy_id = self.policy_id_for_seat(environment_index, seat)
                logits = logits_by_request[request]
                legal_action_mask = legal_action_masks[request]
                legal_mask_tensor = torch.from_numpy(legal_action_mask).unsqueeze(0)
                distribution = MaskedCategorical.from_logits(logits, legal_mask_tensor)
                with torch.no_grad():
                    selected_action = distribution.sample(self.action_sampling_generator)
                if not actions_are_legal(selected_action, legal_mask_tensor):
                    metrics.illegal_action_count += 1
                    raise AssertionError("league policy sampled an action its legal mask forbids")
                action = int(selected_action.item())
                if action == NO_TRUMP_ACTION:
                    metrics.no_trump_action_count += 1
                    raise AssertionError("league policy selected reserved action 64")

                if self.is_learner_seat(environment_index, seat):
                    observation = observations[request]
                    critic_observation = self.critic_observation(environment_index, seat)
                    self._store_decision(
                        environment_index=environment_index,
                        seat=seat,
                        observation=observation,
                        legal_action_mask=legal_action_mask,
                        action=action,
                        log_probability=self._log_probability(logits, legal_mask_tensor, action),
                        value=critic_values[request],
                        actor_slot=self.state.actor_slot(environment_index, seat, self.policy_id),
                        critic_slot=self.state.critic_slot(environment_index, seat, self.policy_id),
                        buffer=buffer,
                        metrics=metrics,
                        critic_observation=critic_observation,
                    )

            recorded_legal_action_mask = legal_action_masks.get(request)
            if recorded_legal_action_mask is None:
                recorded_legal_action_mask = self.environments.environments[
                    environment_index
                ].legal_action_mask()
            if action == NO_TRUMP_ACTION:
                metrics.no_trump_action_count += 1
                raise AssertionError("action 64 is reserved and must never be selected")
            if not bool(recorded_legal_action_mask[action]):
                metrics.illegal_action_count += 1
                raise AssertionError("opponent proposed an action its legal mask forbids")
            self.action_history.append(
                (
                    environment_index,
                    seat,
                    self.policy_id_for_seat(environment_index, seat),
                    action,
                    recorded_legal_action_mask.copy(),
                )
            )
            if self.is_learner_seat(environment_index, seat) and bool(
                legal_action_mask[PASS_ACTION]
            ):
                metrics.auction_decision_count += 1
                if action == PASS_ACTION:
                    metrics.pass_action_count += 1
            step_actions[environment_index] = action

        vector_step = self.environments.step(step_actions)
        metrics.environment_steps += environment_count
        self._advance_policy_states(vector_step.events, metrics)
        self._settle_and_reset(vector_step, buffer, metrics)

    def _is_neural_policy(self, policy_id: str) -> bool:
        return policy_id == self.policy_id or policy_id in self.historical_policy_ids

    def _opponent_action(self, environment_index: int) -> int:
        environment = self.environments.environments[environment_index]
        state = environment.state
        if state is None or state.current_player_seat is None:
            raise AssertionError("a heuristic opponent was requested without an acting seat")
        view = self.views.build(state, state.current_player_seat)
        return int(self.opponents[environment_index].act(view))

    def _actor_decision_outputs(
        self,
        policy_id: str,
        actor: RecurrentActorNetwork,
        requests: list[NeuralRequest],
        observations: dict[NeuralRequest, np.ndarray],
    ) -> dict[NeuralRequest, Tensor]:
        """Run one batched actor forward for the requests owned by one policy."""
        environment_indices = np.array([request[0] for request in requests], dtype=np.int64)
        seats = np.array([request[1] for request in requests], dtype=np.int64)
        observation_batch = torch.from_numpy(
            np.stack([observations[request] for request in requests])
        )
        event_ids = torch.zeros(
            (len(requests), actor.spec.max_segment_length, N_EVENT_FIELDS), dtype=torch.int64
        )
        event_lengths = torch.zeros(len(requests), dtype=torch.int64)
        hidden_state = self.state.gather_actor(environment_indices, seats, policy_id)
        inference_context = (
            torch.no_grad() if policy_id == self.policy_id else torch.inference_mode()
        )
        with inference_context:
            logits, advanced_hidden_state = actor(
                observation_batch, event_ids, event_lengths, hidden_state
            )
        if not torch.equal(advanced_hidden_state, hidden_state):
            raise AssertionError("R2 requires empty decision segments to preserve hidden state")
        if not bool(torch.isfinite(logits).all()):
            self.non_finite_count += 1
            raise AssertionError(f"policy {policy_id!r} produced non-finite logits")
        return {request: logits[row : row + 1] for row, request in enumerate(requests)}

    def _learner_critic_values(self, requests: list[NeuralRequest]) -> dict[NeuralRequest, float]:
        if not requests:
            return {}
        environment_indices = np.array([request[0] for request in requests], dtype=np.int64)
        seats = np.array([request[1] for request in requests], dtype=np.int64)
        critic_observations = np.stack(
            [
                self.critic_observation(environment_index, seat)
                for environment_index, seat in requests
            ]
        )
        observation_tensor = torch.from_numpy(critic_observations)
        event_ids = torch.zeros(
            (len(requests), self.critic.spec.max_segment_length, N_EVENT_FIELDS),
            dtype=torch.int64,
        )
        event_lengths = torch.zeros(len(requests), dtype=torch.int64)
        hidden_state = self.state.gather_critic(environment_indices, seats, self.policy_id)
        with torch.no_grad():
            values, advanced_hidden_state = self.critic(
                observation_tensor, event_ids, event_lengths, hidden_state
            )
        if not torch.equal(advanced_hidden_state, hidden_state):
            raise AssertionError("learner critic advanced state at a decision point")
        if not bool(torch.isfinite(values).all()):
            self.non_finite_count += 1
            raise AssertionError("learner critic produced non-finite values")
        return {request: float(values[row].item()) for row, request in enumerate(requests)}

    def _advance_policy_states(
        self,
        events_by_environment: list[tuple[Any, ...]],
        metrics: RecurrentRolloutMetrics,
    ) -> None:
        """Advance each neural controller's per-seat state on its own event rows."""
        event_rows_by_policy: dict[str, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
        for environment_index, events in enumerate(events_by_environment):
            if not events:
                continue
            if len(events) > self.actor.spec.max_segment_length:
                raise AssertionError(f"one vector step emitted {len(events)} events, above K_MAX")
            for seat in range(NUM_SEATS):
                id_rows = relativized_event_ids(events, seat)
                self.state.append_events(environment_index, seat, id_rows)
                policy_id = self.policy_id_for_seat(environment_index, seat)
                if self._is_neural_policy(policy_id):
                    event_rows_by_policy[policy_id].append((environment_index, seat, id_rows))
            self.state.last_seq_consumed[environment_index, :] = int(events[-1].seq)
        metrics.total_events_consumed += sum(
            len(id_rows)
            for event_rows in event_rows_by_policy.values()
            for _environment_index, _seat, id_rows in event_rows
        )

        for policy_id, event_rows in event_rows_by_policy.items():
            environment_indices = np.array([row[0] for row in event_rows], dtype=np.int64)
            seats = np.array([row[1] for row in event_rows], dtype=np.int64)
            event_lengths = np.array([len(row[2]) for row in event_rows], dtype=np.int64)
            longest = int(event_lengths.max())
            event_ids = np.zeros((len(event_rows), longest, N_EVENT_FIELDS), dtype=np.int64)
            for row, (_environment_index, _seat, id_rows) in enumerate(event_rows):
                event_ids[row, : len(id_rows)] = id_rows
            actor = (
                self.actor if policy_id == self.policy_id else self._historical_actors[policy_id]
            )
            self._advance_actor_group(
                policy_id,
                actor,
                environment_indices,
                seats,
                torch.from_numpy(event_ids),
                torch.from_numpy(event_lengths),
            )

        learner_rows = [
            row
            for row in event_rows_by_policy.get(self.policy_id, [])
            if self.is_learner_seat(row[0], row[1])
        ]
        if learner_rows:
            environment_indices = np.array([row[0] for row in learner_rows], dtype=np.int64)
            seats = np.array([row[1] for row in learner_rows], dtype=np.int64)
            event_lengths = np.array([len(row[2]) for row in learner_rows], dtype=np.int64)
            longest = int(event_lengths.max())
            event_ids = np.zeros((len(learner_rows), longest, N_EVENT_FIELDS), dtype=np.int64)
            for row, (_environment_index, _seat, id_rows) in enumerate(learner_rows):
                event_ids[row, : len(id_rows)] = id_rows
            hidden_state = self.state.gather_critic(environment_indices, seats, self.policy_id)
            with torch.no_grad():
                advanced_hidden_state = self.critic.trunk.history_encoder.advance(
                    torch.from_numpy(event_ids),
                    torch.from_numpy(event_lengths),
                    hidden_state,
                )
            self.state.scatter_critic(
                environment_indices, seats, advanced_hidden_state, self.policy_id
            )

    def _advance_actor_group(
        self,
        policy_id: str,
        actor: RecurrentActorNetwork,
        environment_indices: np.ndarray,
        seats: np.ndarray,
        event_ids: Tensor,
        event_lengths: Tensor,
    ) -> None:
        hidden_state = self.state.gather_actor(environment_indices, seats, policy_id)
        with torch.inference_mode():
            advanced_hidden_state = actor.trunk.history_encoder.advance(
                event_ids, event_lengths, hidden_state
            )
        self.state.scatter_actor(environment_indices, seats, advanced_hidden_state, policy_id)

    def _settle_and_reset(
        self,
        vector_step: Any,
        buffer: RecurrentRolloutBuffer,
        metrics: RecurrentRolloutMetrics,
    ) -> None:
        """Inherit reward/reset behavior, then sample only at new-deal boundaries."""
        cursor_before_reset = self.state.last_seq_consumed[:, 0].copy()
        terminated = np.asarray(vector_step.terminated, dtype=bool)
        active_before_settlement = np.asarray(self._active, dtype=bool).copy()
        for raw_environment_index in np.flatnonzero(terminated & active_before_settlement):
            environment_index = int(raw_environment_index)
            policy_id = self.opponent_policy_ids[int(environment_index)]
            if policy_id is None:
                raise AssertionError("a scored league deal ended without an opponent policy")
            learner_team = int(self.learner_teams[int(environment_index)])
            learner_reward = float(
                vector_step.rewards_by_seat[int(environment_index), learner_team]
            )
            if learner_reward == 0.0:
                raise AssertionError(
                    "a scored Tarneeb deal produced zero learner reward; win-rate tie semantics "
                    "are intentionally undefined"
                )
            self.completed_deal_outcomes.append((policy_id, 1.0 if learner_reward > 0.0 else 0.0))

        super()._settle_and_reset(vector_step, buffer, metrics)
        for environment_index in range(self.environments.num_envs):
            reset_cursor = bool((self.state.last_seq_consumed[environment_index] == -1).all())
            if cursor_before_reset[environment_index] == -1 or not reset_cursor:
                continue
            # A scored deal already reset its agent in the frozen settlement;
            # a passed-out hand needs the same per-hand agent reset here.
            if not bool(vector_step.terminated[environment_index]):
                self.opponents[environment_index].reset()
            if not bool(self._active[environment_index]) or (
                self._draining and bool(vector_step.terminated[environment_index])
            ):
                continue
            self._assign_opponent_policy(environment_index)

    def no_unfinished_learner_sequence(self, buffer: RecurrentRolloutBuffer) -> bool:
        """Expose the inherited G6 sequence-boundary guarantee under its name."""
        return super().no_unfinished_learner_sequence(buffer)

    def recurrent_state_is_zero(self) -> bool:
        """Return whether learner and every historical actor state are reset."""
        return super().recurrent_state_is_zero()

    def _log_probability(self, logits: Tensor, legal_action_mask: Tensor, action: int) -> float:
        distribution = MaskedCategorical.from_logits(logits, legal_action_mask)
        selected_action = torch.tensor([action], dtype=torch.int64)
        with torch.no_grad():
            return float(distribution.log_probability_of(selected_action).item())
