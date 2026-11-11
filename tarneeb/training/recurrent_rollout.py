"""On-policy rollout collection for the Event-GRU policy.

The recurrent counterpart of :mod:`tarneeb.training.rollout`. It drives the
frozen :class:`~tarneeb.env.vec.VecTarneeb` through its public API only, and
implements no game rule: legality comes from the engine's mask, transitions go
through ``VecTarneeb.step``, rewards come from the engine's scored result.

**Step order matters here in a way it does not for a feedforward policy.** The
loop is deliberately *observe, act, store, step, advance* rather than Phase 6's
*observe, act, step, store*:

1. **act** -- the seat uses its hidden state as-is and consumes nothing (R2);
2. **store** -- the stored segment is the backlog since this seat's previous
   stored decision, or empty at a chunk head (R3, R4);
3. **step** the engine;
4. **advance** -- the newly emitted events advance all four actor states and all
   four critic states of that environment, exactly once, and are appended to all
   four pending queues (R1).

Reversing 2 and 3 would put this decision's own action events into its own
stored segment, which is off by one deal-shatteringly: replay would then feed
them in *before* evaluating the decision that produced them.

**Who sits where.** As in Phase 6, the learner holds one partnership and a fixed
non-neural baseline holds the other, alternating by environment index so the
shared policy occupies all four absolute seats. Opponent decisions are never
stored, so nothing an opponent chose can reach a gradient. The opponent chooses
from a :class:`~tarneeb.env.public_view.PublicView`, held to the same
information boundary as in the Phase 5 arena.

Metrics are returned as plain values. This module owns no logger, no writer and
no renderer, and takes no ``verbose`` argument.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from tarneeb.agents.base import Agent
from tarneeb.buffers.recurrent import (
    DEFAULT_CHUNK_LENGTH,
    RecurrentRolloutBuffer,
    RecurrentTransition,
)
from tarneeb.core.actions import NO_TRUMP_ACTION, PASS_ACTION
from tarneeb.core.cards import NUM_SEATS
from tarneeb.core.events import PublicEvent
from tarneeb.core.state import team_of_seat
from tarneeb.env.event_features import EVENT_ID_DTYPE, event_ids
from tarneeb.env.public_view import PublicViewBuilder
from tarneeb.env.spaces import K_MAX, N_EVENT_FIELDS
from tarneeb.env.vec import VecTarneeb
from tarneeb.models.masking import MaskedCategorical, actions_are_legal
from tarneeb.models.recurrent import RecurrentActorNetwork, RecurrentCriticNetwork
from tarneeb.training.recurrent_state import (
    LEARNER_POLICY_ID,
    NO_EVENT_CONSUMED,
    RecurrentStateStore,
)

EMPTY_SEGMENT_IDS = np.zeros((1, K_MAX, N_EVENT_FIELDS), dtype=np.int64)
"""The ``prev_ev_ids`` passed at a rollout decision point. R2: nothing is consumed."""


@dataclass
class RecurrentRolloutMetrics:
    """Diagnostics produced by one recurrent rollout. Every field is a plain value."""

    environment_steps: int = 0
    decision_count: int = 0
    learner_decision_count: int = 0
    stored_transition_count: int = 0
    completed_deals: int = 0
    passed_out_deals: int = 0
    new_hand_boundaries: int = 0
    illegal_action_count: int = 0
    no_trump_action_count: int = 0
    auction_decision_count: int = 0
    pass_action_count: int = 0
    max_segment_length: int = 0
    total_events_consumed: int = 0
    total_reward_paid: float = 0.0
    total_reward_unattached: float = 0.0
    total_reward_emitted: float = 0.0
    total_learner_reward_emitted: float = 0.0
    mean_learner_reward: float | None = None
    collection_seconds: float = 0.0
    decisions_per_second: float = 0.0
    learner_rewards: list[float] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "environment_steps": self.environment_steps,
            "decision_count": self.decision_count,
            "learner_decision_count": self.learner_decision_count,
            "stored_transition_count": self.stored_transition_count,
            "completed_deals": self.completed_deals,
            "passed_out_deals": self.passed_out_deals,
            "new_hand_boundaries": self.new_hand_boundaries,
            "illegal_action_count": self.illegal_action_count,
            "no_trump_action_count": self.no_trump_action_count,
            "max_segment_length": self.max_segment_length,
            "total_events_consumed": self.total_events_consumed,
            "total_reward_paid": self.total_reward_paid,
            "total_reward_unattached": self.total_reward_unattached,
            "total_learner_reward_emitted": self.total_learner_reward_emitted,
            "mean_learner_reward": self.mean_learner_reward,
            "collection_seconds": self.collection_seconds,
            "decisions_per_second": self.decisions_per_second,
        }


def learner_team_by_environment(environment_count: int) -> np.ndarray:
    """Alternate which partnership the learner holds, by environment index."""
    return np.arange(environment_count, dtype=np.int8) % 2


def relativized_event_ids(events: tuple[PublicEvent, ...], seat: int) -> np.ndarray:
    """Return ``(len(events), 7)`` factored ids of ``events`` from ``seat``'s view."""
    if not events:
        return np.zeros((0, N_EVENT_FIELDS), dtype=EVENT_ID_DTYPE)
    return np.stack([event_ids(event, seat) for event in events])


class RecurrentRolloutCollector:
    """Collects one on-policy recurrent rollout per call."""

    def __init__(
        self,
        environments: VecTarneeb,
        actor: RecurrentActorNetwork,
        critic: RecurrentCriticNetwork,
        rollout_length: int,
        action_sampling_generator: torch.Generator,
        opponents: list[Agent],
        chunk_length: int = DEFAULT_CHUNK_LENGTH,
        policy_id: str = LEARNER_POLICY_ID,
    ) -> None:
        if len(opponents) != environments.num_envs:
            raise ValueError(
                f"expected one opponent per environment ({environments.num_envs}), "
                f"got {len(opponents)}"
            )
        if actor.spec != critic.spec:
            raise ValueError("actor and critic were built from different architecture specs")

        self.environments = environments
        self.actor = actor
        self.critic = critic
        self.rollout_length = rollout_length
        self.action_sampling_generator = action_sampling_generator
        self.opponents = opponents
        self.chunk_length = chunk_length
        self.policy_id = policy_id

        self.learner_teams = learner_team_by_environment(environments.num_envs)
        self.views = PublicViewBuilder(environments.engine)
        self.state = RecurrentStateStore(environments.num_envs, actor.spec, policy_ids=(policy_id,))

        self.environments.reset()
        self.state.reset_all()
        self.redeal_counts = np.zeros(environments.num_envs, dtype=np.int64)

    # --- seating ------------------------------------------------------------

    def is_learner_seat(self, environment_index: int, seat: int) -> bool:
        """True when ``seat`` in this environment is played by the learning policy."""
        return team_of_seat(seat) == int(self.learner_teams[environment_index])

    def learner_seats(self, environment_index: int) -> tuple[int, ...]:
        """Return the two absolute seats the learner holds in this environment."""
        return tuple(
            seat for seat in range(NUM_SEATS) if self.is_learner_seat(environment_index, seat)
        )

    # --- one rollout --------------------------------------------------------

    def collect(self) -> tuple[RecurrentRolloutBuffer, RecurrentRolloutMetrics]:
        """Run ``rollout_length`` vector steps and return the buffer and metrics."""
        # The store must agree with the engine before anything else happens. This
        # is what validates a restored resume payload: a corrupted cursor or a
        # stale queue is caught here, before the first step overwrites it.
        self.assert_consumption_invariant()

        buffer = RecurrentRolloutBuffer(self.environments.num_envs, self.chunk_length)
        metrics = RecurrentRolloutMetrics()
        # A chunk never spans two rollout windows: the previous window's tail was
        # stored in a different buffer. Hidden state is deliberately NOT reset --
        # the deal is still in progress and the seat's memory of it is still
        # valid; it simply becomes the new chunk's stored h0.
        self.state.begin_rollout()

        was_training = self.actor.training
        self.actor.eval()
        self.critic.eval()
        start_time = time.perf_counter()

        for _step in range(self.rollout_length):
            self._one_step(buffer, metrics)

        metrics.collection_seconds = time.perf_counter() - start_time
        if metrics.collection_seconds > 0:
            metrics.decisions_per_second = metrics.decision_count / metrics.collection_seconds
        metrics.stored_transition_count = buffer.stored_decision_count
        if metrics.learner_rewards:
            metrics.mean_learner_reward = float(np.mean(metrics.learner_rewards))

        if was_training:
            self.actor.train()
            self.critic.train()
        return buffer, metrics

    def _one_step(self, buffer: RecurrentRolloutBuffer, metrics: RecurrentRolloutMetrics) -> None:
        environment_count = self.environments.num_envs
        acting_seats = self.environments.acting_seats()
        step_actions = np.zeros(environment_count, dtype=np.int64)

        # --- STEP 1 and 2: act and store, per environment. ------------------
        # R2: the acting seat's state is already current, so nothing is consumed
        # at the call site. prev_ev_len is 0 for every rollout forward.
        for environment_index in range(environment_count):
            seat = int(acting_seats[environment_index])
            if seat < 0:  # pragma: no cover - a terminated env is reset before this
                continue
            if self.is_learner_seat(environment_index, seat):
                step_actions[environment_index] = self._act_and_store(
                    environment_index, seat, buffer, metrics
                )
            else:
                step_actions[environment_index] = self._opponent_action(environment_index)
            metrics.decision_count += 1

        # --- STEP 3: step the engine. ---------------------------------------
        vector_step = self.environments.step(step_actions)
        metrics.environment_steps += environment_count

        # --- STEP 4: R1. Every emitted event advances every seat exactly once.
        for environment_index in range(environment_count):
            emitted = vector_step.events[environment_index]
            self._advance_all_seats(environment_index, emitted, metrics)

        # --- Settlement and boundaries. -------------------------------------
        for environment_index in range(environment_count):
            if vector_step.terminated[environment_index]:
                rewards_by_seat = vector_step.rewards_by_seat[environment_index]
                learner_seats = self.learner_seats(environment_index)
                settlement = buffer.settle_completed_deal(
                    environment_index, rewards_by_seat, seats=learner_seats
                )
                metrics.total_reward_paid += settlement.reward_paid
                metrics.total_reward_unattached += settlement.reward_unattached
                metrics.total_reward_emitted += float(rewards_by_seat.sum())
                metrics.total_learner_reward_emitted += float(
                    sum(rewards_by_seat[seat] for seat in learner_seats)
                )
                metrics.completed_deals += 1
                learner_team = int(self.learner_teams[environment_index])
                metrics.learner_rewards.append(float(rewards_by_seat[learner_team]))
                self.opponents[environment_index].reset()

        metrics.passed_out_deals += self._count_new_redeals()

        if vector_step.terminated.any():
            self.environments.reset_terminated(vector_step.terminated)
            self.redeal_counts[vector_step.terminated] = 0

        # A newly dealt hand -- terminal reset or all-pass redeal -- restarts the
        # engine's event stream at seq 0. Detect it by comparing what each seat
        # has consumed against what the engine now holds, which is the same
        # invariant the collector asserts on every ordinary step.
        for environment_index in range(environment_count):
            environment = self.environments.environments[environment_index]
            state = environment.state
            event_count = 0 if state is None else len(state.events)
            consumed = int(self.state.last_seq_consumed[environment_index, 0])
            if consumed != event_count - 1:
                if event_count != 0:  # pragma: no cover - would mean a lost event
                    raise AssertionError(
                        f"environment {environment_index} holds {event_count} events but seat 0 "
                        f"has consumed up to seq {consumed}; a public event was lost"
                    )
                self.state.reset_environment(environment_index)
                metrics.new_hand_boundaries += 1
            else:
                self.state.assert_consumption_matches(environment_index, event_count)

    def assert_consumption_invariant(self) -> None:
        """Assert every seat's cursor matches the events its engine currently holds.

        Cheap, and it turns an entire class of silent recurrent corruption into
        an immediate, located failure. Called at the start of every rollout and
        after every engine step.
        """
        for environment_index, environment in enumerate(self.environments.environments):
            state = environment.state
            event_count = 0 if state is None else len(state.events)
            self.state.assert_consumption_matches(environment_index, event_count)

    def _act_and_store(
        self,
        environment_index: int,
        seat: int,
        buffer: RecurrentRolloutBuffer,
        metrics: RecurrentRolloutMetrics,
    ) -> int:
        """Choose the learner's action and store the transition. Returns the action."""
        environment = self.environments.environments[environment_index]
        observation = environment.actor_observation(seat)
        legal_action_mask = environment.legal_action_mask()

        observation_tensor = torch.from_numpy(observation).unsqueeze(0)
        legal_mask_tensor = torch.from_numpy(legal_action_mask).unsqueeze(0)
        empty_ids = torch.from_numpy(EMPTY_SEGMENT_IDS)
        empty_lengths = torch.zeros(1, dtype=torch.int64)

        actor_hidden = self.state.actor_slot(environment_index, seat, self.policy_id)
        critic_hidden = self.state.critic_slot(environment_index, seat, self.policy_id)

        with torch.no_grad():
            logits, advanced_actor = self.actor(
                observation_tensor, empty_ids, empty_lengths, actor_hidden
            )
            value, advanced_critic = self.critic(
                observation_tensor, empty_ids, empty_lengths, critic_hidden
            )
            distribution = MaskedCategorical.from_logits(logits, legal_mask_tensor)
            selected_action = distribution.sample(self.action_sampling_generator)
            log_probability = distribution.log_probability_of(selected_action)

        # R2 is a correctness property, so it is verified rather than trusted:
        # the decision forward must not have moved the state at all.
        if not bool(torch.equal(advanced_actor, actor_hidden)):
            raise AssertionError(
                "the actor forward advanced the hidden state at a decision point; "
                "rule R2 requires prev_ev_len == 0 to be a strict no-op"
            )
        if not bool(torch.equal(advanced_critic, critic_hidden)):
            raise AssertionError("the critic forward advanced the hidden state at a decision point")

        # Masking is a safety property, so it is verified rather than trusted.
        if not actions_are_legal(selected_action, legal_mask_tensor):
            metrics.illegal_action_count += 1
            raise AssertionError("the policy sampled an action its legal mask forbids")
        action = int(selected_action.item())
        if action == NO_TRUMP_ACTION:
            metrics.no_trump_action_count += 1
            raise AssertionError("the policy selected the reserved no-trump action 64")

        if bool(legal_action_mask[PASS_ACTION]):
            metrics.auction_decision_count += 1
            if action == PASS_ACTION:
                metrics.pass_action_count += 1

        is_head = self.state.is_chunk_head(environment_index, seat)
        segment_ids, segment_length = self.state.stored_segment(environment_index, seat, is_head)
        metrics.max_segment_length = max(metrics.max_segment_length, segment_length)

        state = environment.state
        assert state is not None
        # The stored segment ends at the last event consumed, so it began
        # `segment_length` events earlier. -1 marks an empty segment.
        last_consumed = int(self.state.last_seq_consumed[environment_index, seat])
        first_event_seq = last_consumed - segment_length + 1 if segment_length else -1

        buffer.add(
            RecurrentTransition(
                environment_index=environment_index,
                seat=seat,
                static=observation.copy(),
                critic_static=observation.copy(),
                previous_event_ids=segment_ids,
                previous_event_length=segment_length,
                legal_action_mask=legal_action_mask.copy(),
                action=action,
                old_log_probability=float(log_probability.item()),
                value=float(value.item()),
                is_chunk_head=is_head,
                actor_hidden_head=(
                    actor_hidden.squeeze(1).detach().numpy().copy() if is_head else None
                ),
                critic_hidden_head=(
                    critic_hidden.squeeze(1).detach().numpy().copy() if is_head else None
                ),
                first_event_seq=first_event_seq,
            )
        )
        self.state.note_decision_stored(environment_index, seat, self.chunk_length)
        metrics.learner_decision_count += 1
        return action

    def _opponent_action(self, environment_index: int) -> int:
        """Ask the fixed opponent for its move, from a public view only."""
        environment = self.environments.environments[environment_index]
        state = environment.state
        assert state is not None and state.current_player_seat is not None
        view = self.views.build(state, state.current_player_seat)
        return int(self.opponents[environment_index].act(view))

    def _advance_all_seats(
        self,
        environment_index: int,
        events: tuple[PublicEvent, ...],
        metrics: RecurrentRolloutMetrics,
    ) -> None:
        """R1: fold ``events`` into all four actor and all four critic states once.

        All four seats are batched into a single GRU call per network. Advancing
        every seat unconditionally, rather than only the learner's, costs one
        extra batched call and removes the entire class of bugs that appears the
        moment a different partnership becomes learner-controlled.
        """
        if not events:
            return

        count = len(events)
        if count > K_MAX:
            raise AssertionError(
                f"one engine step emitted {count} events, which exceeds K_MAX={K_MAX}"
            )

        seats = np.arange(NUM_SEATS, dtype=np.int64)
        environment_indices = np.full(NUM_SEATS, environment_index, dtype=np.int64)

        id_rows = np.stack([relativized_event_ids(events, int(seat)) for seat in seats])
        padded = np.zeros((NUM_SEATS, K_MAX, N_EVENT_FIELDS), dtype=np.int64)
        padded[:, :count] = id_rows
        id_tensor = torch.from_numpy(padded)
        length_tensor = torch.full((NUM_SEATS,), count, dtype=torch.int64)

        with torch.no_grad():
            actor_hidden = self.state.gather_actor(environment_indices, seats, self.policy_id)
            critic_hidden = self.state.gather_critic(environment_indices, seats, self.policy_id)
            advanced_actor = self.actor.trunk.history_encoder.advance(
                id_tensor, length_tensor, actor_hidden
            )
            advanced_critic = self.critic.trunk.history_encoder.advance(
                id_tensor, length_tensor, critic_hidden
            )
        self.state.scatter_actor(environment_indices, seats, advanced_actor, self.policy_id)
        self.state.scatter_critic(environment_indices, seats, advanced_critic, self.policy_id)

        for seat in range(NUM_SEATS):
            self.state.append_events(environment_index, seat, id_rows[seat])
        self.state.last_seq_consumed[environment_index, :] = int(events[-1].seq)
        metrics.total_events_consumed += count * NUM_SEATS

    def _count_new_redeals(self) -> int:
        """Return how many all-pass redeals happened since the previous check."""
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

    # --- bootstrap ----------------------------------------------------------

    def bootstrap_values(self) -> np.ndarray:
        """Return ``(environment_count, 4)`` truncation bootstrap values.

        Every seat is valued, not only the one on turn, because the bootstrap
        belongs to each seat's own decision sequence. Each seat is valued with
        **its own** critic state and **its own** relative observation; opponent
        seats are valued too and simply never used.
        """
        environment_count = self.environments.num_envs
        values = np.zeros((environment_count, NUM_SEATS), dtype=np.float32)

        observations = np.zeros(
            (environment_count * NUM_SEATS, self.critic.spec.critic_observation_dim),
            dtype=np.float32,
        )
        environment_indices = np.zeros(environment_count * NUM_SEATS, dtype=np.int64)
        seats = np.zeros(environment_count * NUM_SEATS, dtype=np.int64)
        for environment_index, environment in enumerate(self.environments.environments):
            for seat in range(NUM_SEATS):
                row = environment_index * NUM_SEATS + seat
                observations[row] = environment.actor_observation(seat)
                environment_indices[row] = environment_index
                seats[row] = seat

        empty_ids = torch.zeros(
            (environment_count * NUM_SEATS, K_MAX, N_EVENT_FIELDS), dtype=torch.int64
        )
        empty_lengths = torch.zeros(environment_count * NUM_SEATS, dtype=torch.int64)
        with torch.no_grad():
            hidden = self.state.gather_critic(environment_indices, seats, self.policy_id)
            predicted, _advanced = self.critic(
                torch.from_numpy(observations), empty_ids, empty_lengths, hidden
            )
        values[:] = predicted.numpy().reshape(environment_count, NUM_SEATS)
        return values

    # --- resume -------------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        """Return the collector's own mutable state, for a resume payload."""
        return {
            "recurrent_state": self.state.snapshot(),
            "redeal_counts": self.redeal_counts.tolist(),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        """Put a collector snapshot back."""
        self.state.restore(snapshot["recurrent_state"])  # type: ignore[arg-type]
        self.redeal_counts = np.array(snapshot["redeal_counts"], dtype=np.int64)


def fresh_hand_state_is_zero(store: RecurrentStateStore, environment_index: int) -> bool:
    """Return True when every slot and queue of one environment is at its reset value."""
    for policy_id in store.policy_ids:
        if bool(store.actor_hidden[policy_id][environment_index].any()):
            return False
        if bool(store.critic_hidden[policy_id][environment_index].any()):
            return False
    if any(len(queue) for queue in store.pending[environment_index]):
        return False
    return bool((store.last_seq_consumed[environment_index] == NO_EVENT_CONSUMED).all())
