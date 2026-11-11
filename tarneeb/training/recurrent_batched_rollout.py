"""Cross-environment batched recurrent rollout collection (Phase 9, Part A).

Phase 8 measured the recurrent collector at ~401 decisions/s against ~2231 for
the feedforward baseline, with ~71% of the time inside
``RecurrentRolloutCollector._advance_all_seats``. The diagnosis recorded in
docs/RECURRENT_CONTRACT.md section 11 is that seats are batched *within* one
environment but environments are not batched together: the method runs once per
environment per vector step, and each call does two GRU passes over a batch of
four. The GRU arithmetic is not the cost -- the call count is.

This module is the fix, and it is **additive**. Nothing in Phase 8 is edited.
:class:`BatchedRecurrentRolloutCollector` subclasses the frozen collector and
overrides exactly one method, ``_one_step``. The frozen implementation stays in
place as the **equivalence oracle**: ``tests/test_recurrent_batching.py`` runs
both collectors from identical state and requires them to agree.

Two batching changes, and nothing else:

1. **The decision forward.** Every learner ``(environment, seat)`` due to act
   goes through one actor forward and one critic forward instead of N of each.
   ``prev_ev_len`` is zero for every row, so **R2 is preserved exactly**: the
   acting seat still consumes nothing at its decision point.

2. **Event advancement.** After the engine step, all ``4 x N`` slots go through
   **two** GRU calls with per-row segment lengths, padded to that step's
   maximum. Rows with no events are exact no-ops, guaranteed by the frozen
   ``EventGRUHistoryEncoder.advance``. **R1 is preserved**: every emitted public
   event advances every seat exactly once, and no other code path advances them.

**What is deliberately not batched: sampling.** Actions are still drawn one
environment at a time, in index order, from the same generator. Drawing a whole
batch with a single ``torch.multinomial`` would consume the generator
differently and produce a different action sequence, which would make the
equivalence test meaningless. Sampling is cheap; the network forward was the
cost. So the expensive part is batched and the RNG-consuming part is left alone.

**Why isolation cannot break here.** Rows are addressed by explicit
``(environment, seat)`` index arrays through the frozen store's ``gather``/
``scatter``, so an event gathered for environment A can only be scattered back
to environment A. The ``policy_id`` is passed through unchanged. Neither claim
is left to inspection: both are tested directly.

**Exact float equality is not claimed.** A batch-N matmul and a packed GRU
reorder floating-point work, so results differ in the last bits. Hidden states,
logits and values are required to agree to 1e-5; actions, event cursors,
sequence ownership and reset boundaries are required to agree **exactly**.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from tarneeb.buffers.recurrent import RecurrentRolloutBuffer, RecurrentTransition
from tarneeb.core.actions import NO_TRUMP_ACTION, PASS_ACTION
from tarneeb.core.cards import NUM_SEATS
from tarneeb.env.spaces import K_MAX, N_EVENT_FIELDS
from tarneeb.models.masking import MaskedCategorical, actions_are_legal
from tarneeb.training.recurrent_rollout import (
    RecurrentRolloutCollector,
    RecurrentRolloutMetrics,
    relativized_event_ids,
)


class BatchedRecurrentRolloutCollector(RecurrentRolloutCollector):
    """The frozen recurrent collector with cross-environment batching.

    Semantically identical to :class:`~tarneeb.training.recurrent_rollout.
    RecurrentRolloutCollector`; only the batching changes. Everything else --
    seating, opponent handling, settlement, boundary detection, the consumption
    invariant, resume -- is inherited unchanged.
    """

    def _one_step(self, buffer: RecurrentRolloutBuffer, metrics: RecurrentRolloutMetrics) -> None:
        environment_count = self.environments.num_envs
        acting_seats = self.environments.acting_seats()
        step_actions = np.zeros(environment_count, dtype=np.int64)

        learner_environments: list[int] = []
        learner_seat_ids: list[int] = []
        for environment_index in range(environment_count):
            seat = int(acting_seats[environment_index])
            if seat < 0:  # pragma: no cover - a terminated env is reset before this
                continue
            if self.is_learner_seat(environment_index, seat):
                learner_environments.append(environment_index)
                learner_seat_ids.append(seat)
            else:
                step_actions[environment_index] = self._opponent_action(environment_index)
            metrics.decision_count += 1

        # --- STEP 1 and 2: act and store. -----------------------------------
        if learner_environments:
            self._act_and_store_batched(
                np.array(learner_environments, dtype=np.int64),
                np.array(learner_seat_ids, dtype=np.int64),
                step_actions,
                buffer,
                metrics,
            )

        # --- STEP 3: step the engine. ---------------------------------------
        vector_step = self.environments.step(step_actions)
        metrics.environment_steps += environment_count

        # --- STEP 4: R1, batched across every environment. ------------------
        self._advance_every_environment(vector_step.events, metrics)

        # --- Settlement and boundaries, identical to the frozen collector. ---
        self._settle_and_reset(vector_step, buffer, metrics)

    # --- batched decisions --------------------------------------------------

    def _act_and_store_batched(
        self,
        environment_indices: np.ndarray,
        seats: np.ndarray,
        step_actions: np.ndarray,
        buffer: RecurrentRolloutBuffer,
        metrics: RecurrentRolloutMetrics,
    ) -> None:
        """Run one batched forward for every learner decision due this step."""
        row_count = environment_indices.size
        observation_dim = self.actor.spec.actor_observation_dim

        observations = np.zeros((row_count, observation_dim), dtype=np.float32)
        legal_masks = np.zeros((row_count, self.actor.spec.action_count), dtype=bool)
        for row, environment_index in enumerate(environment_indices):
            environment = self.environments.environments[int(environment_index)]
            observations[row] = environment.actor_observation(int(seats[row]))
            legal_masks[row] = environment.legal_action_mask()

        observation_tensor = torch.from_numpy(observations)
        legal_mask_tensor = torch.from_numpy(legal_masks)
        # R2: nothing is consumed at a decision point, so every row is empty.
        empty_ids = torch.zeros((row_count, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
        empty_lengths = torch.zeros(row_count, dtype=torch.int64)

        actor_hidden = self.state.gather_actor(environment_indices, seats, self.policy_id)
        critic_hidden = self.state.gather_critic(environment_indices, seats, self.policy_id)

        with torch.no_grad():
            logits, advanced_actor = self.actor(
                observation_tensor, empty_ids, empty_lengths, actor_hidden
            )
            values, advanced_critic = self.critic(
                observation_tensor, empty_ids, empty_lengths, critic_hidden
            )

        # R2 is a correctness property, so it is verified rather than trusted.
        if not bool(torch.equal(advanced_actor, actor_hidden)):
            raise AssertionError(
                "the actor forward advanced the hidden state at a decision point; "
                "rule R2 requires prev_ev_len == 0 to be a strict no-op"
            )
        if not bool(torch.equal(advanced_critic, critic_hidden)):
            raise AssertionError("the critic forward advanced the hidden state at a decision point")

        # Sampling stays one row at a time, in environment order, so the
        # generator is consumed exactly as the frozen collector consumes it.
        for row in range(row_count):
            environment_index = int(environment_indices[row])
            seat = int(seats[row])
            row_logits = logits[row : row + 1]
            row_mask = legal_mask_tensor[row : row + 1]
            distribution = MaskedCategorical.from_logits(row_logits, row_mask)
            with torch.no_grad():
                selected = distribution.sample(self.action_sampling_generator)
                log_probability = distribution.log_probability_of(selected)

            if not actions_are_legal(selected, row_mask):
                metrics.illegal_action_count += 1
                raise AssertionError("the policy sampled an action its legal mask forbids")
            action = int(selected.item())
            if action == NO_TRUMP_ACTION:
                metrics.no_trump_action_count += 1
                raise AssertionError("the policy selected the reserved no-trump action 64")

            if bool(legal_masks[row, PASS_ACTION]):
                metrics.auction_decision_count += 1
                if action == PASS_ACTION:
                    metrics.pass_action_count += 1

            self._store_decision(
                environment_index=environment_index,
                seat=seat,
                observation=observations[row],
                legal_action_mask=legal_masks[row],
                action=action,
                log_probability=float(log_probability.item()),
                value=float(values[row].item()),
                actor_slot=actor_hidden[:, row : row + 1],
                critic_slot=critic_hidden[:, row : row + 1],
                buffer=buffer,
                metrics=metrics,
            )
            step_actions[environment_index] = action

    def _store_decision(
        self,
        environment_index: int,
        seat: int,
        observation: np.ndarray,
        legal_action_mask: np.ndarray,
        action: int,
        log_probability: float,
        value: float,
        actor_slot: torch.Tensor,
        critic_slot: torch.Tensor,
        buffer: RecurrentRolloutBuffer,
        metrics: RecurrentRolloutMetrics,
    ) -> None:
        """Store one decision. Mirrors the frozen collector's storage exactly."""
        is_head = self.state.is_chunk_head(environment_index, seat)
        segment_ids, segment_length = self.state.stored_segment(environment_index, seat, is_head)
        metrics.max_segment_length = max(metrics.max_segment_length, segment_length)

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
                old_log_probability=log_probability,
                value=value,
                is_chunk_head=is_head,
                actor_hidden_head=(
                    actor_slot.squeeze(1).detach().numpy().copy() if is_head else None
                ),
                critic_hidden_head=(
                    critic_slot.squeeze(1).detach().numpy().copy() if is_head else None
                ),
                first_event_seq=first_event_seq,
            )
        )
        self.state.note_decision_stored(environment_index, seat, self.chunk_length)
        metrics.learner_decision_count += 1

    # --- batched event advancement -----------------------------------------

    def _advance_every_environment(
        self,
        events_by_environment: list[tuple[Any, ...]],
        metrics: RecurrentRolloutMetrics,
    ) -> None:
        """R1 for the whole vector step in two GRU calls.

        Builds one ``(4 * N, K, 7)`` block with a per-row length, so an
        environment that emitted nothing contributes zero-length rows and is
        left byte-identical by the frozen encoder's no-op path.
        """
        environment_count = self.environments.num_envs
        longest = max((len(events) for events in events_by_environment), default=0)
        if longest == 0:
            return
        if longest > K_MAX:
            raise AssertionError(
                f"one engine step emitted {longest} events, which exceeds K_MAX={K_MAX}"
            )

        row_environments = np.repeat(np.arange(environment_count, dtype=np.int64), NUM_SEATS)
        row_seats = np.tile(np.arange(NUM_SEATS, dtype=np.int64), environment_count)
        row_count = row_environments.size

        padded = np.zeros((row_count, longest, N_EVENT_FIELDS), dtype=np.int64)
        lengths = np.zeros(row_count, dtype=np.int64)
        relativized: dict[int, np.ndarray] = {}
        for environment_index in range(environment_count):
            events = events_by_environment[environment_index]
            if not events:
                continue
            count = len(events)
            per_seat = np.stack([relativized_event_ids(events, seat) for seat in range(NUM_SEATS)])
            relativized[environment_index] = per_seat
            for seat in range(NUM_SEATS):
                row = environment_index * NUM_SEATS + seat
                padded[row, :count] = per_seat[seat]
                lengths[row] = count

        id_tensor = torch.from_numpy(padded)
        length_tensor = torch.from_numpy(lengths)

        with torch.no_grad():
            actor_hidden = self.state.gather_actor(row_environments, row_seats, self.policy_id)
            critic_hidden = self.state.gather_critic(row_environments, row_seats, self.policy_id)
            advanced_actor = self.actor.trunk.history_encoder.advance(
                id_tensor, length_tensor, actor_hidden
            )
            advanced_critic = self.critic.trunk.history_encoder.advance(
                id_tensor, length_tensor, critic_hidden
            )
        self.state.scatter_actor(row_environments, row_seats, advanced_actor, self.policy_id)
        self.state.scatter_critic(row_environments, row_seats, advanced_critic, self.policy_id)

        for environment_index, per_seat in relativized.items():
            events = events_by_environment[environment_index]
            for seat in range(NUM_SEATS):
                self.state.append_events(environment_index, seat, per_seat[seat])
            self.state.last_seq_consumed[environment_index, :] = int(events[-1].seq)
            metrics.total_events_consumed += len(events) * NUM_SEATS

    # --- settlement ---------------------------------------------------------

    def _settle_and_reset(
        self,
        vector_step: object,
        buffer: RecurrentRolloutBuffer,
        metrics: RecurrentRolloutMetrics,
    ) -> None:
        """Settle finished deals and apply hand boundaries.

        A deliberate mirror of the frozen collector's tail. It is duplicated
        rather than shared because sharing would require editing the frozen
        method; the equivalence test is what keeps the two from drifting, and it
        compares every stored field rather than spot-checking.
        """
        environment_count = self.environments.num_envs
        terminated = vector_step.terminated  # type: ignore[attr-defined]
        rewards = vector_step.rewards_by_seat  # type: ignore[attr-defined]

        for environment_index in range(environment_count):
            if not terminated[environment_index]:
                continue
            rewards_by_seat = rewards[environment_index]
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

        if terminated.any():
            self.environments.reset_terminated(terminated)
            self.redeal_counts[terminated] = 0

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
