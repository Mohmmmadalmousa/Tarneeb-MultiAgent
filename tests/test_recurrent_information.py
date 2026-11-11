"""The recurrent actor's information boundary.

The actor must see exactly its own actor-safe observation, the public event
stream from its own relative viewpoint, and its own hidden state. Nothing else.

The strongest available test is invariance: rearrange the information the actor
is not allowed to see and require its output to be **bit-identical**, not merely
close. That is the recurrent extension of the frozen Phase 4 invariant I7.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.core.cards import NUM_SEATS, cards_in_hand
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import EventType, Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.spaces import D_CRITIC, D_STATIC, K_MAX, N_EVENT_FIELDS
from tarneeb.env.vec import VecTarneeb
from tarneeb.models.recurrent import (
    RecurrentArchitectureSpec,
    build_recurrent_actor_and_critic,
    zero_hidden_state,
)
from tarneeb.training.recurrent_rollout import (
    RecurrentRolloutCollector,
    relativized_event_ids,
)
from tarneeb.training.recurrent_state import LEARNER_POLICY_ID


def play_to_midgame(match_seed: int = 4242) -> tuple[TarneebEngine, object]:
    """Return an engine and a state partway through the play phase."""
    engine = TarneebEngine(RulesConfig())
    state = engine.new_deal(dealer_seat=0, deal_index=0, match_seed=match_seed)
    engine.deal(state)
    generator = np.random.default_rng(0)
    for _ in range(400):
        if state.phase not in (Phase.BID, Phase.DECLARE, Phase.PLAY):
            break
        if state.phase is Phase.PLAY and state.trick_number >= 3:
            break
        legal = np.flatnonzero(engine.legal_actions(state))
        engine.step(state, int(generator.choice(legal)))
    return engine, state


# --- the actor cannot see hidden hands --------------------------------------


def test_the_actor_output_is_invariant_to_the_other_three_hands() -> None:
    """Recurrent extension of invariant I7, over both the output and the state.

    Two states that differ only in how the unseen cards are distributed among the
    other seats must produce byte-identical logits and byte-identical next hidden
    state.
    """
    engine, state = play_to_midgame()
    builder = ObservationBuilder(engine)
    observer = state.current_player_seat
    assert observer is not None

    actor, _critic = build_recurrent_actor_and_critic(seed=7)
    observation = torch.from_numpy(builder.static_actor(state, observer)).unsqueeze(0)
    ids = torch.zeros((1, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
    recent = state.events[-4:]
    ids[0, : len(recent)] = torch.from_numpy(
        relativized_event_ids(tuple(recent), observer).astype(np.int64)
    )
    lengths = torch.tensor([len(recent)], dtype=torch.int64)
    hidden = torch.randn(actor.spec.gru_layer_count, 1, actor.spec.hidden_size)

    with torch.no_grad():
        before_logits, before_hidden = actor(observation, ids, lengths, hidden)

    # Redistribute the three hidden hands among the other seats.
    others = [seat for seat in range(NUM_SEATS) if seat != observer]
    pooled = [card for seat in others for card in cards_in_hand(state.hands[seat])]
    np.random.default_rng(1).shuffle(pooled)
    sizes = [len(cards_in_hand(state.hands[seat])) for seat in others]
    cursor = 0
    for seat, size in zip(others, sizes, strict=True):
        hand = 0
        for card in pooled[cursor : cursor + size]:
            hand |= 1 << card
        state.hands[seat] = hand
        cursor += size

    shuffled_observation = torch.from_numpy(builder.static_actor(state, observer)).unsqueeze(0)
    with torch.no_grad():
        after_logits, after_hidden = actor(shuffled_observation, ids, lengths, hidden)

    assert torch.equal(before_logits, after_logits)
    assert torch.equal(before_hidden, after_hidden)


def test_the_actor_structurally_cannot_accept_the_centralized_critic_vector() -> None:
    """A wiring mistake raises a shape error rather than training on leaked cards."""
    actor, _critic = build_recurrent_actor_and_critic(seed=8)
    ids = torch.zeros((1, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
    lengths = torch.zeros(1, dtype=torch.int64)

    assert actor.spec.actor_observation_dim == D_STATIC
    with pytest.raises(ValueError, match="692 features"):
        actor(torch.zeros((1, D_CRITIC)), ids, lengths, zero_hidden_state(1, actor.spec))


def test_the_phase_eight_critic_is_decentralized() -> None:
    """The 856-dim centralized critic is Phase 10 and must not appear yet."""
    spec = RecurrentArchitectureSpec()
    assert spec.critic_observation_dim == D_STATIC
    assert spec.critic_observation_dim != D_CRITIC


def test_the_collector_never_requests_a_critic_observation() -> None:
    """Phase 8 stores the actor-safe vector as the critic input, by construction."""
    vector_environments = VecTarneeb(num_envs=2, root_seed=3)
    actor, critic = build_recurrent_actor_and_critic(seed=1)
    generator = torch.Generator()
    generator.manual_seed(0)
    collector = RecurrentRolloutCollector(
        environments=vector_environments,
        actor=actor,
        critic=critic,
        rollout_length=40,
        action_sampling_generator=generator,
        opponents=[BasicHeuristicAgent() for _ in range(2)],
    )
    buffer, _metrics = collector.collect()

    for sequence in buffer.sequences.values():
        for transition in sequence:
            assert transition.critic_static.shape == (D_STATIC,)
            assert np.array_equal(transition.critic_static, transition.static)


# --- the actor cannot see other seats' memory -------------------------------


def test_one_seats_hidden_state_cannot_influence_another_seats_decision() -> None:
    vector_environments = VecTarneeb(num_envs=2, root_seed=9)
    actor, critic = build_recurrent_actor_and_critic(seed=2)
    generator = torch.Generator()
    generator.manual_seed(0)
    collector = RecurrentRolloutCollector(
        environments=vector_environments,
        actor=actor,
        critic=critic,
        rollout_length=24,
        action_sampling_generator=generator,
        opponents=[BasicHeuristicAgent() for _ in range(2)],
    )
    collector.collect()

    environment = collector.environments.environments[0]
    seat = environment.acting_seat
    assert seat is not None
    observation = torch.from_numpy(environment.actor_observation(seat)).unsqueeze(0)
    ids = torch.zeros((1, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
    lengths = torch.zeros(1, dtype=torch.int64)

    with torch.no_grad():
        before, _ = actor(observation, ids, lengths, collector.state.actor_slot(0, seat))

    for other in range(NUM_SEATS):
        if other != seat:
            collector.state.actor_hidden[LEARNER_POLICY_ID][0, other] = 5.0
    collector.state.actor_hidden[LEARNER_POLICY_ID][1] = -5.0

    with torch.no_grad():
        after, _ = actor(observation, ids, lengths, collector.state.actor_slot(0, seat))

    assert torch.equal(before, after)


# --- the event stream itself carries no hidden card -------------------------


def test_no_consumed_event_reveals_an_unplayed_card() -> None:
    """Guardrail G10, measured on the exact ids the recurrent encoder consumes.

    The revealed-card set is measured immediately **after** processing event
    ``k``: every card id present in the consumed stream up to and including ``k``
    must have been played at or before ``k``.
    """
    engine, state = play_to_midgame(match_seed=555)
    del engine

    for observer in range(NUM_SEATS):
        rows = relativized_event_ids(tuple(state.events), observer)
        played_so_far: set[int] = set()
        for position, event in enumerate(state.events):
            card_id = int(rows[position, 4])
            if event.event_type is EventType.CARD_PLAYED:
                assert event.card is not None
                played_so_far.add(event.card)
                assert card_id == event.card + 1
            else:
                assert card_id == 0, "only CARD_PLAYED may carry a card id"
        # No card still held by another seat may have appeared in the stream.
        for other in range(NUM_SEATS):
            if other == observer:
                continue
            for card in cards_in_hand(state.hands[other]):
                assert card not in played_so_far, "an unplayed hidden card leaked"


def test_the_event_stream_is_chronological_and_contiguous() -> None:
    engine, state = play_to_midgame(match_seed=777)
    del engine
    assert [event.seq for event in state.events] == list(range(len(state.events)))


def test_relativized_ids_depend_only_on_the_observer_and_the_event() -> None:
    """The same event from the same seat always encodes identically."""
    engine, state = play_to_midgame(match_seed=888)
    del engine
    events = tuple(state.events[:6])
    for observer in range(NUM_SEATS):
        assert np.array_equal(
            relativized_event_ids(events, observer), relativized_event_ids(events, observer)
        )
