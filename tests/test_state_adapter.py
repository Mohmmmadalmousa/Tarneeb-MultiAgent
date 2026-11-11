"""The bridge from the Phase 5 arena's view to the Phase 4 observation.

The whole point of the adapter is that there is still exactly **one**
implementation of ``obs_v2``. These tests prove it: an observation built from a
reconstructed state must be byte-identical to one built from the true state, at
every decision of many real deals, and the same for the padded history and the
legal-action mask.

They also prove the property the reconstruction relies on -- that the actor
observation does not depend on how the unseen cards are distributed -- rather
than merely assuming it from the Phase 4 leakage gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.agents.state_adapter import reconstruct_state_from_public_view
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.public_view import PublicViewBuilder

LIVE_PHASES = (Phase.BID, Phase.DECLARE, Phase.PLAY)


def walk_deals(deal_count: int, seed: int = 0):
    """Yield ``(engine, observations, state, acting_seat, view)`` at every decision."""
    rules = RulesConfig()
    engine = TarneebEngine(rules)
    observations = ObservationBuilder(engine)
    views = PublicViewBuilder(engine)
    rng = np.random.default_rng(seed)

    for deal_index in range(deal_count):
        state = engine.new_deal(deal_index % 4, deal_index=deal_index, match_seed=987_654 + seed)
        engine.deal(state)
        while state.phase in LIVE_PHASES:
            acting_seat = state.current_player_seat
            assert acting_seat is not None
            yield engine, observations, state, acting_seat, views.build(state, acting_seat)
            legal = np.flatnonzero(engine.legal_actions(state))
            engine.step(state, int(rng.choice(legal)))
            if state.phase is Phase.DEAL:
                engine.deal(state)


def test_reconstructed_observations_are_byte_identical_over_many_decisions() -> None:
    rules = RulesConfig()
    checked = 0
    for engine, observations, state, acting_seat, view in walk_deals(deal_count=12, seed=1):
        rebuilt = reconstruct_state_from_public_view(view, rules)

        assert np.array_equal(
            observations.static_actor(state, acting_seat),
            observations.static_actor(rebuilt, acting_seat),
        )
        true_ids, true_mask = observations.padded_history(state, acting_seat)
        rebuilt_ids, rebuilt_mask = observations.padded_history(rebuilt, acting_seat)
        assert np.array_equal(true_ids, rebuilt_ids)
        assert np.array_equal(true_mask, rebuilt_mask)
        assert np.array_equal(engine.legal_actions(state), engine.legal_actions(rebuilt))
        checked += 1

    assert checked > 500


def test_the_reconstruction_carries_no_real_hidden_hand() -> None:
    """The placeholder hands must have the right sizes and be nothing more."""
    rules = RulesConfig()
    for _engine, _observations, state, acting_seat, view in walk_deals(deal_count=3, seed=2):
        rebuilt = reconstruct_state_from_public_view(view, rules)
        assert rebuilt.original_hands == [0, 0, 0, 0]
        assert rebuilt.hands[acting_seat] == state.hands[acting_seat]
        for seat in range(4):
            if seat == acting_seat:
                continue
            assert bin(rebuilt.hands[seat]).count("1") == bin(state.hands[seat]).count("1")


def test_the_actor_observation_ignores_how_the_unseen_cards_are_distributed() -> None:
    """The invariant the adapter rests on, verified directly."""
    rng = np.random.default_rng(4)

    for _engine, observations, state, acting_seat, view in walk_deals(deal_count=4, seed=3):
        baseline = observations.static_actor(state, acting_seat)
        for _shuffle in range(3):
            shuffled = state.clone()
            other_seats = [seat for seat in range(4) if seat != acting_seat]
            unseen = sorted(view.unseen_cards)
            rng.shuffle(unseen)
            cursor = 0
            for seat in other_seats:
                size = bin(state.hands[seat]).count("1")
                hand = 0
                for card in unseen[cursor : cursor + size]:
                    hand |= 1 << card
                shuffled.hands[seat] = hand
                cursor += size
            assert np.array_equal(baseline, observations.static_actor(shuffled, acting_seat))


def test_an_inconsistent_view_is_refused() -> None:
    """A view whose card accounting does not add up must fail, not silently pass."""
    rules = RulesConfig()
    _engine, _observations, _state, _seat, view = next(walk_deals(deal_count=1, seed=5))
    broken = type(view)(**{**view.__dict__, "own_hand": view.own_hand[:-1]})
    with pytest.raises(ValueError, match="cards"):
        reconstruct_state_from_public_view(broken, rules)
