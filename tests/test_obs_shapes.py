"""Shape, dtype and layout contract for obs_v2.

Sizes are transcribed by hand from the blueprint's section 5.3 feature table,
not read back from the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.core.actions import PASS_ACTION, action_for_bid, action_for_trump_suit
from tarneeb.core.cards import NUM_CARDS, NUM_SEATS, NUM_SUITS, cards_in_hand, suit_of
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TRICKS_PER_DEAL, TarneebState, team_of_seat
from tarneeb.env.event_features import EVENT_ID_DTYPE
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.spaces import (
    ACTOR_FEATURE_BLOCKS,
    ACTOR_FEATURE_SLICES,
    CRITIC_FEATURE_SLICES,
    CRITIC_ONLY_FEATURE_BLOCKS,
    D_CRITIC,
    D_CRITIC_ONLY,
    D_EVENT_ONEHOT,
    D_STATIC,
    K_MAX,
    L_MAX,
    N_EVENT_FIELDS,
    OBS_SPEC_VERSION,
    absolute_seat,
    relative_seat,
)

# The blueprint's section 5.3 table, written out by hand.
EXPECTED_ACTOR_BLOCK_WIDTHS = {
    "own_hand": 52,
    "played_cards_by_relative_seat": 208,
    "current_trick_by_relative_seat": 208,
    "unseen_card_mask": 52,
    "legal_action_mask": 65,
    "void_matrix": 16,
    "play_order_in_trick": 5,
    "led_suit": 5,
    "current_high_bid": 8,
    "high_bidder_relative": 5,
    "pass_status_relative": 4,
    "trump": 6,
    "contract": 8,
    "declarer_role": 4,
    "tricks_won_one_hot": 28,
    "tricks_remaining": 1,
    "tricks_still_needed": 1,
    "match_scores": 2,
    "score_gap": 1,
    "dealer_relative": 4,
    "leader_relative": 4,
    "phase": 3,
    "trumps_unaccounted_for": 1,
    "trick_number": 1,
}

EXPECTED_CRITIC_ONLY_BLOCK_WIDTHS = {
    "hidden_hands_by_relative_seat": 156,
    "trump_count_by_relative_seat": 4,
    "high_card_count_by_relative_seat": 4,
}


@pytest.fixture
def engine() -> TarneebEngine:
    return TarneebEngine(RulesConfig())


@pytest.fixture
def builder(engine: TarneebEngine) -> ObservationBuilder:
    return ObservationBuilder(engine)


def state_in_play(engine: TarneebEngine, match_seed: int = 42) -> TarneebState:
    state = engine.deal(engine.new_deal(0, match_seed=match_seed))
    engine.step(state, action_for_bid(9))
    for _ in range(NUM_SEATS - 1):
        engine.step(state, PASS_ACTION)
    engine.step(state, action_for_trump_suit(2))
    return state


# --- version and totals -----------------------------------------------------


def test_obs_spec_version() -> None:
    assert OBS_SPEC_VERSION == "obs_v2"


def test_frozen_dimensions() -> None:
    assert D_STATIC == 692
    assert D_CRITIC_ONLY == 164
    assert D_CRITIC == 856
    assert D_CRITIC == D_STATIC + D_CRITIC_ONLY
    assert N_EVENT_FIELDS == 7
    assert D_EVENT_ONEHOT == 99
    assert L_MAX == 80
    assert K_MAX == 16


def test_actor_block_widths_match_the_specification() -> None:
    assert dict(ACTOR_FEATURE_BLOCKS) == EXPECTED_ACTOR_BLOCK_WIDTHS
    assert sum(EXPECTED_ACTOR_BLOCK_WIDTHS.values()) == 692


def test_critic_only_block_widths_match_the_specification() -> None:
    assert dict(CRITIC_ONLY_FEATURE_BLOCKS) == EXPECTED_CRITIC_ONLY_BLOCK_WIDTHS
    assert sum(EXPECTED_CRITIC_ONLY_BLOCK_WIDTHS.values()) == 164


def test_actor_blocks_tile_the_vector_without_gaps_or_overlap() -> None:
    covered = np.zeros(D_STATIC, dtype=int)
    for block_name in EXPECTED_ACTOR_BLOCK_WIDTHS:
        covered[ACTOR_FEATURE_SLICES[block_name]] += 1
    assert covered.min() == 1 and covered.max() == 1


def test_critic_blocks_tile_the_vector_without_gaps_or_overlap() -> None:
    covered = np.zeros(D_CRITIC, dtype=int)
    for block_name in {**EXPECTED_ACTOR_BLOCK_WIDTHS, **EXPECTED_CRITIC_ONLY_BLOCK_WIDTHS}:
        covered[CRITIC_FEATURE_SLICES[block_name]] += 1
    assert covered.min() == 1 and covered.max() == 1


def test_critic_slices_preserve_the_actor_layout() -> None:
    for block_name in EXPECTED_ACTOR_BLOCK_WIDTHS:
        assert CRITIC_FEATURE_SLICES[block_name] == ACTOR_FEATURE_SLICES[block_name]


# --- shapes and dtypes ------------------------------------------------------


def test_actor_observation_shape_and_dtype(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    observation = builder.static_actor(state_in_play(engine), 0)
    assert observation.shape == (692,)
    assert observation.dtype == np.float32


def test_critic_observation_shape_and_dtype(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    observation = builder.static_critic(state_in_play(engine), 0)
    assert observation.shape == (856,)
    assert observation.dtype == np.float32


def test_critic_contains_the_actor_vector_unchanged(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_in_play(engine)
    for observer_seat in range(NUM_SEATS):
        actor_observation = builder.static_actor(state, observer_seat)
        critic_observation = builder.static_critic(state, observer_seat)
        assert np.array_equal(critic_observation[:D_STATIC], actor_observation)


def test_padded_history_shapes_and_dtypes(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    padded_event_history, valid_mask = builder.padded_history(state_in_play(engine), 0)
    assert padded_event_history.shape == (80, 7)
    assert padded_event_history.dtype == EVENT_ID_DTYPE == np.int16
    assert valid_mask.shape == (80,)
    assert valid_mask.dtype == np.bool_


@pytest.mark.parametrize("observer_seat", range(NUM_SEATS))
def test_observations_build_for_every_seat(
    engine: TarneebEngine, builder: ObservationBuilder, observer_seat: int
) -> None:
    state = state_in_play(engine)
    assert builder.static_actor(state, observer_seat).shape == (692,)
    assert builder.static_critic(state, observer_seat).shape == (856,)


@pytest.mark.parametrize("observer_seat", [-1, 4, 9])
def test_invalid_observer_seat_is_rejected(
    engine: TarneebEngine, builder: ObservationBuilder, observer_seat: int
) -> None:
    with pytest.raises(ValueError, match="observer_seat must be in"):
        builder.static_actor(state_in_play(engine), observer_seat)


# --- feature blocks, checked individually -----------------------------------


def test_own_hand_block_is_the_observers_hand(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_in_play(engine)
    for observer_seat in range(NUM_SEATS):
        observation = builder.static_actor(state, observer_seat)
        own_hand_block = observation[ACTOR_FEATURE_SLICES["own_hand"]]
        assert set(np.flatnonzero(own_hand_block).tolist()) == set(
            cards_in_hand(state.hands[observer_seat])
        )


def test_legal_action_mask_block_matches_the_engine(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_in_play(engine)
    acting_seat = state.current_player_seat
    observation = builder.static_actor(state, acting_seat)
    mask_block = observation[ACTOR_FEATURE_SLICES["legal_action_mask"]]
    assert np.array_equal(mask_block.astype(bool), engine.legal_actions(state))


def test_played_cards_block_is_indexed_by_relative_seat(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_in_play(engine)
    rng = np.random.default_rng(0)
    for _ in range(9):  # play two full tricks and one extra card
        engine.step(state, int(rng.choice(np.flatnonzero(engine.legal_actions(state)))))

    for observer_seat in range(NUM_SEATS):
        observation = builder.static_actor(state, observer_seat)
        played_block = observation[ACTOR_FEATURE_SLICES["played_cards_by_relative_seat"]]
        for relative in range(NUM_SEATS):
            seat = absolute_seat(relative, observer_seat)
            expected_cards = {
                card
                for trick in state.completed_tricks
                for played_seat, card in trick
                if played_seat == seat
            } | {card for played_seat, card in state.current_trick if played_seat == seat}
            block = played_block[relative * NUM_CARDS : (relative + 1) * NUM_CARDS]
            assert set(np.flatnonzero(block).tolist()) == expected_cards


def test_current_trick_block_is_indexed_by_relative_seat(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_in_play(engine)
    engine.step(state, int(np.flatnonzero(engine.legal_actions(state))[0]))
    engine.step(state, int(np.flatnonzero(engine.legal_actions(state))[0]))

    for observer_seat in range(NUM_SEATS):
        observation = builder.static_actor(state, observer_seat)
        trick_block = observation[ACTOR_FEATURE_SLICES["current_trick_by_relative_seat"]]
        assert trick_block.sum() == len(state.current_trick)
        for seat, card in state.current_trick:
            relative = relative_seat(seat, observer_seat)
            assert trick_block[relative * NUM_CARDS + card] == 1.0


def test_phase_block_is_one_hot_over_bid_declare_play(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = engine.deal(engine.new_deal(0, match_seed=7))
    assert builder.static_actor(state, 0)[ACTOR_FEATURE_SLICES["phase"]].tolist() == [1, 0, 0]

    engine.step(state, action_for_bid(8))
    for _ in range(NUM_SEATS - 1):
        engine.step(state, PASS_ACTION)
    assert state.phase is Phase.DECLARE
    assert builder.static_actor(state, 0)[ACTOR_FEATURE_SLICES["phase"]].tolist() == [0, 1, 0]

    engine.step(state, action_for_trump_suit(1))
    assert builder.static_actor(state, 0)[ACTOR_FEATURE_SLICES["phase"]].tolist() == [0, 0, 1]


def test_trump_and_contract_blocks(engine: TarneebEngine, builder: ObservationBuilder) -> None:
    state = state_in_play(engine)  # bid 9, trump 2 (hearts)
    observation = builder.static_actor(state, 0)
    trump_block = observation[ACTOR_FEATURE_SLICES["trump"]]
    assert trump_block.tolist() == [0, 0, 0, 1, 0, 0]  # none, C, D, H, S, NT
    assert trump_block[5] == 0.0  # the reserved no-trump slot is never set

    contract_block = observation[ACTOR_FEATURE_SLICES["contract"]]
    assert np.flatnonzero(contract_block).tolist() == [9 - 7 + 1]


def test_declarer_role_block_is_relative(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_in_play(engine)
    declarer_seat = state.declarer_seat
    roles = {}
    for observer_seat in range(NUM_SEATS):
        block = builder.static_actor(state, observer_seat)[ACTOR_FEATURE_SLICES["declarer_role"]]
        roles[observer_seat] = int(np.flatnonzero(block)[0])

    assert roles[declarer_seat] == 0  # self
    assert roles[(declarer_seat + 2) % NUM_SEATS] == 1  # partner
    assert roles[(declarer_seat + 1) % NUM_SEATS] == 2  # opponent
    assert roles[(declarer_seat + 3) % NUM_SEATS] == 2  # opponent


def test_declarer_role_is_none_before_the_auction_ends(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = engine.deal(engine.new_deal(0, match_seed=3))
    block = builder.static_actor(state, 0)[ACTOR_FEATURE_SLICES["declarer_role"]]
    assert block.tolist() == [0, 0, 0, 1]


def test_tricks_won_block_is_two_one_hots(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_in_play(engine)
    rng = np.random.default_rng(1)
    for _ in range(12):
        engine.step(state, int(rng.choice(np.flatnonzero(engine.legal_actions(state)))))

    for observer_seat in range(NUM_SEATS):
        block = builder.static_actor(state, observer_seat)[
            ACTOR_FEATURE_SLICES["tricks_won_one_hot"]
        ]
        observer_team = team_of_seat(observer_seat)
        assert block[: TRICKS_PER_DEAL + 1].sum() == 1.0
        assert block[TRICKS_PER_DEAL + 1 :].sum() == 1.0
        assert block[state.tricks_won[observer_team]] == 1.0
        assert block[(TRICKS_PER_DEAL + 1) + state.tricks_won[1 - observer_team]] == 1.0


def test_void_matrix_is_relative(engine: TarneebEngine, builder: ObservationBuilder) -> None:
    state = state_in_play(engine)
    state.void_flags[2][1] = True  # seat 2 is void in diamonds

    for observer_seat in range(NUM_SEATS):
        block = builder.static_actor(state, observer_seat)[ACTOR_FEATURE_SLICES["void_matrix"]]
        relative = relative_seat(2, observer_seat)
        assert block[relative * NUM_SUITS + 1] == 1.0
        assert block.sum() == 1.0


def test_led_suit_block(engine: TarneebEngine, builder: ObservationBuilder) -> None:
    state = state_in_play(engine)
    none_block = builder.static_actor(state, 0)[ACTOR_FEATURE_SLICES["led_suit"]]
    assert none_block.tolist() == [1, 0, 0, 0, 0]

    led_card = int(np.flatnonzero(engine.legal_actions(state))[0])
    engine.step(state, led_card)
    led_block = builder.static_actor(state, 0)[ACTOR_FEATURE_SLICES["led_suit"]]
    assert led_block[1 + suit_of(led_card)] == 1.0
    assert led_block.sum() == 1.0


def test_scalar_blocks_are_normalised(engine: TarneebEngine, builder: ObservationBuilder) -> None:
    state = state_in_play(engine)
    observation = builder.static_actor(state, 0)
    assert observation[ACTOR_FEATURE_SLICES["tricks_remaining"]][0] == pytest.approx(1.0)
    assert observation[ACTOR_FEATURE_SLICES["trick_number"]][0] == pytest.approx(0.0)
    assert 0.0 <= observation[ACTOR_FEATURE_SLICES["trumps_unaccounted_for"]][0] <= 1.0
    assert observation[ACTOR_FEATURE_SLICES["match_scores"]].tolist() == [0.0, 0.0]
    assert observation[ACTOR_FEATURE_SLICES["score_gap"]][0] == 0.0


def test_match_score_scalars_are_clipped(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_in_play(engine)
    state.match_score = [10_000, -10_000]
    observation = builder.static_actor(state, 0)
    assert observation[ACTOR_FEATURE_SLICES["match_scores"]].tolist() == [1.5, -1.5]
    assert observation[ACTOR_FEATURE_SLICES["score_gap"]][0] == 1.5


def test_all_actor_values_are_finite_and_bounded(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_in_play(engine)
    rng = np.random.default_rng(2)
    while state.phase is Phase.PLAY:
        observation = builder.static_actor(state, state.current_player_seat)
        assert np.isfinite(observation).all()
        assert observation.min() >= -1.5 and observation.max() <= 1.5
        engine.step(state, int(rng.choice(np.flatnonzero(engine.legal_actions(state)))))


# --- padded history ---------------------------------------------------------


def test_padded_history_is_chronological_and_right_padded(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_in_play(engine)
    padded_event_history, valid_mask = builder.padded_history(state, 0)

    history_length = len(state.events)
    assert valid_mask[:history_length].all()
    assert not valid_mask[history_length:].any()
    assert (padded_event_history[history_length:] == 0).all()  # PAD rows are all zeros

    for position, event in enumerate(state.events):
        assert np.array_equal(padded_event_history[position], builder.event_ids(event, 0))


def test_padded_history_rejects_overflow(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    """Truncating history would silently corrupt the record, so it must raise."""
    state = state_in_play(engine)
    state.events = list(state.events) * 40  # 200 events, well over L_MAX
    with pytest.raises(ValueError, match="exceeds L_MAX"):
        builder.padded_history(state, 0)


def test_full_deal_history_fits_within_l_max(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_in_play(engine)
    rng = np.random.default_rng(4)
    while state.phase is Phase.PLAY:
        engine.step(state, int(rng.choice(np.flatnonzero(engine.legal_actions(state)))))

    _padded, valid_mask = builder.padded_history(state, 0)
    assert int(valid_mask.sum()) == len(state.events) <= L_MAX
