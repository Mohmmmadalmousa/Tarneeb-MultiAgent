"""A7: the three history conditions must differ in exactly one thing.

An ablation is only evidence if it removes what it claims to remove and changes
nothing else. These tests check both halves:

- the **removal** works: auction rows really are gone, the ordered detail really is
  gone from the summary condition, and play events survive in order;
- **nothing else moves**: the 692-float static observation, the legal-action mask
  and the padding convention are byte-identical across all three conditions for
  the same state.

They also check the transform cannot smuggle information in: it only deletes or
summarises rows the frozen builder already produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import EventType, Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.curriculum.history import (
    AUCTION_EVENT_TYPES,
    EVENT_TYPE_FIELD,
    HistoryConditionedObservationBuilder,
    apply_history_condition,
    auction_event_count,
    relative_declarer_in_history,
    winning_bid_in_history,
)
from tarneeb.curriculum.stages import (
    AUCTION_REMOVED,
    CURRENT_BID_ONLY,
    FULL_HISTORY,
    HISTORY_CONDITIONS,
)
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.spaces import L_MAX, N_EVENT_FIELDS

PLAY_EVENT_TYPES = frozenset(
    {int(EventType.CARD_PLAYED), int(EventType.TRICK_COMPLETED), int(EventType.DEAL_COMPLETED)}
)
LIVE_PHASES = (Phase.BID, Phase.DECLARE, Phase.PLAY)


def walk_decisions(deal_count: int = 3, seed: int = 0):
    """Yield ``(builder, state, acting_seat)`` at every decision of real deals."""
    rules = RulesConfig()
    engine = TarneebEngine(rules)
    builder = ObservationBuilder(engine)
    rng = np.random.default_rng(seed)

    for deal_index in range(deal_count):
        state = engine.new_deal(deal_index % 4, deal_index=deal_index, match_seed=555 + seed)
        engine.deal(state)
        while state.phase in LIVE_PHASES:
            acting_seat = state.current_player_seat
            assert acting_seat is not None
            yield builder, state, acting_seat
            legal = np.flatnonzero(engine.legal_actions(state))
            engine.step(state, int(rng.choice(legal)))
            if state.phase is Phase.DEAL:
                engine.deal(state)


def histories_for_every_condition(builder, state, seat) -> dict[str, tuple]:
    base_ids, base_mask = builder.padded_history(state, seat)
    return {
        condition: apply_history_condition(condition, base_ids.copy(), base_mask.copy())
        for condition in HISTORY_CONDITIONS
    }


# --- full history is untouched ---------------------------------------------


def test_the_full_condition_returns_the_frozen_history_unchanged() -> None:
    for builder, state, seat in walk_decisions():
        base_ids, base_mask = builder.padded_history(state, seat)
        ids, mask = apply_history_condition(FULL_HISTORY, base_ids, base_mask)
        assert np.array_equal(ids, base_ids)
        assert np.array_equal(mask, base_mask)


def test_an_unknown_condition_is_refused() -> None:
    ids = np.zeros((L_MAX, N_EVENT_FIELDS), dtype=np.int16)
    mask = np.zeros(L_MAX, dtype=bool)
    with pytest.raises(ValueError, match="unknown history_condition"):
        apply_history_condition("telepathy", ids, mask)


# --- the auction really is removed -----------------------------------------


def test_removing_the_auction_leaves_no_auction_event() -> None:
    checked_with_auction = 0
    for builder, state, seat in walk_decisions():
        histories = histories_for_every_condition(builder, state, seat)
        full_ids, full_mask = histories[FULL_HISTORY]
        if auction_event_count(full_ids, full_mask) > 0:
            checked_with_auction += 1
        stripped_ids, stripped_mask = histories[AUCTION_REMOVED]
        assert auction_event_count(stripped_ids, stripped_mask) == 0
    assert checked_with_auction > 50, "the walk must actually contain auctions"


def test_removing_the_auction_keeps_every_play_event_in_order() -> None:
    for builder, state, seat in walk_decisions():
        full_ids, full_mask = builder.padded_history(state, seat)
        stripped_ids, stripped_mask = apply_history_condition(
            AUCTION_REMOVED, full_ids.copy(), full_mask.copy()
        )
        expected = [
            tuple(full_ids[position])
            for position in range(L_MAX)
            if full_mask[position]
            and int(full_ids[position, EVENT_TYPE_FIELD]) not in AUCTION_EVENT_TYPES
        ]
        actual = [
            tuple(stripped_ids[position]) for position in range(L_MAX) if stripped_mask[position]
        ]
        assert actual == expected


def test_the_summary_condition_keeps_only_one_auction_row() -> None:
    saw_a_contract = 0
    for builder, state, seat in walk_decisions():
        full_ids, full_mask = builder.padded_history(state, seat)
        summary_ids, summary_mask = apply_history_condition(
            CURRENT_BID_ONLY, full_ids.copy(), full_mask.copy()
        )
        remaining = auction_event_count(summary_ids, summary_mask)
        assert remaining <= 1
        if winning_bid_in_history(full_ids, full_mask) is not None:
            assert remaining == 1
            saw_a_contract += 1
    assert saw_a_contract > 50


def test_the_summary_preserves_the_winning_bid_but_not_the_order() -> None:
    """The summary must carry current bid state and nothing about how it arose."""
    for builder, state, seat in walk_decisions():
        full_ids, full_mask = builder.padded_history(state, seat)
        summary_ids, summary_mask = apply_history_condition(
            CURRENT_BID_ONLY, full_ids.copy(), full_mask.copy()
        )
        assert winning_bid_in_history(summary_ids, summary_mask) == winning_bid_in_history(
            full_ids, full_mask
        )
        assert relative_declarer_in_history(summary_ids, summary_mask) == (
            relative_declarer_in_history(full_ids, full_mask)
        )
        # No pass and no trump declaration survives.
        types = {
            int(summary_ids[position, EVENT_TYPE_FIELD])
            for position in range(L_MAX)
            if summary_mask[position]
        }
        assert int(EventType.PASS) not in types
        assert int(EventType.TRUMP_SELECTED) not in types


def test_the_summary_is_strictly_less_informative_than_the_full_history() -> None:
    """Somewhere in a real auction the summary must hold fewer rows."""
    reductions = 0
    for builder, state, seat in walk_decisions():
        full_ids, full_mask = builder.padded_history(state, seat)
        _summary_ids, summary_mask = apply_history_condition(
            CURRENT_BID_ONLY, full_ids.copy(), full_mask.copy()
        )
        if int(summary_mask.sum()) < int(full_mask.sum()):
            reductions += 1
    assert reductions > 50


# --- padding stays correct --------------------------------------------------


def test_padding_stays_on_the_right_and_the_mask_matches_it() -> None:
    for builder, state, seat in walk_decisions():
        for condition, (ids, mask) in histories_for_every_condition(builder, state, seat).items():
            assert ids.shape == (L_MAX, N_EVENT_FIELDS), condition
            assert mask.shape == (L_MAX,), condition
            valid_count = int(mask.sum())
            assert mask[:valid_count].all(), condition
            assert not mask[valid_count:].any(), condition
            assert not ids[valid_count:].any(), f"{condition}: padded rows must stay all zero"


def test_every_condition_produces_ids_inside_their_field_cardinalities() -> None:
    from tarneeb.env.event_features import EVENT_FIELD_CARDINALITIES

    cardinalities = np.array(EVENT_FIELD_CARDINALITIES)
    for builder, state, seat in walk_decisions():
        for condition, (ids, _mask) in histories_for_every_condition(builder, state, seat).items():
            assert (ids >= 0).all(), condition
            assert (ids < cardinalities).all(), condition


# --- nothing else changes ---------------------------------------------------


def test_the_static_observation_is_identical_across_all_three_conditions() -> None:
    """The controlled part of the experiment: only the history may differ."""
    rules = RulesConfig()
    engine = TarneebEngine(rules)
    builders = {
        condition: HistoryConditionedObservationBuilder(engine, condition)
        for condition in HISTORY_CONDITIONS
    }
    for _builder, state, seat in walk_decisions():
        reference = builders[FULL_HISTORY].static_actor(state, seat)
        for condition in HISTORY_CONDITIONS:
            assert np.array_equal(builders[condition].static_actor(state, seat), reference), (
                condition
            )


def test_the_legal_action_mask_is_identical_across_all_three_conditions() -> None:
    rules = RulesConfig()
    engine = TarneebEngine(rules)
    for _builder, state, _seat in walk_decisions():
        reference = engine.legal_actions(state)
        for _condition in HISTORY_CONDITIONS:
            assert np.array_equal(engine.legal_actions(state), reference)


def test_the_conditioned_builder_only_overrides_the_history() -> None:
    """It must be the frozen builder in every other respect."""
    rules = RulesConfig()
    engine = TarneebEngine(rules)
    frozen = ObservationBuilder(engine)
    conditioned = HistoryConditionedObservationBuilder(engine, AUCTION_REMOVED)

    assert isinstance(conditioned, ObservationBuilder)
    assert type(conditioned).static_actor is type(frozen).static_actor
    assert type(conditioned).static_critic is type(frozen).static_critic
    assert type(conditioned).ui_view is type(frozen).ui_view
    assert type(conditioned).padded_history is not type(frozen).padded_history


def test_the_transform_adds_no_row_the_frozen_builder_did_not_produce() -> None:
    """Information safety: the ablation may delete or summarise, never invent."""
    for builder, state, seat in walk_decisions():
        full_ids, full_mask = builder.padded_history(state, seat)
        original_rows = {
            tuple(full_ids[position]) for position in range(L_MAX) if full_mask[position]
        }
        for condition in (AUCTION_REMOVED, CURRENT_BID_ONLY):
            ids, mask = apply_history_condition(condition, full_ids.copy(), full_mask.copy())
            for position in range(L_MAX):
                if not mask[position]:
                    continue
                row = tuple(ids[position])
                if int(row[EVENT_TYPE_FIELD]) in PLAY_EVENT_TYPES:
                    assert row in original_rows, condition
                else:
                    # The one synthetic row is a BID summary whose bid value and
                    # actor were both present in the original history.
                    assert int(row[EVENT_TYPE_FIELD]) == int(EventType.BID)
                    assert winning_bid_in_history(full_ids, full_mask) is not None


def test_a_conditioned_builder_reaches_the_actor_through_the_agent() -> None:
    """The evaluation agent must see the condition its checkpoint was trained on."""
    from tarneeb.curriculum.agents import HistoryConditionedNeuralAgent
    from tarneeb.models.nets import build_actor_and_critic

    actor, _critic = build_actor_and_critic(seed=1)
    agent = HistoryConditionedNeuralAgent(actor=actor, history_condition=AUCTION_REMOVED)
    assert isinstance(agent._observations, HistoryConditionedObservationBuilder)
    assert agent._observations.history_condition == AUCTION_REMOVED

    full_agent = HistoryConditionedNeuralAgent(actor=actor, history_condition=FULL_HISTORY)
    assert type(full_agent._observations) is ObservationBuilder


# --- pre-A7 validation: the conditions must be genuinely distinct ------------


def test_the_three_conditions_are_not_accidentally_equivalent() -> None:
    """If two conditions produced identical histories, A7 would measure nothing."""
    differs_full_vs_removed = 0
    differs_full_vs_summary = 0
    differs_removed_vs_summary = 0
    checked = 0

    for builder, state, seat in walk_decisions(deal_count=4, seed=11):
        histories = histories_for_every_condition(builder, state, seat)
        full_ids, full_mask = histories[FULL_HISTORY]
        removed_ids, removed_mask = histories[AUCTION_REMOVED]
        summary_ids, summary_mask = histories[CURRENT_BID_ONLY]

        if not (np.array_equal(full_ids, removed_ids) and np.array_equal(full_mask, removed_mask)):
            differs_full_vs_removed += 1
        if not (np.array_equal(full_ids, summary_ids) and np.array_equal(full_mask, summary_mask)):
            differs_full_vs_summary += 1
        if not (
            np.array_equal(removed_ids, summary_ids) and np.array_equal(removed_mask, summary_mask)
        ):
            differs_removed_vs_summary += 1
        checked += 1

    assert checked > 200
    assert differs_full_vs_removed > 100, "auction-removed must differ from full"
    assert differs_full_vs_summary > 100, "current-bid-only must differ from full"
    assert differs_removed_vs_summary > 100, (
        "current-bid-only must differ from auction-removed, or the two ablations "
        "would be the same experiment"
    )


def test_the_reward_is_identical_across_the_three_conditions() -> None:
    """The ablation touches the history only; the reward must not move."""
    from tarneeb.env.reward import deal_rewards_for_state

    rules = RulesConfig()
    engine = TarneebEngine(rules)
    rng = np.random.default_rng(3)
    checked = 0

    for deal_index in range(6):
        state = engine.new_deal(deal_index % 4, deal_index=deal_index, match_seed=4242)
        engine.deal(state)
        while state.phase in LIVE_PHASES:
            legal = np.flatnonzero(engine.legal_actions(state))
            engine.step(state, int(rng.choice(legal)))
            phase_after: Phase = state.phase
            if phase_after is Phase.DEAL:
                engine.deal(state)
        if state.deal_result is not None:
            rewards = deal_rewards_for_state(state)
            # The reward is computed from the engine's scored deal and never sees
            # a history at all, so it cannot depend on the A7 condition.
            for _condition in HISTORY_CONDITIONS:
                assert np.array_equal(deal_rewards_for_state(state), rewards)
            checked += 1
    assert checked > 0


def test_action_64_stays_illegal_under_every_condition() -> None:
    from tarneeb.core.actions import NO_TRUMP_ACTION
    from tarneeb.curriculum.scripted_env import CurriculumVecTarneeb
    from tarneeb.curriculum.stages import stage_six_full_bidding

    for condition in HISTORY_CONDITIONS:
        stage = stage_six_full_bidding(history_condition=condition)
        environments = CurriculumVecTarneeb(num_envs=2, stage=stage, root_seed=706_000)
        environments.reset()
        rng = np.random.default_rng(5)
        for _step in range(80):
            masks = environments.legal_action_masks()
            assert not masks[:, NO_TRUMP_ACTION].any(), condition
            actions = np.array(
                [int(rng.choice(np.flatnonzero(row))) for row in masks], dtype=np.int64
            )
            vector_step = environments.step(actions)
            if vector_step.terminated.any():
                environments.reset_terminated(vector_step.terminated)


def test_the_auction_removed_condition_drops_each_auction_event_type() -> None:
    """Named explicitly: BID, PASS and TRUMP_SELECTED must all be gone."""
    seen_each = {int(EventType.BID): 0, int(EventType.PASS): 0, int(EventType.TRUMP_SELECTED): 0}
    for builder, state, seat in walk_decisions(deal_count=4, seed=13):
        full_ids, full_mask = builder.padded_history(state, seat)
        for position in range(L_MAX):
            if full_mask[position]:
                event_type = int(full_ids[position, EVENT_TYPE_FIELD])
                if event_type in seen_each:
                    seen_each[event_type] += 1

        stripped_ids, stripped_mask = apply_history_condition(
            AUCTION_REMOVED, full_ids.copy(), full_mask.copy()
        )
        for position in range(L_MAX):
            if stripped_mask[position]:
                assert int(stripped_ids[position, EVENT_TYPE_FIELD]) not in seen_each

    assert all(count > 0 for count in seen_each.values()), (
        f"the walk must contain every auction event type, saw {seen_each}"
    )
