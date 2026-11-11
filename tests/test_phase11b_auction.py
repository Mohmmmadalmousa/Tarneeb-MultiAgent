from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest

from tarneeb.auction_repair import (
    AuctionTeacher,
    collect_phase11b_probe,
    summarize_learner_auction_metrics,
)
from tarneeb.auction_repair.diagnostics import summarize_probe_auction_states
from tarneeb.core.actions import PASS_ACTION, action_for_bid
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.env.public_view import PublicViewBuilder


@pytest.mark.parametrize("team", [0, 1])
def test_summary_pass_opportunities_team_and_repeatability(team: int) -> None:
    probe = collect_phase11b_probe(seed_base=800_000, deal_count=4, learner_team=team)
    result = summarize_probe_auction_states(probe)
    assert result.learner_team == team
    assert all(s.seat % 2 == team for s in probe.states)
    assert result.pass_count == sum(s.phase is Phase.BID for s in probe.states)
    assert result.pass_count > 0
    assert result.legal_actions == sum(int(s.legal_action_mask.sum()) for s in probe.states)
    assert result == summarize_probe_auction_states(
        collect_phase11b_probe(seed_base=800_000, deal_count=4, learner_team=team)
    )
    assert (
        summarize_probe_auction_states(
            collect_phase11b_probe(deal_count=0, learner_team=team)
        ).learner_team
        == team
    )


def test_metric_denominators_exclude_opponents_and_non_bidding() -> None:
    records = [
        {
            "is_learner": True,
            "was_auction_decision": True,
            "current_high_bid": high,
            "action": action,
        }
        for high, action in [(None, 52), (None, 59), (7, 53), (12, 59), (13, 59)]
    ]
    records += [
        dict(records[0], is_learner=False),
        dict(records[0], was_auction_decision=False),
        dict(records[0], action=60),
    ]
    result = summarize_learner_auction_metrics(records)
    assert result["learner_auction_decisions"] == 5
    assert result["learner_pass_actions"] == 3
    assert result["learner_pass_rate"] == 3 / 5
    assert result["learner_opening_opportunities"] == 2
    assert result["learner_raise_opportunities"] == 2
    assert result["learner_opening_rate"] == result["learner_raise_rate"] == 0.5
    empty = summarize_learner_auction_metrics([])
    assert empty["learner_pass_rate"] is None
    assert empty["learner_opening_rate"] is None
    assert empty["learner_raise_rate"] is None


@pytest.mark.parametrize("invalid", [-1, 0, 65])
def test_legacy_teacher_rejects_invalid_agent_output(invalid: int) -> None:
    engine = TarneebEngine(RulesConfig())
    state = engine.new_deal(0, match_seed=800_001)
    engine.deal(state)
    seat = state.current_player_seat
    assert seat is not None
    teacher = AuctionTeacher()
    teacher._agent = Mock(act=Mock(return_value=invalid))
    with pytest.raises(ValueError, match="illegal label"):
        teacher.choose_action(state, seat)


def test_learner_only_auction_metrics_are_explicit() -> None:
    records = [
        {
            "owner": "learner",
            "is_learner": True,
            "action": PASS_ACTION,
            "was_auction_decision": True,
        },
        {
            "owner": "opponent",
            "is_learner": False,
            "action": PASS_ACTION,
            "was_auction_decision": True,
        },
        {
            "owner": "learner",
            "is_learner": True,
            "action": action_for_bid(7),
            "was_auction_decision": True,
        },
        {
            "owner": "learner",
            "is_learner": True,
            "action": action_for_bid(9),
            "was_auction_decision": False,
        },
    ]

    metrics = summarize_learner_auction_metrics(records)

    assert metrics["learner_auction_decisions"] == 2
    assert metrics["learner_pass_actions"] == 1
    assert metrics["learner_pass_rate"] == 0.5
    assert metrics["denominator_owner"] == "learner"
    assert metrics["owner_scope"] == "learner_only"


def test_public_auction_teacher_returns_legal_action() -> None:
    engine = TarneebEngine(RulesConfig())
    state = engine.new_deal(0, deal_index=0, match_seed=12345)
    engine.deal(state)
    seat = state.current_player_seat
    assert seat is not None
    view = PublicViewBuilder(engine).build(state, seat)

    teacher = AuctionTeacher(strategy="strong")
    action = teacher.choose_action(state, seat, view=view)
    legal_mask = engine.legal_actions(state)

    assert bool(legal_mask[action])
    assert action != 64
    assert teacher.last_input_is_public is True
    assert teacher.last_hidden_card_leak is False


def test_phase11b_probe_collects_auction_states() -> None:
    probe = collect_phase11b_probe(seed_base=800_000, deal_count=4)
    assert len(probe) >= 1
    first = probe.states[0]
    assert first.phase in (Phase.BID, Phase.DECLARE)
    assert np.asarray(first.legal_action_mask).shape == (65,)


def test_public_teacher_is_invariant_to_hidden_determinizations() -> None:
    """Two states with the same actor-visible auction information must yield the same target.

    The only differences here are hidden-card assignments among the opponents. The public view,
    legal mask, and teacher action must remain unchanged because the teacher consumes only the
    public information set, not the hidden hand state.
    """
    engine = TarneebEngine(RulesConfig())
    state = engine.new_deal(0, deal_index=0, match_seed=12345)
    engine.deal(state)
    seat = state.current_player_seat
    assert seat is not None

    teacher = AuctionTeacher(strategy="strong")
    view = PublicViewBuilder(engine).build(state, seat)
    baseline_action = teacher.choose_action(state, seat, view=view)

    for seed in range(8):
        variant = deepcopy(state)
        opponent_seats = [s for s in range(4) if s != seat]
        rng = np.random.default_rng(seed)
        rng.shuffle(opponent_seats)
        variant.hands[opponent_seats[0]], variant.hands[opponent_seats[1]] = (
            variant.hands[opponent_seats[1]],
            variant.hands[opponent_seats[0]],
        )

        variant_view = PublicViewBuilder(engine).build(variant, seat)
        assert np.array_equal(variant_view.legal_action_mask, view.legal_action_mask)
        assert variant_view.own_hand == view.own_hand
        assert variant_view.events == view.events
        assert variant_view.current_high_bid == view.current_high_bid
        assert teacher.choose_action(variant, seat, view=variant_view) == baseline_action


def test_open_and_competitive_auction_legalities_match_the_frozen_rules() -> None:
    engine = TarneebEngine(RulesConfig())
    state = engine.new_deal(0, deal_index=0, match_seed=12345)
    engine.deal(state)

    open_legal = set(np.flatnonzero(engine.legal_actions(state)).tolist())
    expected_open = {action_for_bid(bid) for bid in range(7, 14)} | {PASS_ACTION}
    assert open_legal == expected_open
    assert 64 not in open_legal

    engine.step(state, action_for_bid(9))
    competitive_legal = set(np.flatnonzero(engine.legal_actions(state)).tolist())
    expected_competitive = {action_for_bid(bid) for bid in range(10, 14)} | {PASS_ACTION}
    assert competitive_legal == expected_competitive
    assert action_for_bid(9) not in competitive_legal


def test_permanent_pass_keeps_a_seat_out_of_the_auction_and_action_64_is_illegal() -> None:
    engine = TarneebEngine(RulesConfig())
    state = engine.new_deal(0, deal_index=0, match_seed=7)
    engine.deal(state)
    passing_seat = state.current_player_seat
    assert passing_seat is not None

    engine.step(state, PASS_ACTION)
    assert passing_seat not in state.active_bidders

    all_legal = set(np.flatnonzero(engine.legal_actions(state)).tolist())
    assert PASS_ACTION in all_legal
    assert 64 not in all_legal


def test_phase11b_probe_is_repeatable_for_the_same_seed() -> None:
    first = collect_phase11b_probe(seed_base=800_000, deal_count=5)
    second = collect_phase11b_probe(seed_base=800_000, deal_count=5)
    assert first.deal_count == second.deal_count == 5
    assert len(first.states) == len(second.states)
    assert [state.state.deal_index for state in first.states] == [
        state.state.deal_index for state in second.states
    ]
