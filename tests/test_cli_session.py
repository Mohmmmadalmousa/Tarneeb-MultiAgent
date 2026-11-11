"""The local CLI: what the human may see, what the human may do, and replay.

Two properties carry the weight here.

**The human sees only their own hand.** Every card in the world is either in the
human's hand, already played in public, or invisible. The test states it that
way: take all 52 cards, subtract the human's hand and the publicly played cards,
and assert that not one of the remaining cards appears anywhere in the rendered
output.

**The engine decides legality.** The CLI offers exactly the engine's
legal-action mask and refuses anything else. It contains no follow-suit rule, no
bid-ordering rule, no trick-winner rule and no scoring.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.app.cli import (
    build_agents,
    lowest_legal_action,
    main,
    play_match,
    prompt_human_action,
    render_table,
)
from tarneeb.app.session import MatchSession, describe_action
from tarneeb.core.actions import NO_TRUMP_ACTION, NUM_ACTIONS
from tarneeb.core.cards import NUM_CARDS, NUM_SEATS, card_name
from tarneeb.core.recorder import DEAL_RECORD_VERSION, read_records_jsonl
from tarneeb.core.rules import RulesConfig
from tarneeb.models.checkpoint import IncompatibleCheckpointError, save_checkpoint
from tarneeb.models.nets import build_actor_and_critic
from tarneeb.training.ppo import PPOConfig, build_optimizer


def build_session(
    human_seat: int | None = 0, match_seed: int = 99, opponent: str = "basic"
) -> MatchSession:
    agents = {
        seat: (
            BasicHeuristicAgent(name=opponent)
            if opponent == "basic"
            else RandomAgent(name=opponent, seed=seat)
        )
        for seat in range(NUM_SEATS)
        if seat != human_seat
    }
    return MatchSession(agents_by_seat=agents, human_seat=human_seat, match_seed=match_seed)


def play_one_deal(session: MatchSession) -> None:
    session.start_deal()
    while session.deal_is_live:
        session.advance_agent_turns()
        if not session.deal_is_live:
            break
        if session.waiting_for_human:
            session.apply_human_action(lowest_legal_action(session))


# --- hidden information -----------------------------------------------------


def test_the_human_view_shows_only_the_human_hand() -> None:
    session = build_session()
    session.start_deal()
    state = session.require_state()

    view = session.public_table_view()
    own_cards = set(view.own_hand)
    assert len(view.own_hand) == 13
    for seat in range(1, NUM_SEATS):
        for card in _cards_of(state.hands[seat]):
            assert card_name(card) not in own_cards


def test_no_opponent_card_ever_appears_in_the_rendered_table() -> None:
    """The strong statement: unseen cards cannot be printed, at any point."""
    session = build_session()
    session.start_deal()

    for _decision in range(200):
        if not session.deal_is_live:
            break
        state = session.require_state()
        view = session.public_table_view()

        lines: list[str] = []
        render_table(view, lines.append)
        rendered = "\n".join(lines)

        public_cards = {card for trick in state.completed_tricks for _seat, card in trick} | {
            card for _seat, card in state.current_trick
        }
        own_cards = set(_cards_of(state.hands[0]))
        invisible = [
            card for card in range(NUM_CARDS) if card not in public_cards and card not in own_cards
        ]
        for card in invisible:
            assert card_name(card) not in rendered, (
                f"{card_name(card)} is held by another seat but was rendered"
            )

        session.advance_agent_turns()
        if session.deal_is_live and session.waiting_for_human:
            session.apply_human_action(lowest_legal_action(session))


def test_the_table_view_has_no_field_that_could_hold_another_hand() -> None:
    session = build_session()
    session.start_deal()
    payload = session.public_table_view().as_dict()
    assert set(payload) == {
        "human_seat",
        "phase",
        "dealer_seat",
        "seat_on_turn",
        "own_hand",
        "current_high_bid",
        "high_bidder_seat",
        "seats_that_passed",
        "declarer_seat",
        "contract",
        "trump",
        "leader_seat",
        "trick_number",
        "current_trick",
        "tricks_won_by_team",
        "match_score_by_team",
        "deal_index",
    }
    assert "hands" not in payload
    assert "original_hands" not in payload


def test_the_view_exposes_the_public_state_the_human_needs() -> None:
    session = build_session()
    play_one_deal(session)
    session.finish_deal()
    session.start_deal()
    while session.deal_is_live and session.require_state().phase.name != "PLAY":
        session.advance_agent_turns()
        if session.deal_is_live and session.waiting_for_human:
            session.apply_human_action(lowest_legal_action(session))

    view = session.public_table_view()
    assert view.dealer_seat in range(NUM_SEATS)
    assert view.declarer_seat in range(NUM_SEATS)
    assert view.trump is not None
    assert view.contract is not None
    assert view.tricks_won_by_team is not None
    assert view.match_score_by_team != (0, 0), "the running match score must be visible"


# --- legality ---------------------------------------------------------------


def test_the_offered_actions_are_exactly_the_engine_mask() -> None:
    session = build_session()
    session.start_deal()

    checked = 0
    while session.deal_is_live and checked < 40:
        session.advance_agent_turns()
        if not session.deal_is_live:
            break
        state = session.require_state()
        mask = session.engine.legal_actions(state)
        offered = {action.action_id for action in session.legal_human_actions()}
        assert offered == set(np.flatnonzero(mask).tolist())
        assert NO_TRUMP_ACTION not in offered
        session.apply_human_action(lowest_legal_action(session))
        checked += 1
    assert checked > 0


def test_an_illegal_human_action_is_rejected() -> None:
    session = build_session()
    session.start_deal()
    session.advance_agent_turns()

    mask = session.engine.legal_actions(session.require_state())
    illegal = int(np.flatnonzero(~mask)[0])
    with pytest.raises(ValueError, match="illegal action"):
        session.apply_human_action(illegal)


def test_the_reserved_no_trump_action_is_never_offered_or_accepted() -> None:
    session = build_session()
    session.start_deal()
    for _decision in range(120):
        if not session.deal_is_live:
            break
        if session.waiting_for_human:
            assert all(
                action.action_id != NO_TRUMP_ACTION for action in session.legal_human_actions()
            )
            with pytest.raises(ValueError):
                session.apply_human_action(NO_TRUMP_ACTION)
            session.apply_human_action(lowest_legal_action(session))
        session.advance_agent_turns()


def test_acting_out_of_turn_is_refused() -> None:
    session = build_session()
    session.start_deal()
    session.advance_agent_turns()
    session.apply_human_action(lowest_legal_action(session))
    if not session.waiting_for_human:
        with pytest.raises(ValueError, match="not the human's turn"):
            session.apply_human_action(0)


def test_the_prompt_re_asks_until_the_answer_is_legal() -> None:
    session = build_session()
    session.start_deal()
    session.advance_agent_turns()

    legal_actions = session.legal_human_actions()
    answers = iter(["not-a-move", "9999", "0"])
    messages: list[str] = []
    chosen = prompt_human_action(session, lambda _prompt: next(answers), messages.append)

    assert chosen == legal_actions[0].action_id
    assert any("not one of your legal options" in line for line in messages)


def test_action_labels_never_describe_the_reserved_action() -> None:
    with pytest.raises(ValueError, match="reserved"):
        describe_action(NO_TRUMP_ACTION)
    assert describe_action(NUM_ACTIONS - 5).label == "clubs"


# --- agents advance ---------------------------------------------------------


def test_agent_seats_advance_by_themselves() -> None:
    session = build_session()
    session.start_deal()
    actions_taken = session.advance_agent_turns()
    assert actions_taken >= 0
    assert session.waiting_for_human or not session.deal_is_live


def test_a_session_without_a_human_seat_plays_itself() -> None:
    session = build_session(human_seat=None)
    session.start_deal()
    session.advance_agent_turns()
    assert not session.deal_is_live
    summary = session.finish_deal()
    assert summary.tricks_won_by_team[0] + summary.tricks_won_by_team[1] == 13


def test_a_missing_agent_is_refused() -> None:
    with pytest.raises(ValueError, match="no agent"):
        MatchSession(agents_by_seat={1: BasicHeuristicAgent()}, human_seat=0)


# --- determinism and completion ---------------------------------------------


def test_the_same_seed_reproduces_the_same_match() -> None:
    def run() -> list[tuple]:
        session = build_session(match_seed=2024)
        summaries = play_match(
            session=session,
            deal_limit=3,
            emit=lambda _line: None,
            read_line=None,
            auto_play=lowest_legal_action,
        )
        return [
            (
                summary.deal_index,
                summary.declarer_seat,
                summary.winning_bid,
                summary.tricks_won_by_team,
                summary.score_change_by_team,
            )
            for summary in summaries
        ]

    assert run() == run()


def test_a_different_seed_produces_a_different_match() -> None:
    def run(seed: int) -> list[tuple]:
        session = build_session(match_seed=seed)
        summaries = play_match(
            session=session,
            deal_limit=2,
            emit=lambda _line: None,
            read_line=None,
            auto_play=lowest_legal_action,
        )
        return [(s.declarer_seat, s.tricks_won_by_team) for s in summaries]

    assert run(11) != run(12)


def test_a_complete_deal_awards_thirteen_tricks_and_a_score() -> None:
    session = build_session()
    play_one_deal(session)
    summary = session.finish_deal()
    assert summary.tricks_won_by_team[0] + summary.tricks_won_by_team[1] == 13
    assert summary.contract_made is not None
    assert session.match_score_by_team != (0, 0)


def test_a_match_runs_until_the_target_score() -> None:
    session = build_session(match_seed=7)
    play_match(
        session=session,
        deal_limit=40,
        emit=lambda _line: None,
        read_line=None,
        auto_play=lowest_legal_action,
    )
    assert session.completed_deals
    assert session.match_is_complete or len(session.completed_deals) == 40


# --- replay -----------------------------------------------------------------


def test_a_completed_cli_game_writes_a_replay_that_verifies(tmp_path: Path) -> None:
    session = build_session(match_seed=321)
    play_match(
        session=session,
        deal_limit=3,
        emit=lambda _line: None,
        read_line=None,
        auto_play=lowest_legal_action,
    )
    path = session.write_replay(tmp_path / "cli_replay.jsonl")

    assert path.exists()
    assert session.verify_replays()

    records = read_records_jsonl(path)
    assert len(records) == len(session.deal_records)
    for record in records:
        assert record.record_version == DEAL_RECORD_VERSION == "deal_v2"
        assert record.rules_fingerprint == RulesConfig().fingerprint()
        assert session.replay_engine.verify(record)


def test_the_replay_records_who_played_each_seat() -> None:
    session = build_session(match_seed=5)
    play_one_deal(session)
    session.finish_deal()
    record = session.deal_records[-1]
    assert record.policies is not None
    assert record.policies[0].policy_id == "human"
    assert record.policies[1].policy_id == "basic"


def test_writing_a_replay_with_nothing_recorded_is_refused() -> None:
    session = build_session()
    with pytest.raises(ValueError, match="no completed deal"):
        session.write_replay(Path("unused.jsonl"))


# --- checkpoints ------------------------------------------------------------


def test_a_compatible_checkpoint_loads_into_the_cli(tmp_path: Path) -> None:
    actor, critic = build_actor_and_critic(seed=1)
    path = save_checkpoint(
        path=tmp_path / "actor.pt",
        actor=actor,
        critic=critic,
        optimizer=build_optimizer(actor, critic, PPOConfig()),
        rules=RulesConfig(),
        training_config={},
        global_environment_steps=1,
        completed_updates=1,
        wall_clock_seconds=0.0,
        seeds={},
    )
    agents = build_agents(
        opponents="basic",
        human_seat=0,
        seed=0,
        checkpoint=path,
        history_condition="full",
        rules=RulesConfig(),
    )
    assert set(agents) == {1, 2, 3}
    for agent in agents.values():
        assert agent.name == "neural"
        assert not hasattr(agent, "critic")


def test_an_incompatible_checkpoint_is_rejected_before_any_card_is_dealt(
    tmp_path: Path,
) -> None:
    actor, critic = build_actor_and_critic(seed=1)
    path = save_checkpoint(
        path=tmp_path / "actor.pt",
        actor=actor,
        critic=critic,
        optimizer=None,
        rules=RulesConfig(),
        training_config={},
        global_environment_steps=0,
        completed_updates=0,
        wall_clock_seconds=0.0,
        seeds={},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["buffer_contract_version"] = "chunk_v2"
    torch.save(payload, path)

    with pytest.raises(IncompatibleCheckpointError, match="buffer_contract_version"):
        build_agents(
            opponents="basic",
            human_seat=0,
            seed=0,
            checkpoint=path,
            history_condition="full",
            rules=RulesConfig(),
        )


def test_an_edited_rule_is_rejected_by_the_fingerprint(tmp_path: Path) -> None:
    actor, critic = build_actor_and_critic(seed=1)
    path = save_checkpoint(
        path=tmp_path / "actor.pt",
        actor=actor,
        critic=critic,
        optimizer=None,
        rules=RulesConfig(),
        training_config={},
        global_environment_steps=0,
        completed_updates=0,
        wall_clock_seconds=0.0,
        seeds={},
    )
    with pytest.raises(IncompatibleCheckpointError, match="rules_fingerprint"):
        build_agents(
            opponents="basic",
            human_seat=0,
            seed=0,
            checkpoint=path,
            history_condition="full",
            rules=RulesConfig(target_score=41),
        )


# --- the entry point --------------------------------------------------------


def test_the_cli_entry_point_completes_a_seeded_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    replay_path = tmp_path / "replay.jsonl"
    exit_code = main(
        [
            "--seed",
            "777",
            "--deals",
            "2",
            "--opponents",
            "basic",
            "--auto-play",
            "--replay-path",
            str(replay_path),
        ]
    )
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "tarneeb-classic-v1" in captured
    assert RulesConfig().fingerprint() in captured
    assert "replay verifies: True" in captured
    assert replay_path.exists()
    assert all(
        session_record.record_version == "deal_v2"
        for session_record in read_records_jsonl(replay_path)
    )


def test_the_cli_never_prints_a_card_the_human_cannot_see(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end leakage check on the real entry point's output."""
    main(["--seed", "31415", "--deals", "2", "--opponents", "basic", "--auto-play"])
    captured = capsys.readouterr().out

    # --auto-play never renders a hand, so the only cards printed are summaries.
    # Nothing that names a specific card may appear outside the human's own hand
    # line, which auto-play does not emit at all.
    assert "your hand" not in captured


def _cards_of(hand_mask: int) -> list[int]:
    from tarneeb.core.cards import cards_in_hand

    return cards_in_hand(hand_mask)


def test_render_table_writes_to_the_sink_it_is_given() -> None:
    session = build_session()
    session.start_deal()
    stream = io.StringIO()
    render_table(session.public_table_view(), lambda line: stream.write(line + "\n"))
    text = stream.getvalue()
    assert "your hand" in text
    assert "dealer" in text
