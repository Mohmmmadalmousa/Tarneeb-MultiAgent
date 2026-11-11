"""The local command-line game: play Tarneeb against the heuristics or a checkpoint.

A thin renderer over :class:`~tarneeb.app.session.MatchSession`. It asks the
session what the human may see and what the human may do, and prints it. It
decides nothing about the game: there is no legality check, no scoring, no trick
resolution and no auction logic anywhere in this file.

Usage::

    python -m tarneeb.app.cli --seed 42 --opponents strong
    python -m tarneeb.app.cli --seed 42 --human-seat 2 --checkpoint checkpoints/stage6.pt
    python -m tarneeb.app.cli --seed 42 --deals 1 --replay-path results/mvp/replay.jsonl
    python -m tarneeb.app.cli --seed 42 --auto-play --deals 2   # no human, for smoke runs

A neural seat is loaded through the frozen checkpoint loader, so a checkpoint
built under different rules, observation, action, event, architecture or buffer
contracts is refused before a card is dealt.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from tarneeb.agents.base import Agent
from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.app.session import DealSummary, MatchSession, PublicTableView
from tarneeb.core.cards import NUM_SEATS
from tarneeb.core.rules import RulesConfig
from tarneeb.curriculum.agents import HistoryConditionedNeuralAgent
from tarneeb.curriculum.stages import FULL_HISTORY

HEURISTIC_AGENTS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "basic": BasicHeuristicAgent,
    "strong": StrongHeuristicAgent,
}

SEAT_LABELS = ("South", "West", "North", "East")
"""Seat 0 is the usual human seat; partners sit opposite (0 with 2, 1 with 3)."""


def seat_label(seat: int) -> str:
    return f"seat {seat} ({SEAT_LABELS[seat]})"


def build_opponent(name: str, seed: int) -> Agent:
    """Build one frozen heuristic opponent."""
    if name not in HEURISTIC_AGENTS:
        raise ValueError(f"unknown opponent {name!r}; expected one of {sorted(HEURISTIC_AGENTS)}")
    agent_class = HEURISTIC_AGENTS[name]
    if agent_class is RandomAgent:
        return RandomAgent(name=name, seed=seed)
    return agent_class(name=name)  # type: ignore[call-arg]


def build_agents(
    opponents: str,
    human_seat: int | None,
    seed: int,
    checkpoint: Path | None,
    history_condition: str,
    rules: RulesConfig,
) -> dict[int, Agent]:
    """Seat an agent at every seat the human is not sitting in.

    A checkpoint, when given, controls every agent seat. The frozen loader refuses
    an incompatible one, and the critic is never loaded.
    """
    neural_agent: Agent | None = None
    if checkpoint is not None:
        neural_agent = HistoryConditionedNeuralAgent.from_checkpoint(
            checkpoint,
            history_condition=history_condition,
            name="neural",
            deterministic=True,
            seed=seed,
            rules=rules,
        )
    return {
        seat: (
            neural_agent
            if neural_agent is not None
            else build_opponent(opponents, seed=seed + seat)
        )
        for seat in range(NUM_SEATS)
        if seat != human_seat
    }


# --- rendering --------------------------------------------------------------


def render_table(view: PublicTableView, emit: Callable[[str], None]) -> None:
    """Print exactly what the human may see, and nothing else."""
    emit("")
    emit(f"  deal {view.deal_index}   phase {view.phase}   dealer {seat_label(view.dealer_seat)}")
    emit(
        f"  match score   team 0: {view.match_score_by_team[0]}   team 1: {view.match_score_by_team[1]}"
    )

    if view.current_high_bid is None:
        emit("  auction       no bid yet")
    else:
        bidder = "nobody" if view.high_bidder_seat is None else seat_label(view.high_bidder_seat)
        emit(f"  auction       high bid {view.current_high_bid} by {bidder}")
    if view.seats_that_passed:
        emit(f"  passed        {', '.join(seat_label(seat) for seat in view.seats_that_passed)}")

    if view.declarer_seat is not None:
        emit(
            f"  contract      {view.contract} by {seat_label(view.declarer_seat)}"
            f"   trump {view.trump or 'not chosen'}"
        )
        emit(
            f"  tricks        team 0: {view.tricks_won_by_team[0]}"
            f"   team 1: {view.tricks_won_by_team[1]}   (trick {view.trick_number})"
        )
    if view.current_trick:
        played = "  ".join(f"{seat_label(seat)} {card}" for seat, card in view.current_trick)
        emit(f"  current trick {played}")

    emit(f"  your hand     {' '.join(view.own_hand) if view.own_hand else '(empty)'}")
    if view.seat_on_turn is not None:
        turn = "you" if view.seat_on_turn == view.human_seat else seat_label(view.seat_on_turn)
        emit(f"  on turn       {turn}")


def render_deal_summary(summary: DealSummary, emit: Callable[[str], None]) -> None:
    emit("")
    if summary.passed_out:
        emit(f"  deal {summary.deal_index}: passed out, nobody scored")
        return
    declarer = "nobody" if summary.declarer_seat is None else seat_label(summary.declarer_seat)
    made = "made" if summary.contract_made else "failed"
    emit(
        f"  deal {summary.deal_index}: {declarer} contracted {summary.winning_bid} "
        f"in {summary.trump} and {made} it"
    )
    emit(
        f"  tricks  team 0: {summary.tricks_won_by_team[0]}   "
        f"team 1: {summary.tricks_won_by_team[1]}"
    )
    emit(
        f"  score   team 0: {summary.score_change_by_team[0]:+d}   "
        f"team 1: {summary.score_change_by_team[1]:+d}   "
        f"match {summary.match_score_by_team[0]} - {summary.match_score_by_team[1]}"
    )


def prompt_human_action(
    session: MatchSession, read_line: Callable[[str], str], emit: Callable[[str], None]
) -> int:
    """Ask until the human names one of the actions the engine allows.

    The offered set is the engine's legal-action mask. Anything else is rejected
    and re-prompted; the CLI never decides that a move is illegal on its own.
    """
    legal_actions = session.legal_human_actions()
    if not legal_actions:  # pragma: no cover - only called when the human is on turn
        raise ValueError("the human has no legal action")

    emit("")
    emit("  your options:")
    for index, action in enumerate(legal_actions):
        emit(f"    [{index}] {action.description}")

    labels_by_choice = {action.label.lower(): action.action_id for action in legal_actions}
    while True:
        answer = read_line("  choose (number or name): ").strip().lower()
        if answer.isdigit() and int(answer) < len(legal_actions):
            return legal_actions[int(answer)].action_id
        if answer in labels_by_choice:
            return labels_by_choice[answer]
        emit("  that is not one of your legal options; try again")


# --- the loop ---------------------------------------------------------------


def play_match(
    session: MatchSession,
    deal_limit: int,
    emit: Callable[[str], None],
    read_line: Callable[[str], str] | None,
    auto_play: Callable[[MatchSession], int] | None = None,
) -> list[DealSummary]:
    """Play up to ``deal_limit`` deals, or until the match target is reached.

    ``auto_play`` replaces the human with a callable choosing from the engine's
    legal actions, which is how the end-to-end tests drive a complete match with
    no console.
    """
    for _deal in range(deal_limit):
        session.start_deal()
        while session.deal_is_live:
            session.advance_agent_turns()
            if not session.deal_is_live:
                break
            if session.waiting_for_human:
                if auto_play is not None:
                    session.apply_human_action(auto_play(session))
                else:
                    assert read_line is not None
                    render_table(session.public_table_view(), emit)
                    session.apply_human_action(prompt_human_action(session, read_line, emit))

        if session.state is not None and session.state.deal_result is not None:
            render_deal_summary(session.finish_deal(), emit)
        if session.match_is_complete:
            emit("")
            emit(f"  match complete: {session.match_score_by_team}")
            break
    return session.completed_deals


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="match seed; the same seed replays")
    parser.add_argument("--human-seat", type=int, default=0)
    parser.add_argument("--dealer-seat", type=int, default=0)
    parser.add_argument("--deals", type=int, default=4)
    parser.add_argument("--opponents", type=str, default="strong", choices=sorted(HEURISTIC_AGENTS))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--history-condition", type=str, default=FULL_HISTORY)
    parser.add_argument("--replay-path", type=Path, default=None)
    parser.add_argument(
        "--auto-play",
        action="store_true",
        help="no human: the human seat plays its lowest legal action, for smoke runs",
    )
    return parser.parse_args(argv)


def lowest_legal_action(session: MatchSession) -> int:
    """A deterministic stand-in for a human, used by ``--auto-play``."""
    return min(action.action_id for action in session.legal_human_actions())


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    rules = RulesConfig()

    agents_by_seat = build_agents(
        opponents=arguments.opponents,
        human_seat=arguments.human_seat,
        seed=arguments.seed,
        checkpoint=arguments.checkpoint,
        history_condition=arguments.history_condition,
        rules=rules,
    )
    session = MatchSession(
        agents_by_seat=agents_by_seat,
        human_seat=arguments.human_seat,
        match_seed=arguments.seed,
        dealer_seat=arguments.dealer_seat,
        rules=rules,
    )

    print(f"Tarneeb -- rules {rules.version} ({rules.fingerprint()})")
    print(f"seed {arguments.seed}   you are {seat_label(arguments.human_seat)}")
    print(f"your partner is {seat_label((arguments.human_seat + 2) % NUM_SEATS)}")
    if arguments.checkpoint is not None:
        print(f"AI seats play the checkpoint {arguments.checkpoint}")
    else:
        print(f"AI seats play the {arguments.opponents} heuristic")

    play_match(
        session=session,
        deal_limit=arguments.deals,
        emit=print,
        read_line=None if arguments.auto_play else input,
        auto_play=lowest_legal_action if arguments.auto_play else None,
    )

    if arguments.replay_path is not None and session.deal_records:
        path = session.write_replay(arguments.replay_path)
        print(f"\nreplay written to {path}")
        print(f"replay verifies: {session.verify_replays()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
