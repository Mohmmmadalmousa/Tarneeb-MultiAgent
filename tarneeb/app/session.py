"""A local match session: the thin layer between a human and the frozen engine.

This orchestrates. It does not decide anything about the game.

- **Legality** comes from ``TarneebEngine.legal_actions``. The session never
  computes which card may be played, which bid is allowed, or whether a seat must
  follow suit.
- **Transitions** come from ``TarneebEngine.step``. Trick winners, dealer
  rotation, auction advancement and scoring are all the engine's.
- **What a seat may see** comes from the frozen
  :class:`~tarneeb.env.public_view.PublicView`, which is the same actor-safe
  object every evaluated agent receives and is proved not to contain another
  seat's cards.
- **Replay** uses the frozen ``deal_v2`` recorder, so a saved CLI game verifies
  with the same code that verifies an evaluation game.

The human's view is built from a ``PublicView`` for the human's own seat and
nothing else, which is why the hidden-information tests can be stated simply: if a
card is not in the human's hand and has not been played, the view cannot mention
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tarneeb.agents.base import Agent, checked_action
from tarneeb.core.actions import (
    BID_ACTION_START,
    BID_ACTION_STOP,
    NO_TRUMP_ACTION,
    PASS_ACTION,
    SUIT_NAMES,
    TRUMP_ACTION_START,
    TRUMP_ACTION_STOP,
    bid_for_action,
)
from tarneeb.core.cards import NUM_SEATS, card_name
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.recorder import (
    DealRecord,
    GameRecorder,
    ReplayEngine,
    SeatPolicy,
    write_records_jsonl,
)
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TarneebState, team_of_seat
from tarneeb.env.public_view import PublicViewBuilder

LIVE_PHASES = (Phase.BID, Phase.DECLARE, Phase.PLAY)
MAX_DECISIONS_PER_DEAL = 200
"""A deal needs at most ~10 auction + 1 trump + 52 cards. A safety stop."""


@dataclass(frozen=True, slots=True)
class LegalHumanAction:
    """One action the engine says the human may take, with a label to show."""

    action_id: int
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class PublicTableView:
    """Everything the human is allowed to see, ready to render.

    Built only from the human's own :class:`PublicView`. There is deliberately no
    field that could hold another seat's cards.
    """

    human_seat: int
    phase: str
    dealer_seat: int
    seat_on_turn: int | None
    own_hand: tuple[str, ...]
    current_high_bid: int | None
    high_bidder_seat: int | None
    seats_that_passed: tuple[int, ...]
    declarer_seat: int | None
    contract: int | None
    trump: str | None
    leader_seat: int | None
    trick_number: int
    current_trick: tuple[tuple[int, str], ...]
    tricks_won_by_team: tuple[int, int]
    match_score_by_team: tuple[int, int]
    deal_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "human_seat": self.human_seat,
            "phase": self.phase,
            "dealer_seat": self.dealer_seat,
            "seat_on_turn": self.seat_on_turn,
            "own_hand": list(self.own_hand),
            "current_high_bid": self.current_high_bid,
            "high_bidder_seat": self.high_bidder_seat,
            "seats_that_passed": list(self.seats_that_passed),
            "declarer_seat": self.declarer_seat,
            "contract": self.contract,
            "trump": self.trump,
            "leader_seat": self.leader_seat,
            "trick_number": self.trick_number,
            "current_trick": [[seat, card] for seat, card in self.current_trick],
            "tricks_won_by_team": list(self.tricks_won_by_team),
            "match_score_by_team": list(self.match_score_by_team),
            "deal_index": self.deal_index,
        }


@dataclass
class DealSummary:
    """What one finished deal produced."""

    deal_index: int
    dealer_seat: int
    declarer_seat: int | None
    winning_bid: int | None
    trump: str | None
    tricks_won_by_team: tuple[int, int]
    score_change_by_team: tuple[int, int]
    contract_made: bool | None
    passed_out: bool
    match_score_by_team: tuple[int, int]


class MatchSession:
    """One seeded local match: a human seat plus three agent seats."""

    def __init__(
        self,
        agents_by_seat: dict[int, Agent],
        human_seat: int | None = 0,
        match_seed: int = 0,
        dealer_seat: int = 0,
        rules: RulesConfig | None = None,
        record_replay: bool = True,
    ) -> None:
        if human_seat is not None and not 0 <= human_seat < NUM_SEATS:
            raise ValueError(f"human_seat must be 0..{NUM_SEATS - 1}, got {human_seat}")
        for seat in range(NUM_SEATS):
            if seat == human_seat:
                continue
            if seat not in agents_by_seat:
                raise ValueError(f"seat {seat} has no agent and is not the human seat")

        self.rules = rules if rules is not None else RulesConfig()
        self.engine = TarneebEngine(self.rules)
        self.views = PublicViewBuilder(self.engine)
        self.replay_engine = ReplayEngine(self.rules)

        self.agents_by_seat = dict(agents_by_seat)
        self.human_seat = human_seat
        self.match_seed = match_seed
        self.dealer_seat = dealer_seat
        self.record_replay = record_replay

        self.deal_index = 0
        self.match_score_by_team: tuple[int, int] = (0, 0)
        self.state: TarneebState | None = None
        self.recorder: GameRecorder | None = None
        self.completed_deals: list[DealSummary] = []
        self.deal_records: list[DealRecord] = []

    # --- lifecycle ----------------------------------------------------------

    def start_deal(self) -> None:
        """Deal a fresh hand from the seeded engine."""
        state = self.engine.new_deal(
            self.dealer_seat, deal_index=self.deal_index, match_seed=self.match_seed
        )
        self.engine.deal(state)
        # The engine's per-deal state deliberately keeps its own match score at
        # zero. A deal_v2 record is self-contained and replays from a fresh deal,
        # so injecting a running total here would make every replay fail to
        # verify. The running match total is the session's, and the rendered view
        # takes it from there. This also matches training, where each episode is
        # one deal starting from a zero match score.
        self.state = state
        self.recorder = GameRecorder.start(state) if self.record_replay else None
        for agent in self.agents_by_seat.values():
            agent.reset()

    def require_state(self) -> TarneebState:
        if self.state is None:
            raise ValueError("no deal is in progress; call start_deal() first")
        return self.state

    @property
    def seat_on_turn(self) -> int | None:
        state = self.require_state()
        return state.current_player_seat

    @property
    def deal_is_live(self) -> bool:
        return self.state is not None and self.state.phase in LIVE_PHASES

    @property
    def waiting_for_human(self) -> bool:
        return self.deal_is_live and self.seat_on_turn == self.human_seat

    # --- what the human may see --------------------------------------------

    def public_table_view(self) -> PublicTableView:
        """Return the human's view of the table.

        Raises if there is no human seat, because there is then nobody whose view
        this would legitimately be.
        """
        if self.human_seat is None:
            raise ValueError("this session has no human seat")
        return self.table_view_for(self.human_seat)

    def table_view_for(self, seat: int) -> PublicTableView:
        """Return one seat's actor-safe view, built from the frozen PublicView."""
        state = self.require_state()
        view = self.views.build(state, seat)
        return PublicTableView(
            human_seat=seat,
            phase=view.phase.name,
            dealer_seat=view.dealer_seat,
            seat_on_turn=state.current_player_seat,
            own_hand=tuple(card_name(card) for card in view.own_hand),
            current_high_bid=view.current_high_bid,
            high_bidder_seat=view.current_high_bidder,
            seats_that_passed=tuple(sorted(view.seats_that_passed)),
            declarer_seat=view.declarer_seat,
            contract=view.current_high_bid if view.declarer_seat is not None else None,
            trump=None if view.trump_suit is None else SUIT_NAMES[view.trump_suit],
            leader_seat=view.leader_seat,
            trick_number=view.trick_number,
            current_trick=tuple(
                (seat_index, card_name(card)) for seat_index, card in view.current_trick
            ),
            tricks_won_by_team=view.tricks_won_by_team,
            match_score_by_team=self.match_score_by_team,
            deal_index=state.deal_index,
        )

    def legal_human_actions(self) -> list[LegalHumanAction]:
        """Return exactly the actions the engine allows the human right now.

        The labels are presentation; the action ids and the set itself come
        straight from ``TarneebEngine.legal_actions``.
        """
        if self.human_seat is None:
            raise ValueError("this session has no human seat")
        state = self.require_state()
        if state.current_player_seat != self.human_seat or state.phase not in LIVE_PHASES:
            return []
        mask = self.engine.legal_actions(state)
        return [describe_action(int(action_id)) for action_id in mask.nonzero()[0]]

    # --- advancing ----------------------------------------------------------

    def apply_human_action(self, action_id: int) -> None:
        """Apply the human's action after the engine has confirmed it is legal."""
        if self.human_seat is None:
            raise ValueError("this session has no human seat")
        state = self.require_state()
        if state.current_player_seat != self.human_seat:
            raise ValueError("it is not the human's turn")
        mask = self.engine.legal_actions(state)
        checked_action(action_id, mask, agent_name="human")
        self._step(action_id)

    def advance_agent_turns(self) -> int:
        """Let every agent-controlled seat act until the human is on turn.

        Returns how many agent actions were taken. Stops at the end of the deal.
        """
        actions_taken = 0
        while self.deal_is_live and self.seat_on_turn != self.human_seat:
            if actions_taken > MAX_DECISIONS_PER_DEAL:  # pragma: no cover - safety stop
                raise AssertionError("the deal exceeded its maximum decision count")
            state = self.require_state()
            acting_seat = state.current_player_seat
            assert acting_seat is not None
            agent = self.agents_by_seat[acting_seat]
            view = self.views.build(state, acting_seat)
            selected_action = agent.act(view)
            checked_action(selected_action, view.legal_action_mask, agent.name)
            self._step(selected_action)
            actions_taken += 1
        return actions_taken

    def _step(self, action_id: int) -> None:
        state = self.require_state()
        _state, emitted_events, _deal_result = self.engine.step(state, action_id)
        if self.recorder is not None:
            self.recorder.record_step(action_id, emitted_events)
        if state.phase is Phase.DEAL:
            # All four passed: the engine throws the hand in and keeps the dealer.
            self._finish_passed_out_deal()

    # --- finishing ----------------------------------------------------------

    def _finish_passed_out_deal(self) -> None:
        state = self.require_state()
        self.completed_deals.append(
            DealSummary(
                deal_index=self.deal_index,
                dealer_seat=self.dealer_seat,
                declarer_seat=None,
                winning_bid=None,
                trump=None,
                tricks_won_by_team=(0, 0),
                score_change_by_team=(0, 0),
                contract_made=None,
                passed_out=True,
                match_score_by_team=self.match_score_by_team,
            )
        )
        self.deal_index = state.deal_index
        self.recorder = None
        self.engine.deal(state)
        if self.record_replay:
            self.recorder = GameRecorder.start(state)

    def finish_deal(self) -> DealSummary:
        """Close a scored deal, update the match score and record the replay."""
        state = self.require_state()
        if state.deal_result is None:
            raise ValueError("the deal has not been scored yet")

        result = state.deal_result
        declaring_team = team_of_seat(state.declarer_seat)  # type: ignore[arg-type]
        score_change = [0, 0]
        score_change[declaring_team] = result.score_delta_bidding_team
        score_change[1 - declaring_team] = result.score_delta_defending_team
        self.match_score_by_team = (
            self.match_score_by_team[0] + score_change[0],
            self.match_score_by_team[1] + score_change[1],
        )

        if self.recorder is not None:
            self.deal_records.append(self.recorder.finish(state, policies=self.seat_policies()))
            self.recorder = None

        summary = DealSummary(
            deal_index=self.deal_index,
            dealer_seat=self.dealer_seat,
            declarer_seat=state.declarer_seat,
            winning_bid=state.current_high_bid,
            trump=None
            if state.selected_trump_suit is None
            else SUIT_NAMES[state.selected_trump_suit],
            tricks_won_by_team=(state.tricks_won[0], state.tricks_won[1]),
            score_change_by_team=(score_change[0], score_change[1]),
            contract_made=result.tricks_bidding_team >= result.bid,
            passed_out=False,
            match_score_by_team=self.match_score_by_team,
        )
        self.completed_deals.append(summary)
        self.deal_index += 1
        self.dealer_seat = (self.dealer_seat + 1) % NUM_SEATS
        self.state = None
        return summary

    def seat_policies(self) -> tuple[SeatPolicy, SeatPolicy, SeatPolicy, SeatPolicy]:
        """Return who controlled each seat, for the replay record.

        The frozen recorder never fabricates this; the session knows it, so it
        supplies it.
        """
        policies = tuple(
            SeatPolicy(policy_id="human")
            if seat == self.human_seat
            else SeatPolicy(policy_id=self.agents_by_seat[seat].name)
            for seat in range(NUM_SEATS)
        )
        return policies  # type: ignore[return-value]

    @property
    def match_is_complete(self) -> bool:
        return max(self.match_score_by_team) >= self.rules.target_score

    # --- replay -------------------------------------------------------------

    def write_replay(self, path: Path) -> Path:
        """Write every recorded deal using the frozen ``deal_v2`` writer."""
        if not self.deal_records:
            raise ValueError("no completed deal has been recorded")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_records_jsonl(self.deal_records, path)
        return path

    def verify_replays(self) -> bool:
        """Re-run every recorded deal through the frozen replay engine."""
        if not self.deal_records:
            raise ValueError("no completed deal has been recorded")
        return all(self.replay_engine.verify(record) for record in self.deal_records)


def describe_action(action_id: int) -> LegalHumanAction:
    """Return a human-readable label for one action id.

    Presentation only. Whether the action is allowed was already decided by the
    engine before this is ever called.
    """
    if action_id == NO_TRUMP_ACTION:  # pragma: no cover - never legal
        raise ValueError("action 64 is reserved and must never be offered")
    if action_id == PASS_ACTION:
        return LegalHumanAction(action_id, "pass", "pass")
    if BID_ACTION_START <= action_id < BID_ACTION_STOP:
        bid = bid_for_action(action_id)
        return LegalHumanAction(action_id, f"bid {bid}", f"bid {bid} tricks")
    if TRUMP_ACTION_START <= action_id < TRUMP_ACTION_STOP:
        suit = SUIT_NAMES[action_id - TRUMP_ACTION_START]
        return LegalHumanAction(action_id, suit, f"choose {suit} as trump")
    return LegalHumanAction(action_id, card_name(action_id), f"play {card_name(action_id)}")
