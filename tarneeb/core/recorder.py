"""Deal recording and deterministic replay -- the ``deal_v2`` schema.

A :class:`DealRecord` holds everything needed to reproduce one deal exactly and
to prove that the reproduction is identical: the rule identity it was played
under, the seed and dealer, the four original hands, the complete action
sequence, and the complete public event stream. Actions and events are both
stored on purpose -- replay regenerates the events from the actions and compares
them field by field, so a silent change in event emission cannot pass unnoticed.

**Deliberately unpopulated fields.** ``policies`` and ``seating`` belong to the
league and self-play phases, which do not exist yet. They are part of the frozen
``deal_v2`` schema and are typed ``None`` until that functionality arrives, so
the league can fill in the existing fields rather than changing the schema. This
module never invents a policy id or a seating assignment, and replay never needs
either of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tarneeb.core.actions import ACTION_SPEC_VERSION
from tarneeb.core.cards import CARDS_PER_HAND, hand_size
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import (
    EVENT_SCHEMA_VERSION,
    NUM_SEATS,
    DealResult,
    EventType,
    Phase,
    PublicEvent,
)
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TarneebState

DEAL_RECORD_VERSION = "deal_v2"


@dataclass(frozen=True, slots=True)
class SeatPolicy:
    """Which policy controlled one seat.

    Populated from Phase 10 (league self-play) onward. Nothing in the engine or
    in replay reads it.
    """

    policy_id: str
    checkpoint_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"policy_id": self.policy_id, "checkpoint_version": self.checkpoint_version}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SeatPolicy:
        return cls(
            policy_id=data["policy_id"],
            checkpoint_version=data.get("checkpoint_version"),
        )


@dataclass(frozen=True, slots=True)
class SeatingRecord:
    """Which team was learning and which seats owned gradients.

    Populated from Phase 10 onward, when league seating and gradient ownership
    exist. Nothing in the engine or in replay reads it.
    """

    learning_team: tuple[int, int]
    mode: str
    grad_eligible: tuple[bool, bool, bool, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_team": list(self.learning_team),
            "mode": self.mode,
            "grad_eligible": list(self.grad_eligible),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SeatingRecord:
        learning_team = tuple(data["learning_team"])
        grad_eligible = tuple(data["grad_eligible"])
        if len(learning_team) != 2 or len(grad_eligible) != NUM_SEATS:
            raise ValueError(f"malformed seating record: {data!r}")
        return cls(
            learning_team=(int(learning_team[0]), int(learning_team[1])),
            mode=str(data["mode"]),
            grad_eligible=(
                bool(grad_eligible[0]),
                bool(grad_eligible[1]),
                bool(grad_eligible[2]),
                bool(grad_eligible[3]),
            ),
        )


def _event_to_dict(event: PublicEvent) -> dict[str, Any]:
    return {
        "seq": event.seq,
        "phase": int(event.phase),
        "actor_seat": event.actor_seat,
        "event_type": int(event.event_type),
        "bid_value": event.bid_value,
        "trump": event.trump,
        "card": event.card,
        "trick_no": event.trick_no,
        "trick_winner_seat": event.trick_winner_seat,
        "deal_result": _deal_result_to_dict(event.deal_result),
    }


def _event_from_dict(data: dict[str, Any]) -> PublicEvent:
    return PublicEvent(
        seq=data["seq"],
        phase=Phase(data["phase"]),
        actor_seat=data["actor_seat"],
        event_type=EventType(data["event_type"]),
        bid_value=data["bid_value"],
        trump=data["trump"],
        card=data["card"],
        trick_no=data["trick_no"],
        trick_winner_seat=data["trick_winner_seat"],
        deal_result=_deal_result_from_dict(data["deal_result"]),
    )


def _deal_result_to_dict(deal_result: DealResult | None) -> dict[str, Any] | None:
    if deal_result is None:
        return None
    return {
        "bid": deal_result.bid,
        "tricks_bidding_team": deal_result.tricks_bidding_team,
        "score_delta_bidding_team": deal_result.score_delta_bidding_team,
        "score_delta_defending_team": deal_result.score_delta_defending_team,
    }


def _deal_result_from_dict(data: dict[str, Any] | None) -> DealResult | None:
    if data is None:
        return None
    return DealResult(
        bid=data["bid"],
        tricks_bidding_team=data["tricks_bidding_team"],
        score_delta_bidding_team=data["score_delta_bidding_team"],
        score_delta_defending_team=data["score_delta_defending_team"],
    )


@dataclass(frozen=True, slots=True)
class DealRecord:
    """One deal, complete enough to replay and to verify (schema ``deal_v2``)."""

    # --- schema and rule identity -------------------------------------
    record_version: str
    rules_version: str
    rules_fingerprint: str
    action_spec_version: str
    event_schema_version: str
    obs_spec_version: str | None
    """Unset until observations exist (Phase 4); replay does not use it."""

    # --- deal identity -------------------------------------------------
    match_seed: int
    deal_index: int
    dealer_seat: int

    # --- the deal itself ------------------------------------------------
    original_hands: tuple[int, int, int, int]
    """The four dealt hands as card bitmasks, indexed by seat."""

    actions: tuple[int, ...]
    """Every action id applied, in order, across BID, DECLARE and PLAY."""

    events: tuple[PublicEvent, ...]
    """The public event stream, stored redundantly so replay can compare it."""

    deal_result: DealResult | None
    """The scored outcome, or ``None`` for a deal that was not played out."""

    final_match_score: tuple[int, int]

    # --- later-phase metadata, absent until those phases exist -----------
    policies: tuple[SeatPolicy, SeatPolicy, SeatPolicy, SeatPolicy] | None = None
    """Per-seat controlling policy. ``None`` until league self-play (Phase 10)."""

    seating: SeatingRecord | None = None
    """Learning team and gradient ownership. ``None`` until Phase 10."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary, preserving unset optionals."""
        return {
            "record_version": self.record_version,
            "rules_version": self.rules_version,
            "rules_fingerprint": self.rules_fingerprint,
            "action_spec_version": self.action_spec_version,
            "event_schema_version": self.event_schema_version,
            "obs_spec_version": self.obs_spec_version,
            "match_seed": self.match_seed,
            "deal_index": self.deal_index,
            "dealer_seat": self.dealer_seat,
            "original_hands": list(self.original_hands),
            "actions": list(self.actions),
            "events": [_event_to_dict(event) for event in self.events],
            "deal_result": _deal_result_to_dict(self.deal_result),
            "final_match_score": list(self.final_match_score),
            "policies": (
                None
                if self.policies is None
                else [seat_policy.to_dict() for seat_policy in self.policies]
            ),
            "seating": None if self.seating is None else self.seating.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DealRecord:
        """Rebuild a record from :meth:`to_dict` output, preserving unset optionals."""
        original_hands = tuple(int(hand) for hand in data["original_hands"])
        if len(original_hands) != NUM_SEATS:
            raise ValueError(f"expected {NUM_SEATS} hands, got {len(original_hands)}")

        stored_policies = data.get("policies")
        policies = (
            None
            if stored_policies is None
            else tuple(SeatPolicy.from_dict(entry) for entry in stored_policies)
        )
        if policies is not None and len(policies) != NUM_SEATS:
            raise ValueError(f"expected {NUM_SEATS} seat policies, got {len(policies)}")

        stored_seating = data.get("seating")
        final_match_score = tuple(int(value) for value in data["final_match_score"])

        return cls(
            record_version=data["record_version"],
            rules_version=data["rules_version"],
            rules_fingerprint=data["rules_fingerprint"],
            action_spec_version=data["action_spec_version"],
            event_schema_version=data["event_schema_version"],
            obs_spec_version=data.get("obs_spec_version"),
            match_seed=int(data["match_seed"]),
            deal_index=int(data["deal_index"]),
            dealer_seat=int(data["dealer_seat"]),
            original_hands=(
                original_hands[0],
                original_hands[1],
                original_hands[2],
                original_hands[3],
            ),
            actions=tuple(int(action) for action in data["actions"]),
            events=tuple(_event_from_dict(entry) for entry in data["events"]),
            deal_result=_deal_result_from_dict(data.get("deal_result")),
            final_match_score=(final_match_score[0], final_match_score[1]),
            policies=policies,  # type: ignore[arg-type]
            seating=None if stored_seating is None else SeatingRecord.from_dict(stored_seating),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> DealRecord:
        return cls.from_dict(json.loads(text))


@dataclass
class GameRecorder:
    """Accumulates the actions and events of one deal into a :class:`DealRecord`.

    Start it once the cards are dealt, feed it each action as it is applied, and
    finish it when the deal is scored. It observes only; it never influences a
    transition.
    """

    rules: RulesConfig
    match_seed: int = 0
    deal_index: int = 0
    dealer_seat: int = 0
    original_hands: tuple[int, int, int, int] = (0, 0, 0, 0)
    actions: list[int] = field(default_factory=list)
    events: list[PublicEvent] = field(default_factory=list)

    @classmethod
    def start(cls, state: TarneebState) -> GameRecorder:
        """Begin recording a freshly dealt hand."""
        if any(hand_size(hand) != CARDS_PER_HAND for hand in state.hands):
            raise ValueError("GameRecorder.start requires a freshly dealt state")
        hands = tuple(state.original_hands)
        return cls(
            rules=state.rules,
            match_seed=state.match_seed or 0,
            deal_index=state.deal_index,
            dealer_seat=state.dealer_seat,
            original_hands=(hands[0], hands[1], hands[2], hands[3]),
        )

    def record_step(self, action_id: int, emitted_events: list[PublicEvent]) -> None:
        """Record one applied action and the events it produced."""
        self.actions.append(action_id)
        self.events.extend(emitted_events)

    def finish(
        self,
        state: TarneebState,
        policies: tuple[SeatPolicy, SeatPolicy, SeatPolicy, SeatPolicy] | None = None,
        seating: SeatingRecord | None = None,
    ) -> DealRecord:
        """Return the finished record.

        ``policies`` and ``seating`` stay ``None`` unless a caller that actually
        knows them supplies them; this module never fabricates either.
        """
        return DealRecord(
            record_version=DEAL_RECORD_VERSION,
            rules_version=self.rules.version,
            rules_fingerprint=self.rules.fingerprint(),
            action_spec_version=ACTION_SPEC_VERSION,
            event_schema_version=EVENT_SCHEMA_VERSION,
            obs_spec_version=None,
            match_seed=self.match_seed,
            deal_index=self.deal_index,
            dealer_seat=self.dealer_seat,
            original_hands=self.original_hands,
            actions=tuple(self.actions),
            events=tuple(self.events),
            deal_result=state.deal_result,
            final_match_score=(state.match_score[0], state.match_score[1]),
            policies=policies,
            seating=seating,
        )


class ReplayEngine:
    """Re-simulates a recorded deal and checks it reproduces exactly.

    Replay depends only on the deterministic game inputs -- the original hands
    and the action sequence -- plus compatible schema and rule versions. It never
    reads ``policies`` or ``seating``, so a Phase 3 record with both unset
    replays exactly like a later record that has them.
    """

    def __init__(self, rules: RulesConfig) -> None:
        self.rules = rules
        self.engine = TarneebEngine(rules)

    def check_compatible(self, record: DealRecord) -> None:
        """Raise if the record was produced under different rules or schemas."""
        if record.record_version != DEAL_RECORD_VERSION:
            raise ValueError(f"record version {record.record_version!r} != {DEAL_RECORD_VERSION!r}")
        if record.rules_version != self.rules.version:
            raise ValueError(f"rules version {record.rules_version!r} != {self.rules.version!r}")
        if record.rules_fingerprint != self.rules.fingerprint():
            raise ValueError("rules fingerprint differs; the rules have changed since recording")
        if record.action_spec_version != ACTION_SPEC_VERSION:
            raise ValueError(
                f"action spec {record.action_spec_version!r} != {ACTION_SPEC_VERSION!r}"
            )
        if record.event_schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"event schema {record.event_schema_version!r} != {EVENT_SCHEMA_VERSION!r}"
            )

    def replay(self, record: DealRecord) -> TarneebState:
        """Re-simulate the recorded deal and return the resulting state."""
        self.check_compatible(record)

        state = self.engine.new_deal(
            dealer_seat=record.dealer_seat,
            deal_index=record.deal_index,
            match_seed=record.match_seed,
        )
        self.engine.deal(state)

        # Replay from the recorded hands, not from the reshuffle, so a record
        # stays replayable even if it was produced by injected hands.
        state.hands = list(record.original_hands)
        state.original_hands = list(record.original_hands)

        for action_id in record.actions:
            self.engine.step(state, action_id)
        return state

    def verify(self, record: DealRecord) -> bool:
        """Return whether replay regenerates the record field for field."""
        state = self.replay(record)
        if tuple(state.events) != record.events:
            return False
        if state.deal_result != record.deal_result:
            return False
        return tuple(state.match_score) == record.final_match_score


def write_records_jsonl(records: list[DealRecord], path: Path) -> None:
    """Write one JSON object per line."""
    path.write_text(
        "".join(record.to_json() + "\n" for record in records),
        encoding="utf-8",
    )


def read_records_jsonl(path: Path) -> list[DealRecord]:
    """Read records written by :func:`write_records_jsonl`."""
    return [
        DealRecord.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
