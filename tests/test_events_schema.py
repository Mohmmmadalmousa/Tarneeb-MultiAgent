"""Schema tests for event_v1.

These check the shape of the frozen event representation only. No engine
exists yet, so nothing here emits or validates an event stream.
"""

from __future__ import annotations

import dataclasses

import pytest

from tarneeb.core.events import (
    EVENT_SCHEMA_VERSION,
    NUM_SEATS,
    DealResult,
    EventType,
    Phase,
    PublicEvent,
    RelPublicEvent,
)

# Written from the blueprint's event_v1 definition, in declaration order.
PUBLIC_EVENT_FIELDS = (
    "seq",
    "phase",
    "actor_seat",
    "event_type",
    "bid_value",
    "trump",
    "card",
    "trick_no",
    "trick_winner_seat",
    "deal_result",
)

REL_PUBLIC_EVENT_FIELDS = (
    "seq",
    "phase",
    "event_type",
    "actor_rel",
    "bid_value",
    "trump",
    "card",
    "trick_no",
    "winner_rel",
)

# event_v1 numeric values. Reordering these invalidates stored buffers.
EXPECTED_EVENT_TYPES = {
    "PAD": 0,
    "BID": 1,
    "PASS": 2,
    "TRUMP_SELECTED": 3,
    "CARD_PLAYED": 4,
    "TRICK_COMPLETED": 5,
    "DEAL_COMPLETED": 6,
}

EXPECTED_PHASES = {
    "DEAL": 0,
    "BID": 1,
    "DECLARE": 2,
    "PLAY": 3,
    "SCORE": 4,
    "DONE": 5,
}


def _pad_event(**overrides: object) -> PublicEvent:
    base: dict[str, object] = {
        "seq": 0,
        "phase": Phase.BID,
        "actor_seat": None,
        "event_type": EventType.PAD,
        "bid_value": None,
        "trump": None,
        "card": None,
        "trick_no": 0,
        "trick_winner_seat": None,
        "deal_result": None,
    }
    base.update(overrides)
    return PublicEvent(**base)  # type: ignore[arg-type]


def test_schema_version_constant() -> None:
    assert EVENT_SCHEMA_VERSION == "event_v1"


def test_num_seats() -> None:
    assert NUM_SEATS == 4


# --- enums ------------------------------------------------------------------


def test_event_type_members_and_values() -> None:
    assert {member.name: member.value for member in EventType} == EXPECTED_EVENT_TYPES


def test_event_type_pad_is_zero() -> None:
    """Right-padding relies on PAD being the zero id."""
    assert EventType.PAD == 0


def test_phase_members_and_values() -> None:
    assert {member.name: member.value for member in Phase} == EXPECTED_PHASES


# --- field sets -------------------------------------------------------------


def test_public_event_fields_are_exactly_the_schema() -> None:
    names = tuple(f.name for f in dataclasses.fields(PublicEvent))
    assert names == PUBLIC_EVENT_FIELDS


def test_rel_public_event_fields_are_exactly_the_schema() -> None:
    names = tuple(f.name for f in dataclasses.fields(RelPublicEvent))
    assert names == REL_PUBLIC_EVENT_FIELDS


def test_rel_public_event_drops_absolute_seats() -> None:
    names = {f.name for f in dataclasses.fields(RelPublicEvent)}
    assert "actor_seat" not in names
    assert "trick_winner_seat" not in names


def test_deal_result_fields() -> None:
    names = tuple(f.name for f in dataclasses.fields(DealResult))
    assert names == (
        "bid",
        "tricks_bidding_team",
        "score_delta_bidding_team",
        "score_delta_defending_team",
    )


# --- EV-1: no field can hold private information ----------------------------

PRIVATE_FIELD_MARKERS = (
    "hand",
    "hands",
    "mask",
    "bitmask",
    "hidden",
    "deck",
    "private",
    "held",
    "remaining",
    "original",
    "state",
)


@pytest.mark.parametrize("event_class", [PublicEvent, RelPublicEvent])
def test_no_field_name_suggests_private_information(event_class: type) -> None:
    for field in dataclasses.fields(event_class):
        lowered = field.name.lower()
        for marker in PRIVATE_FIELD_MARKERS:
            assert marker not in lowered, f"{event_class.__name__}.{field.name}"


@pytest.mark.parametrize("event_class", [PublicEvent, RelPublicEvent, DealResult])
def test_no_field_is_a_collection(event_class: type) -> None:
    """A hand could only be smuggled in as a collection; every field is scalar."""
    allowed = {"int", "int | None", "Phase", "EventType", "DealResult | None"}
    for field in dataclasses.fields(event_class):
        assert field.type in allowed, f"{event_class.__name__}.{field.name}: {field.type}"


def test_card_is_a_single_optional_id() -> None:
    (card_field,) = [f for f in dataclasses.fields(PublicEvent) if f.name == "card"]
    assert card_field.type == "int | None"


# --- immutability -----------------------------------------------------------


@pytest.mark.parametrize(
    ("instance", "field"),
    [
        (_pad_event(), "seq"),
        (
            RelPublicEvent(
                seq=0,
                phase=Phase.BID,
                event_type=EventType.PAD,
                actor_rel=None,
                bid_value=None,
                trump=None,
                card=None,
                trick_no=0,
                winner_rel=None,
            ),
            "seq",
        ),
        (
            DealResult(
                bid=7,
                tricks_bidding_team=7,
                score_delta_bidding_team=7,
                score_delta_defending_team=0,
            ),
            "bid",
        ),
    ],
)
def test_events_are_immutable(instance: object, field: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, 99)


def test_events_compare_by_value() -> None:
    assert _pad_event(seq=3) == _pad_event(seq=3)
    assert _pad_event(seq=3) != _pad_event(seq=4)


# --- the schema can express the required streams ----------------------------


def test_schema_supports_contiguous_zero_based_sequence_numbers() -> None:
    stream = [_pad_event(seq=i) for i in range(5)]
    assert [ev.seq for ev in stream] == list(range(5))


def test_schema_supports_the_end_of_deal_ordering() -> None:
    """CARD_PLAYED -> TRICK_COMPLETED -> DEAL_COMPLETED must be representable."""
    result = DealResult(
        bid=9,
        tricks_bidding_team=9,
        score_delta_bidding_team=9,
        score_delta_defending_team=0,
    )
    stream = [
        _pad_event(
            seq=40,
            phase=Phase.PLAY,
            actor_seat=2,
            event_type=EventType.CARD_PLAYED,
            card=51,
            trick_no=12,
        ),
        _pad_event(
            seq=41,
            phase=Phase.PLAY,
            event_type=EventType.TRICK_COMPLETED,
            trick_no=12,
            trick_winner_seat=2,
        ),
        _pad_event(
            seq=42,
            phase=Phase.SCORE,
            event_type=EventType.DEAL_COMPLETED,
            trick_no=13,
            deal_result=result,
        ),
    ]
    assert [ev.event_type for ev in stream] == [
        EventType.CARD_PLAYED,
        EventType.TRICK_COMPLETED,
        EventType.DEAL_COMPLETED,
    ]
    assert [ev.seq for ev in stream] == [40, 41, 42]
    assert stream[0].actor_seat == 2
    assert stream[1].actor_seat is None
    assert stream[2].deal_result is result
