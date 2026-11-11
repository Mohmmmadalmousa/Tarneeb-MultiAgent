"""Deal recording and replay: the deal_v2 schema and invariant I12.

Replay must regenerate the recorded event stream field for field, and must
depend only on the deterministic game inputs -- the original hands, the action
sequence, and matching rule/schema versions. It must never need ``policies`` or
``seating``, which stay unset until the league phases exist.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tarneeb.core.actions import (
    ACTION_SPEC_VERSION,
    PASS_ACTION,
    action_for_bid,
    action_for_trump_suit,
)
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import EVENT_SCHEMA_VERSION, NUM_SEATS, EventType
from tarneeb.core.recorder import (
    DEAL_RECORD_VERSION,
    DealRecord,
    GameRecorder,
    ReplayEngine,
    SeatingRecord,
    SeatPolicy,
    read_records_jsonl,
    write_records_jsonl,
)
from tarneeb.core.rules import RULES_VERSION, RulesConfig
from tarneeb.core.state import TRICKS_PER_DEAL


@pytest.fixture
def rules() -> RulesConfig:
    return RulesConfig()


@pytest.fixture
def engine(rules: RulesConfig) -> TarneebEngine:
    return TarneebEngine(rules)


def record_one_deal(
    engine: TarneebEngine,
    match_seed: int = 42,
    dealer_seat: int = 0,
    deal_index: int = 0,
    bid_value: int = 9,
    trump_suit: int = 2,
    policy_seed: int = 5,
) -> DealRecord:
    """Play and record one complete deal with a seeded card-choice policy."""
    state = engine.deal(engine.new_deal(dealer_seat, deal_index=deal_index, match_seed=match_seed))
    recorder = GameRecorder.start(state)
    policy_rng = np.random.default_rng(policy_seed)

    for action_id in (
        action_for_bid(bid_value),
        PASS_ACTION,
        PASS_ACTION,
        PASS_ACTION,
        action_for_trump_suit(trump_suit),
    ):
        _, events, _ = engine.step(state, action_id)
        recorder.record_step(action_id, events)

    deal_result = None
    while deal_result is None:
        legal_action_ids = np.flatnonzero(engine.legal_actions(state))
        chosen_action = int(policy_rng.choice(legal_action_ids))
        _, events, deal_result = engine.step(state, chosen_action)
        recorder.record_step(chosen_action, events)

    return recorder.finish(state)


# --- schema -----------------------------------------------------------------


def test_record_version_is_deal_v2() -> None:
    assert DEAL_RECORD_VERSION == "deal_v2"


def test_record_carries_the_full_version_metadata(engine: TarneebEngine) -> None:
    record = record_one_deal(engine)
    assert record.record_version == DEAL_RECORD_VERSION
    assert record.rules_version == RULES_VERSION
    assert record.rules_fingerprint == engine.rules.fingerprint()
    assert record.action_spec_version == ACTION_SPEC_VERSION
    assert record.event_schema_version == EVENT_SCHEMA_VERSION


def test_record_has_every_deal_v2_field() -> None:
    field_names = {field.name for field in dataclasses.fields(DealRecord)}
    assert field_names == {
        "record_version",
        "rules_version",
        "rules_fingerprint",
        "action_spec_version",
        "event_schema_version",
        "obs_spec_version",
        "match_seed",
        "deal_index",
        "dealer_seat",
        "original_hands",
        "actions",
        "events",
        "deal_result",
        "final_match_score",
        "policies",
        "seating",
    }


def test_record_captures_the_deal_identity_and_content(engine: TarneebEngine) -> None:
    record = record_one_deal(engine, match_seed=2024, dealer_seat=1, deal_index=3)
    assert record.match_seed == 2024
    assert record.dealer_seat == 1
    assert record.deal_index == 3
    assert len(record.original_hands) == NUM_SEATS
    assert len(record.actions) == 5 + 52  # auction + declaration + 52 cards
    assert record.deal_result is not None
    assert sum(1 for event in record.events if event.event_type is EventType.CARD_PLAYED) == 52
    assert (
        sum(1 for event in record.events if event.event_type is EventType.TRICK_COMPLETED)
        == TRICKS_PER_DEAL
    )


def test_records_are_immutable(engine: TarneebEngine) -> None:
    record = record_one_deal(engine)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.match_seed = 0  # type: ignore[misc]


def test_recorder_requires_a_dealt_state(engine: TarneebEngine) -> None:
    state = engine.new_deal(0)
    with pytest.raises(ValueError, match="freshly dealt"):
        GameRecorder.start(state)


# --- later-phase fields stay unset and type-safe ----------------------------


def test_policies_and_seating_are_unset_in_phase_three(engine: TarneebEngine) -> None:
    record = record_one_deal(engine)
    assert record.policies is None  # league self-play does not exist yet
    assert record.seating is None
    assert record.obs_spec_version is None  # observations do not exist yet


def test_nothing_fabricates_a_policy_id(engine: TarneebEngine) -> None:
    serialized = record_one_deal(engine).to_dict()
    assert serialized["policies"] is None
    assert serialized["seating"] is None
    assert serialized["obs_spec_version"] is None


def test_the_schema_accepts_populated_league_metadata(engine: TarneebEngine) -> None:
    """A later phase must be able to fill these fields without a schema change."""
    base_record = record_one_deal(engine)
    populated = dataclasses.replace(
        base_record,
        policies=(
            SeatPolicy("learner", "v3"),
            SeatPolicy("snapshot_17"),
            SeatPolicy("learner", "v3"),
            SeatPolicy("snapshot_17"),
        ),
        seating=SeatingRecord(
            learning_team=(0, 2), mode="snapshot", grad_eligible=(True, False, True, False)
        ),
        obs_spec_version="obs_v2",
    )

    round_tripped = DealRecord.from_json(populated.to_json())
    assert round_tripped == populated
    assert round_tripped.policies is not None
    assert round_tripped.policies[0] == SeatPolicy("learner", "v3")
    assert round_tripped.policies[1].checkpoint_version is None
    assert round_tripped.seating == SeatingRecord((0, 2), "snapshot", (True, False, True, False))
    assert round_tripped.obs_spec_version == "obs_v2"


def test_seating_record_rejects_malformed_data() -> None:
    with pytest.raises(ValueError, match="malformed seating record"):
        SeatingRecord.from_dict({"learning_team": [0], "mode": "x", "grad_eligible": [True]})


# --- serialization ----------------------------------------------------------


def test_json_round_trip_is_exact(engine: TarneebEngine) -> None:
    record = record_one_deal(engine)
    assert DealRecord.from_json(record.to_json()) == record


def test_dict_round_trip_is_exact(engine: TarneebEngine) -> None:
    record = record_one_deal(engine)
    assert DealRecord.from_dict(record.to_dict()) == record


def test_round_trip_preserves_events_field_by_field(engine: TarneebEngine) -> None:
    record = record_one_deal(engine)
    restored = DealRecord.from_json(record.to_json())
    assert restored.events == record.events
    for original_event, restored_event in zip(record.events, restored.events, strict=True):
        assert restored_event.event_type is original_event.event_type
        assert restored_event.phase is original_event.phase
        assert restored_event.deal_result == original_event.deal_result


def test_jsonl_round_trip(engine: TarneebEngine, tmp_path) -> None:
    records = [record_one_deal(engine, match_seed=seed) for seed in (1, 2, 3)]
    path = tmp_path / "deals.jsonl"
    write_records_jsonl(records, path)
    assert read_records_jsonl(path) == records


# --- replay (I12) -----------------------------------------------------------


def test_replay_reproduces_the_event_stream(engine: TarneebEngine, rules: RulesConfig) -> None:
    record = record_one_deal(engine)
    replayed_state = ReplayEngine(rules).replay(record)
    assert tuple(replayed_state.events) == record.events


def test_verify_is_true_for_a_recorded_deal(engine: TarneebEngine, rules: RulesConfig) -> None:
    assert ReplayEngine(rules).verify(record_one_deal(engine))


@pytest.mark.parametrize("match_seed", range(20))
def test_verify_over_many_deals(engine: TarneebEngine, rules: RulesConfig, match_seed: int) -> None:
    record = record_one_deal(engine, match_seed=match_seed, policy_seed=match_seed + 100)
    assert ReplayEngine(rules).verify(record)


def test_replay_restores_hands_tricks_and_score(engine: TarneebEngine, rules: RulesConfig) -> None:
    record = record_one_deal(engine)
    replayed_state = ReplayEngine(rules).replay(record)

    assert list(replayed_state.original_hands) == list(record.original_hands)
    assert all(hand == 0 for hand in replayed_state.hands)
    assert replayed_state.deal_result == record.deal_result
    assert tuple(replayed_state.match_score) == record.final_match_score
    assert len(replayed_state.completed_tricks) == TRICKS_PER_DEAL


def test_replay_does_not_need_policies_or_seating(
    engine: TarneebEngine, rules: RulesConfig
) -> None:
    """The same deal replays identically with and without league metadata."""
    bare_record = record_one_deal(engine)
    decorated_record = dataclasses.replace(
        bare_record,
        policies=(SeatPolicy("a"), SeatPolicy("b"), SeatPolicy("c"), SeatPolicy("d")),
        seating=SeatingRecord((1, 3), "mirror", (False, True, False, True)),
    )

    replay = ReplayEngine(rules)
    assert replay.verify(bare_record)
    assert replay.verify(decorated_record)
    assert tuple(replay.replay(bare_record).events) == tuple(replay.replay(decorated_record).events)


def test_replay_after_a_json_round_trip(engine: TarneebEngine, rules: RulesConfig) -> None:
    record = DealRecord.from_json(record_one_deal(engine).to_json())
    assert ReplayEngine(rules).verify(record)


# --- version guards ---------------------------------------------------------


def test_replay_rejects_a_different_rules_fingerprint(
    engine: TarneebEngine, rules: RulesConfig
) -> None:
    record = dataclasses.replace(record_one_deal(engine), rules_fingerprint="sha1:deadbeef")
    with pytest.raises(ValueError, match="fingerprint differs"):
        ReplayEngine(rules).replay(record)


def test_replay_rejects_a_different_rules_version(
    engine: TarneebEngine, rules: RulesConfig
) -> None:
    record = dataclasses.replace(record_one_deal(engine), rules_version="tarneeb-classic-v1-alt13")
    with pytest.raises(ValueError, match="rules version"):
        ReplayEngine(rules).replay(record)


def test_replay_rejects_a_different_record_version(
    engine: TarneebEngine, rules: RulesConfig
) -> None:
    record = dataclasses.replace(record_one_deal(engine), record_version="deal_v1")
    with pytest.raises(ValueError, match="record version"):
        ReplayEngine(rules).replay(record)


def test_replay_rejects_different_action_or_event_specs(
    engine: TarneebEngine, rules: RulesConfig
) -> None:
    base_record = record_one_deal(engine)
    with pytest.raises(ValueError, match="action spec"):
        ReplayEngine(rules).replay(dataclasses.replace(base_record, action_spec_version="act_v2"))
    with pytest.raises(ValueError, match="event schema"):
        ReplayEngine(rules).replay(
            dataclasses.replace(base_record, event_schema_version="event_v2")
        )


def test_verify_fails_when_the_recorded_events_are_tampered_with(
    engine: TarneebEngine, rules: RulesConfig
) -> None:
    """A silent change to the event stream must not pass verification."""
    record = record_one_deal(engine)
    tampered_events = list(record.events)
    tampered_events[10] = dataclasses.replace(tampered_events[10], card=0)
    tampered_record = dataclasses.replace(record, events=tuple(tampered_events))
    assert not ReplayEngine(rules).verify(tampered_record)


def test_verify_fails_when_the_recorded_score_is_tampered_with(
    engine: TarneebEngine, rules: RulesConfig
) -> None:
    record = dataclasses.replace(record_one_deal(engine), final_match_score=(999, -999))
    assert not ReplayEngine(rules).verify(record)
