"""The behaviour-cloning dataset: contract, regeneration, splits and leakage.

Two claims carry the weight here.

**Regeneration is exact.** The dataset stores seeds and teacher actions, not
observations, and rebuilds everything by replaying the frozen engine. If that
replay ever diverged from the original play-through, every label would be
attached to the wrong state. So it is checked directly rather than assumed.

**The demonstrations leak nothing.** The teacher reads only a ``PublicView``,
which is actor-safe by construction. The tests below confirm the property
survives into the generated data: no stored observation moves when the other
three hands are rearranged, and no event id names an unplayed card.
"""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.buffers.bc import (
    BC_DATASET_VERSION,
    TEACHER_IDENTITY,
    BCDataset,
    collate_sequences,
)
from tarneeb.core.actions import NO_TRUMP_ACTION, NUM_ACTIONS
from tarneeb.core.cards import NUM_SEATS, cards_in_hand
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import EventType, Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.env.public_view import PublicViewBuilder
from tarneeb.env.spaces import D_STATIC, K_MAX, L_MAX, N_EVENT_FIELDS
from tarneeb.evaluation.arena import HELD_OUT_EVALUATION_SEEDS, TRAINING_SEEDS
from tarneeb.training.bc import generate_bc_dataset, iterate_batches, regenerate_sequences
from tarneeb.training.phase9_seeds import (
    PHASE9_SPLITS,
    describe_splits,
    ranges_overlap,
    splits_are_disjoint,
    splits_are_inside_training_range,
    splits_avoid_held_out_range,
    training_environment_seed,
)

RULES = RulesConfig()


@pytest.fixture(scope="module")
def dataset() -> BCDataset:
    return generate_bc_dataset(range(1_000_000, 1_000_060), "test_split")


# --- seed splits ------------------------------------------------------------


def test_every_phase9_split_is_disjoint_from_every_other() -> None:
    assert splits_are_disjoint()


def test_no_phase9_split_can_reach_the_held_out_range() -> None:
    """Structural, not a matter of discipline: the ranges cannot overlap."""
    assert splits_are_inside_training_range()
    assert splits_avoid_held_out_range()
    for span in PHASE9_SPLITS.values():
        assert not ranges_overlap(span, HELD_OUT_EVALUATION_SEEDS)
        assert span.stop <= TRAINING_SEEDS.stop


def test_bc_and_ppo_splits_do_not_share_a_seed() -> None:
    """Otherwise "it plays well" could restate "it memorised its demonstrations"."""
    assert not ranges_overlap(PHASE9_SPLITS["bc_train"], PHASE9_SPLITS["ppo_train"])
    assert not ranges_overlap(PHASE9_SPLITS["bc_train"], PHASE9_SPLITS["ppo_development"])
    assert not ranges_overlap(PHASE9_SPLITS["bc_test"], PHASE9_SPLITS["ppo_train"])


def test_independent_training_seeds_are_far_apart() -> None:
    seeds = [training_environment_seed(index) for index in range(3)]
    assert len(set(seeds)) == 3
    assert min(abs(a - b) for a in seeds for b in seeds if a != b) >= 100_000
    for seed in seeds:
        assert seed in PHASE9_SPLITS["ppo_train"]


def test_the_split_description_is_json_safe() -> None:
    described = describe_splits()
    assert set(described) == set(PHASE9_SPLITS)
    for entry in described.values():
        assert entry["stop"] - entry["start"] == entry["size"]


# --- dataset contract -------------------------------------------------------


def test_the_dataset_records_every_frozen_contract(dataset: BCDataset) -> None:
    metadata = dataset.metadata
    assert metadata.dataset_version == BC_DATASET_VERSION
    assert metadata.teacher == TEACHER_IDENTITY
    assert metadata.rules_version == "tarneeb-classic-v1"
    assert metadata.rules_fingerprint == RULES.fingerprint()
    assert metadata.obs_spec_version == "obs_v2"
    assert metadata.action_spec_version == "act_v1"
    assert metadata.event_schema_version == "event_v1"
    assert metadata.deal_count == len(dataset)


def test_the_dataset_round_trips_through_a_file(dataset: BCDataset, tmp_path) -> None:
    path = dataset.save(tmp_path / "bc.npz")
    loaded = BCDataset.load(path, rules=RULES)

    assert loaded.metadata.as_dict() == dataset.metadata.as_dict()
    assert len(loaded) == len(dataset)
    for original, restored in zip(dataset.deals, loaded.deals, strict=True):
        assert restored.match_seed == original.match_seed
        assert restored.dealer_seat == original.dealer_seat
        assert restored.teacher_actions == original.teacher_actions
        assert restored.passed_out == original.passed_out


def test_a_dataset_from_different_contracts_is_refused(dataset: BCDataset, tmp_path) -> None:
    import json

    path = dataset.save(tmp_path / "bc.npz")
    with np.load(path, allow_pickle=False) as stored:
        payload = {key: stored[key] for key in stored.files}
    corrupted = json.loads(str(payload["metadata"]))
    corrupted["rules_fingerprint"] = "sha1:0000"
    payload["metadata"] = np.array(json.dumps(corrupted))
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="rules_fingerprint"):
        BCDataset.load(path, rules=RULES)


def test_the_stored_file_is_small(dataset: BCDataset, tmp_path) -> None:
    """Storing seeds rather than observations is what keeps this trivial."""
    path = dataset.save(tmp_path / "bc.npz")
    assert path.stat().st_size < 64 * 1024


# --- regeneration is exact --------------------------------------------------


def test_regeneration_reproduces_the_teachers_own_decisions(dataset: BCDataset) -> None:
    """The strongest correctness claim: replay the record, ask the teacher again.

    If regeneration drifted, the recovered states would differ and the teacher
    would choose differently.
    """
    engine = TarneebEngine(RULES)
    views = PublicViewBuilder(engine)
    teacher = StrongHeuristicAgent()

    for record in dataset.deals[:12]:
        state = engine.new_deal(
            record.dealer_seat, deal_index=record.deal_index, match_seed=record.match_seed
        )
        engine.deal(state)
        teacher.reset()
        for stored_action in record.teacher_actions:
            acting_seat = state.current_player_seat
            assert acting_seat is not None
            assert teacher.act(views.build(state, acting_seat)) == stored_action
            engine.step(state, stored_action)
            if state.phase is Phase.DEAL:
                break


def test_regeneration_is_deterministic(dataset: BCDataset) -> None:
    first = regenerate_sequences(dataset.deals[0], RULES)
    second = regenerate_sequences(dataset.deals[0], RULES)
    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.statics, right.statics)
        assert np.array_equal(left.segment_ids, right.segment_ids)
        assert np.array_equal(left.teacher_actions, right.teacher_actions)


def test_each_deal_yields_one_sequence_per_seat(dataset: BCDataset) -> None:
    for record in dataset.deals[:10]:
        sequences = regenerate_sequences(record, RULES)
        assert {sequence.seat for sequence in sequences} == set(range(NUM_SEATS))


def test_sequence_shapes_match_the_frozen_contracts(dataset: BCDataset) -> None:
    sequence = regenerate_sequences(dataset.deals[0], RULES)[0]
    count = sequence.decision_count
    assert sequence.statics.shape == (count, D_STATIC)
    assert sequence.segment_ids.shape == (count, K_MAX, N_EVENT_FIELDS)
    assert sequence.history_ids.shape == (count, L_MAX, N_EVENT_FIELDS)
    assert sequence.legal_action_masks.shape == (count, NUM_ACTIONS)
    assert sequence.teacher_actions.shape == (count,)


def test_every_teacher_action_is_legal_under_its_own_stored_mask(dataset: BCDataset) -> None:
    for record in dataset.deals[:15]:
        for sequence in regenerate_sequences(record, RULES):
            for index in range(sequence.decision_count):
                action = int(sequence.teacher_actions[index])
                assert bool(sequence.legal_action_masks[index, action])
                assert not bool(sequence.legal_action_masks[index, NO_TRUMP_ACTION])


def test_segments_partition_the_event_stream_without_gap_or_overlap(
    dataset: BCDataset,
) -> None:
    """Each seat's segments must tile the events before its last decision.

    A gap would mean a lost event; an overlap would mean one counted twice.
    Either would put replay in a state the policy never held.
    """
    for record in dataset.deals[:10]:
        for sequence in regenerate_sequences(record, RULES):
            covered = int(sequence.segment_lengths.sum())
            # The padded history at the final decision says how many events
            # preceded it; the segments must sum to exactly that.
            events_before_last = int(sequence.history_masks[-1].sum())
            assert covered == events_before_last


def test_no_segment_exceeds_the_proved_bound(dataset: BCDataset) -> None:
    observed = 0
    for record in dataset.deals[:20]:
        for sequence in regenerate_sequences(record, RULES):
            observed = max(observed, int(sequence.segment_lengths.max()))
    assert 0 < observed <= 14


def test_the_first_segment_is_not_empty_for_a_later_seat(dataset: BCDataset) -> None:
    """BC sequences start at h=0, so decision 0 must consume its prior events.

    This is the one place BC deliberately differs from ``chunk_v2``, whose chunk
    heads store an empty segment because they carry a stored ``h0``.
    """
    saw_nonempty_first = False
    for record in dataset.deals[:10]:
        for sequence in regenerate_sequences(record, RULES):
            if int(sequence.segment_lengths[0]) > 0:
                saw_nonempty_first = True
    assert saw_nonempty_first


# --- batching ---------------------------------------------------------------


def test_batches_cover_every_decision_exactly_once(dataset: BCDataset) -> None:
    batches = iterate_batches(dataset, batch_sequences=8, architecture="gru")
    total = sum(batch.valid_decision_count for batch in batches)
    expected = sum(
        sequence.decision_count
        for record in dataset.deals
        for sequence in regenerate_sequences(record, RULES)
    )
    assert total == expected


def test_padded_positions_carry_a_usable_sentinel_mask(dataset: BCDataset) -> None:
    batches = iterate_batches(dataset, batch_sequences=16, architecture="gru")
    padded_found = False
    for batch in batches:
        padded = ~batch.valid
        if not padded.any():
            continue
        padded_found = True
        assert bool(batch.legal_action_masks[padded].sum(axis=-1).min() >= 1)
        assert int(batch.segment_lengths[padded].sum()) == 0
    assert padded_found


def test_batch_building_materializes_only_what_the_architecture_needs(
    dataset: BCDataset,
) -> None:
    """Materializing both modalities would double memory for nothing."""
    gru = iterate_batches(dataset, batch_sequences=16, architecture="gru")[0]
    ff = iterate_batches(dataset, batch_sequences=16, architecture="ff")[0]
    assert gru.segment_ids.shape[2] == K_MAX
    assert gru.history_ids.shape[2] == 0
    assert ff.history_ids.shape[2] == L_MAX
    assert ff.segment_ids.shape[2] == 0


def test_collate_refuses_an_empty_list() -> None:
    with pytest.raises(ValueError, match="empty list"):
        collate_sequences([])


def test_both_architectures_see_the_same_decisions(dataset: BCDataset) -> None:
    """A1 compares architectures on identical demonstrations, not similar ones."""
    gru = iterate_batches(dataset, batch_sequences=16, architecture="gru")
    ff = iterate_batches(dataset, batch_sequences=16, architecture="ff")
    assert len(gru) == len(ff)
    for left, right in zip(gru, ff, strict=True):
        assert np.array_equal(left.teacher_actions, right.teacher_actions)
        assert np.array_equal(left.valid, right.valid)
        assert np.array_equal(left.statics, right.statics)
        assert np.array_equal(left.legal_action_masks, right.legal_action_masks)


# --- information safety -----------------------------------------------------


def test_no_stored_observation_moves_when_the_other_hands_are_rearranged() -> None:
    """Invariant I7, applied to the demonstration data itself."""
    engine = TarneebEngine(RULES)
    from tarneeb.env.observation import ObservationBuilder

    builder = ObservationBuilder(engine)
    state = engine.new_deal(0, deal_index=0, match_seed=4321)
    engine.deal(state)
    teacher = StrongHeuristicAgent()
    views = PublicViewBuilder(engine)
    for _ in range(20):
        if state.phase not in (Phase.BID, Phase.DECLARE, Phase.PLAY):
            break
        seat = state.current_player_seat
        assert seat is not None
        engine.step(state, teacher.act(views.build(state, seat)))

    observer = state.current_player_seat
    assert observer is not None
    before = builder.static_actor(state, observer)

    others = [seat for seat in range(NUM_SEATS) if seat != observer]
    pooled = [card for seat in others for card in cards_in_hand(state.hands[seat])]
    np.random.default_rng(0).shuffle(pooled)
    cursor = 0
    for seat in others:
        size = len(cards_in_hand(state.hands[seat]))
        hand = 0
        for card in pooled[cursor : cursor + size]:
            hand |= 1 << card
        state.hands[seat] = hand
        cursor += size

    assert np.array_equal(before, builder.static_actor(state, observer))


def test_no_stored_event_names_an_unplayed_card(dataset: BCDataset) -> None:
    """The event ids the student consumes must carry no hidden card."""
    engine = TarneebEngine(RULES)
    for record in dataset.deals[:8]:
        state = engine.new_deal(
            record.dealer_seat, deal_index=record.deal_index, match_seed=record.match_seed
        )
        engine.deal(state)
        for action in record.teacher_actions:
            if state.phase not in (Phase.BID, Phase.DECLARE, Phase.PLAY):
                break
            engine.step(state, action)
            if state.phase is Phase.DEAL:
                break

        played: set[int] = set()
        for event in state.events:
            if event.event_type is EventType.CARD_PLAYED:
                assert event.card is not None
                played.add(event.card)
        for seat in range(NUM_SEATS):
            for card in cards_in_hand(state.hands[seat]):
                assert card not in played, "an unplayed hidden card reached the event stream"


def test_the_teacher_reads_only_a_public_view() -> None:
    """Structural: the frozen teacher's only input is the actor-safe view."""
    import inspect

    signature = inspect.signature(StrongHeuristicAgent.act)
    assert list(signature.parameters) == ["self", "view"]
    assert signature.parameters["view"].annotation in ("PublicView", "PublicView")


def test_stored_observations_have_the_actor_width_not_the_critic_width(
    dataset: BCDataset,
) -> None:
    """692, never 856. The centralized critic vector is Phase 10."""
    for sequence in regenerate_sequences(dataset.deals[0], RULES):
        assert sequence.statics.shape[1] == D_STATIC == 692
