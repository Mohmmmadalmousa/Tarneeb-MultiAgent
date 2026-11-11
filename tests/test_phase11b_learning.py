"""Small synthetic optimizer checks. Never optimize a research checkpoint."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tarneeb.auction_repair.artifacts import save_repair_new, sha256
from tarneeb.auction_repair.data import (
    encode_public_view,
    frozen_features,
    generate_auction_examples,
    generate_play_examples,
    public_teacher_label,
    validate_label,
    validate_seeds,
)
from tarneeb.auction_repair.learning import (
    AuctionBCConfig,
    AuctionRows,
    assert_preserved,
    fit_auction_rows,
    preservation_report,
)
from tarneeb.auction_repair.measurement import measure_actor, training_seating_diagnostic
from tarneeb.core.actions import action_for_bid
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.env.public_view import PublicViewBuilder
from tarneeb.models.recurrent import (
    RecurrentActorNetwork,
    RecurrentArchitectureSpec,
    build_recurrent_actor_and_critic,
)
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint, save_recurrent_checkpoint


@pytest.fixture
def actor() -> RecurrentActorNetwork:
    spec = RecurrentArchitectureSpec(hidden_size=16, static_hidden_dim=24)
    return build_recurrent_actor_and_critic(spec, seed=701)[0]


def test_examples_are_legal_public_repeatable_and_cover_bid_levels() -> None:
    first = generate_auction_examples(range(8_200_000, 8_200_004))
    second = generate_auction_examples(range(8_200_000, 8_200_004))
    assert len(first) == 30 * 4
    assert first.metadata == second.metadata
    for name in ("statics", "histories", "lengths", "legal", "labels"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert {m["current_high_bid"] for m in first.metadata} == {None, 7, 8, 9, 10, 11, 12, 13}
    assert first.statics.shape[1] == 692
    assert not first.legal[:, 64].any()
    assert first.legal[np.arange(len(first)), first.labels].all()
    for team in (0, 1):
        subset = generate_auction_examples(range(8_200_000, 8_200_004), learner_team=team)
        assert all(m["seat"] % 2 == team for m in subset.metadata)


def test_bid13_has_no_raise_decision() -> None:
    engine = TarneebEngine(RulesConfig())
    state = engine.new_deal(0, match_seed=8_200_001)
    engine.deal(state)
    engine.step(state, action_for_bid(13))
    assert state.phase is Phase.BID
    assert set(np.flatnonzero(engine.legal_actions(state))) == {59}
    for _ in range(3):
        engine.step(state, 59)
    final_phase: Phase = state.phase
    assert final_phase is Phase.DECLARE


def test_constructed_coverage_has_exceptional_hands_and_suit_rotations() -> None:
    examples = generate_auction_examples(range(8_200_000, 8_200_001), include_stress=True)
    assert len(examples) == 30 + 960
    assert set(examples.labels.tolist()) == set(range(52, 60))
    assert {m["strength"] for m in examples.metadata} == {"weak", "medium", "strong", "exceptional"}
    assert {m["best_suit"] for m in examples.metadata} == {0, 1, 2, 3}


def test_hidden_hands_cannot_change_examples_or_teacher() -> None:
    engine = TarneebEngine(RulesConfig())
    state = engine.new_deal(0, match_seed=8_200_003)
    engine.deal(state)
    engine.step(state, action_for_bid(8))
    seat = state.current_player_seat
    assert seat is not None
    view = PublicViewBuilder(engine).build(state, seat)
    baseline = encode_public_view(view)
    others = [s for s in range(4) if s != seat]
    for shift in (1, 2):
        variant = state.clone()
        for i, other in enumerate(others):
            variant.hands[other] = state.hands[others[(i + shift) % 3]]
            variant.original_hands[other] = state.original_hands[others[(i + shift) % 3]]
        changed = PublicViewBuilder(engine).build(variant, seat)
        encoded = encode_public_view(changed)
        assert np.array_equal(baseline[0], encoded[0])
        assert np.array_equal(baseline[1], encoded[1])
        assert baseline[2] == encoded[2]
        assert public_teacher_label(view) == public_teacher_label(changed)


def test_teacher_has_both_weak_opening_passes_and_supported_bids() -> None:
    data = generate_auction_examples(range(8_200_000, 8_200_064))
    openings = [i for i, m in enumerate(data.metadata) if m["current_high_bid"] is None]
    assert any(data.labels[i] == 59 for i in openings)
    assert any(data.labels[i] != 59 for i in openings)
    # No lower clamp: every opening label whose rounded estimate is below 7 passes.
    for i in openings:
        if round(data.metadata[i]["estimate"] + 1.5) < 7:
            assert data.labels[i] == 59


@pytest.mark.parametrize("label", [-1, 0, 64, 65, 1000])
def test_invalid_teacher_labels_raise_without_fallback(label: int) -> None:
    legal = np.zeros(65, dtype=bool)
    legal[52:60] = True
    with pytest.raises(ValueError, match="illegal label"):
        validate_label(label, legal)


@pytest.mark.parametrize("seeds", [range(20250910, 20250911), range(10**9, 10**9 + 1), range(0)])
def test_reserved_or_empty_seeds_rejected(seeds: range) -> None:
    with pytest.raises(ValueError):
        validate_seeds(seeds)


def test_optimizer_learns_only_auction_rows_repeatably_and_preserves_play(
    actor: RecurrentActorNetwork,
) -> None:
    examples = generate_auction_examples(range(8_200_000, 8_200_004))
    fixture = generate_play_examples(range(8_210_000, 8_210_002))
    before = deepcopy(actor.state_dict())
    config = AuctionBCConfig(learning_rate=1e-3, epochs=3, batch_size=32)
    repaired, metrics = fit_auction_rows(actor, examples, config)
    repeated, second_metrics = fit_auction_rows(actor, examples, config)
    assert metrics == second_metrics
    assert metrics["epoch_nll"][-1] < metrics["initial_nll"]
    assert metrics["optimizer_steps"] == 12
    assert metrics["trainable_count"] == 8 * 17
    assert metrics["trainable_names"] == ["auction_weight", "auction_bias"]
    assert_preserved(actor, repaired)
    assert not torch.equal(actor.policy_head.weight[52:60], repaired.policy_head.weight[52:60])
    for name, value in actor.state_dict().items():
        assert torch.equal(value, before[name])  # caller is untouched
        assert torch.equal(repaired.state_dict()[name], repeated.state_dict()[name])
    report = preservation_report(actor, repaired, fixture)
    assert report["deterministic_action_agreement"] == 1
    assert report["max_legal_logit_drift"] == 0
    assert report["max_masked_log_probability_drift"] == 0
    assert report["max_illegal_probability"] == 0
    assert report["hidden_state_bit_equal"]


def test_optimizer_cannot_touch_actor_even_with_weight_decay(actor: RecurrentActorNetwork) -> None:
    before = deepcopy(actor.state_dict())
    rows = AuctionRows(actor)
    optimizer = torch.optim.AdamW(rows.parameters(), lr=0.01, weight_decay=0.1)
    for _ in range(3):
        optimizer.zero_grad()
        rows(torch.ones(2, actor.spec.hidden_size)).square().sum().backward()
        optimizer.step()
    assert all(torch.equal(value, before[name]) for name, value in actor.state_dict().items())


def test_illegal_targets_and_play_data_rejected_before_updates(
    actor: RecurrentActorNetwork,
) -> None:
    examples = generate_auction_examples(range(8_200_000, 8_200_001))
    with pytest.raises(ValueError, match="auction actions"):
        fit_auction_rows(
            actor,
            replace(examples, labels=np.zeros(len(examples), dtype=np.int64)),
            AuctionBCConfig(),
        )
    labels = examples.labels.copy()
    index = next(i for i, m in enumerate(examples.metadata) if m["current_high_bid"] == 12)
    labels[index] = 52
    with pytest.raises(ValueError, match="Illegal teacher"):
        fit_auction_rows(actor, replace(examples, labels=labels), AuctionBCConfig())
    with pytest.raises(ValueError, match="Only BID"):
        fit_auction_rows(
            actor, generate_play_examples(range(8_210_000, 8_210_001)), AuctionBCConfig()
        )
    with pytest.raises(ValueError, match="drift exceeded"):
        fit_auction_rows(actor, examples, AuctionBCConfig(max_parameter_drift=1e-12))


def test_full_history_chunking_matches_incremental_and_resets(actor: RecurrentActorNetwork) -> None:
    fixture = generate_play_examples(range(8_210_000, 8_210_002))
    _, hidden = frozen_features(actor, fixture)
    assert max(fixture.lengths) > 16
    for i in (0, len(fixture) // 2, len(fixture) - 1):
        h = actor.initial_hidden_state(1)
        with torch.no_grad():
            for position in range(int(fixture.lengths[i])):
                h = actor.trunk.history_encoder.advance(
                    torch.from_numpy(fixture.histories[i : i + 1, position : position + 1]),
                    torch.tensor([1]),
                    h,
                )
        torch.testing.assert_close(hidden[:, i : i + 1], h, atol=2e-6, rtol=2e-6)
    # Repeated independent calls start at zero; no persistent state from a prior deal.
    _, again = frozen_features(actor, fixture)
    assert torch.equal(hidden, again)


def test_diagnostics_repeat_for_both_learner_teams(actor: RecurrentActorNetwork) -> None:
    for team in (0, 1):
        examples = generate_auction_examples(range(8_200_000, 8_200_002), learner_team=team)
        result = measure_actor(actor, examples)
        assert result == measure_actor(actor, examples)
        assert set(result["buckets"]["learner_team"]) == {str(team)}
        assert result["max_illegal_probability"] == 0


def test_frozen_training_seating_exposure_is_reported_not_changed() -> None:
    report = training_seating_diagnostic()
    assert report["first_bidder_opportunities"] == 32
    assert report["learner_first_bidder_count"] == 0
    assert report == training_seating_diagnostic()


def test_repair_checkpoint_roundtrip_provenance_and_no_overwrite(tmp_path: Path) -> None:
    actor, critic = build_recurrent_actor_and_critic(
        RecurrentArchitectureSpec(hidden_size=16, static_hidden_dim=24),
        seed=81,
    )
    source = tmp_path / "source.pt"
    save_recurrent_checkpoint(source, actor, critic, None, RulesConfig(), {}, 0, 0, 0, {})
    parent = load_recurrent_checkpoint(source, load_critic=False)
    source_hash = sha256(source)
    repaired, report = fit_auction_rows(
        actor, generate_auction_examples(range(8_200_000, 8_200_001)), AuctionBCConfig(epochs=1)
    )
    destination = tmp_path / "phase11b" / "smoke.pt"
    save_repair_new(destination, source, parent, repaired, report)
    loaded = load_recurrent_checkpoint(destination)
    assert loaded.critic is None
    assert loaded.payload["parameter_counts"]["critic"] == 0
    assert loaded.payload["resume"] is None
    assert loaded.payload["optimizer_state_dict"] is None
    assert loaded.payload["phase11b"]["parent"]["sha256"] == source_hash
    assert sha256(source) == source_hash
    assert_preserved(parent.actor, loaded.actor)
    with pytest.raises(FileExistsError):
        save_repair_new(destination, source, parent, repaired, report)
    with pytest.raises(ValueError, match="phase11b"):
        save_repair_new(source, source, parent, repaired, report)
