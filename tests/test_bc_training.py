"""Behaviour-cloning training, its checkpoint, and the recurrent arena agent.

Three things are checked here that the dataset tests cannot reach:

- cloning actually learns, and does so **recurrently** rather than collapsing to
  a per-decision classifier;
- masking survives training -- an illegal action keeps probability exactly zero,
  and action 64 is never reachable;
- the cloned policy can be played in the frozen arena by an agent whose memory is
  per seat, reset per deal, and equivalent to what training would have produced.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.agents.recurrent_agent import RecurrentNeuralAgent
from tarneeb.core.actions import NO_TRUMP_ACTION
from tarneeb.core.cards import NUM_SEATS
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.env.public_view import PublicViewBuilder
from tarneeb.env.spaces import D_CRITIC, D_STATIC, K_MAX
from tarneeb.evaluation.arena import Arena
from tarneeb.models.bc_checkpoint import (
    BC_CHECKPOINT_FORMAT_VERSION,
    load_bc_checkpoint,
    save_bc_checkpoint,
)
from tarneeb.models.checkpoint import IncompatibleCheckpointError
from tarneeb.models.nets import build_actor_and_critic, count_parameters
from tarneeb.models.recurrent import (
    build_recurrent_actor_and_critic,
    zero_hidden_state,
)
from tarneeb.training.bc import (
    BCConfig,
    evaluate_bc,
    generate_bc_dataset,
    iterate_batches,
    masked_cross_entropy,
    recurrent_batch_log_probabilities,
    train_behaviour_cloning,
)
from tarneeb.training.bc import _to_tensors as to_tensors
from tarneeb.training.phase9_seeds import describe_splits

RULES = RulesConfig()


@pytest.fixture(scope="module")
def batches():
    train = generate_bc_dataset(range(1_000_000, 1_000_120), "bc_train")
    validation = generate_bc_dataset(range(1_040_000, 1_040_030), "bc_validation")
    return {
        "gru": (
            iterate_batches(train, 32, architecture="gru"),
            iterate_batches(validation, 32, architecture="gru"),
        ),
        "ff": (
            iterate_batches(train, 32, architecture="ff"),
            iterate_batches(validation, 32, architecture="ff"),
        ),
        "metadata": train.metadata,
    }


# --- training works ---------------------------------------------------------


def test_cloning_improves_on_an_untrained_policy(batches) -> None:
    train_batches, validation_batches = batches["gru"]
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    before = evaluate_bc(actor, validation_batches).accuracy

    result = train_behaviour_cloning(
        actor, train_batches, validation_batches, BCConfig(epochs=2, batch_sequences=32)
    )

    after = result.validation_metrics[-1].accuracy
    # An untrained masked policy is well above 1/65: many decisions have only a
    # few legal cards, because following suit is forced. So the meaningful claim
    # is the improvement, not the absolute level.
    assert 0.2 < before < 0.55, f"untrained baseline looks wrong: {before:.3f}"
    assert after > before + 0.15, f"cloning should learn; {before:.3f} -> {after:.3f}"
    assert result.optimizer_steps == 2 * len(train_batches)


def test_both_architectures_clone_from_the_same_demonstrations(batches) -> None:
    """A1 needs both conditions to come from identical data."""
    for architecture in ("gru", "ff"):
        train_batches, validation_batches = batches[architecture]
        actor = (
            build_recurrent_actor_and_critic(seed=1)[0]
            if architecture == "gru"
            else build_actor_and_critic(seed=1)[0]
        )
        result = train_behaviour_cloning(
            actor,
            train_batches,
            validation_batches,
            BCConfig(epochs=1, batch_sequences=32, architecture=architecture),
        )
        assert result.validation_metrics[-1].accuracy > 0.3


def test_masking_survives_training(batches) -> None:
    """Illegal actions keep probability exactly zero, not merely small."""
    train_batches, validation_batches = batches["gru"]
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    train_behaviour_cloning(
        actor, train_batches, validation_batches, BCConfig(epochs=1, batch_sequences=32)
    )

    metrics = evaluate_bc(actor, validation_batches)

    assert metrics.max_illegal_probability == 0.0
    assert metrics.max_no_trump_probability == 0.0


def test_metrics_are_split_by_game_phase(batches) -> None:
    """Overall accuracy hides that bidding and card play are different problems."""
    _train_batches, validation_batches = batches["gru"]
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    metrics = evaluate_bc(actor, validation_batches)

    phases = metrics.phase_accuracy()
    counts = metrics.phase_counts()
    assert set(phases) == {"BID", "DECLARE", "PLAY"}
    assert counts["PLAY"] > counts["BID"] > 0
    assert sum(counts.values()) == metrics.decision_count


def test_the_cloning_objective_ignores_padded_positions(batches) -> None:
    """A padded position must contribute exactly zero to the loss."""
    train_batches, _validation = batches["gru"]
    batch = next(b for b in train_batches if not b.valid.all())
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    tensors = to_tensors(batch)

    with torch.no_grad():
        log_probabilities = recurrent_batch_log_probabilities(actor, tensors)
    clean = masked_cross_entropy(log_probabilities, tensors["targets"], tensors["valid"])

    noisy_targets = tensors["targets"].clone()
    padded = ~tensors["valid"]
    noisy_targets[padded] = 0
    noisy = masked_cross_entropy(log_probabilities, noisy_targets, tensors["valid"])

    assert float(clean) == pytest.approx(float(noisy), abs=1e-9)


def test_cloning_trains_the_gru_rather_than_bypassing_it(batches) -> None:
    """If BC were a per-decision classifier the GRU would receive no gradient."""
    train_batches, _validation = batches["gru"]
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    tensors = to_tensors(train_batches[0])

    log_probabilities = recurrent_batch_log_probabilities(actor, tensors)
    loss = masked_cross_entropy(log_probabilities, tensors["targets"], tensors["valid"])
    actor.zero_grad()
    loss.backward()

    gru_gradients = [
        parameter.grad
        for name, parameter in actor.named_parameters()
        if ".gru." in name and parameter.grad is not None
    ]
    assert gru_gradients
    assert any(bool((gradient != 0).any()) for gradient in gru_gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gru_gradients)


def test_the_recurrent_forward_carries_state_across_positions(batches) -> None:
    """Position t's output must depend on the positions before it.

    If it did not, BC would be a per-decision classifier wearing a GRU's name and
    the resulting initialization would be useless for recurrent PPO. The
    comparison is made at a position that is *valid* for every sequence in the
    batch -- a padded position carries the sentinel mask and is identical by
    construction, which would make the test pass for the wrong reason.
    """
    train_batches, _validation = batches["gru"]
    batch = max(train_batches, key=lambda candidate: candidate.sequence_length)
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    tensors = to_tensors(batch)

    later = 3
    assert bool(batch.valid[later].all()), "need a position valid in every sequence"

    with torch.no_grad():
        reference = recurrent_batch_log_probabilities(actor, tensors)
        perturbed = dict(tensors)
        lengths = tensors["segment_lengths"].clone()
        lengths[0] = 0  # erase what the first position contributed to memory
        perturbed["segment_lengths"] = lengths
        changed = recurrent_batch_log_probabilities(actor, perturbed)

    assert not torch.allclose(reference[later], changed[later], atol=1e-6)


# --- the BC checkpoint ------------------------------------------------------


def test_a_bc_checkpoint_round_trips(batches, tmp_path) -> None:
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    path = save_bc_checkpoint(
        path=tmp_path / "bc.pt",
        actor=actor,
        architecture="gru",
        rules=RULES,
        bc_config=BCConfig().as_dict(),
        seed_split=describe_splits(),
        metrics={"test": {"accuracy": 0.5}},
        dataset_metadata=batches["metadata"].as_dict(),
    )

    loaded = load_bc_checkpoint(path, rules=RULES)

    assert loaded.architecture == "gru"
    assert count_parameters(loaded.actor) == 956_617
    for saved, restored in zip(actor.parameters(), loaded.actor.parameters(), strict=True):
        assert torch.equal(saved, restored)
    assert loaded.seed_split["bc_train"]["start"] == 1_000_000


def test_a_feedforward_bc_checkpoint_round_trips(batches, tmp_path) -> None:
    actor, _critic = build_actor_and_critic(seed=1)
    path = save_bc_checkpoint(
        path=tmp_path / "bc_ff.pt",
        actor=actor,
        architecture="ff",
        rules=RULES,
        bc_config=BCConfig(architecture="ff").as_dict(),
        seed_split=describe_splits(),
        metrics={},
        dataset_metadata=batches["metadata"].as_dict(),
    )

    loaded = load_bc_checkpoint(path, rules=RULES)

    assert loaded.architecture == "ff"
    assert count_parameters(loaded.actor) == 1_031_041


def test_the_bc_format_is_distinct_from_the_ppo_formats() -> None:
    from tarneeb.models.checkpoint import CHECKPOINT_FORMAT_VERSION
    from tarneeb.models.recurrent_checkpoint import RECURRENT_CHECKPOINT_FORMAT_VERSION

    assert BC_CHECKPOINT_FORMAT_VERSION == "bc_actor_v1"
    assert BC_CHECKPOINT_FORMAT_VERSION not in (
        CHECKPOINT_FORMAT_VERSION,
        RECURRENT_CHECKPOINT_FORMAT_VERSION,
    )


def test_a_bc_checkpoint_from_a_different_teacher_is_refused(batches, tmp_path) -> None:
    """A clone of a different teacher is a different experiment."""
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    path = save_bc_checkpoint(
        path=tmp_path / "bc.pt",
        actor=actor,
        architecture="gru",
        rules=RULES,
        bc_config={},
        seed_split={},
        metrics={},
        dataset_metadata={},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["teacher"] = "SomeOtherAgent"
    torch.save(payload, path)

    with pytest.raises(IncompatibleCheckpointError, match="teacher"):
        load_bc_checkpoint(path, rules=RULES)


def test_a_bc_checkpoint_with_a_changed_fingerprint_is_refused(tmp_path) -> None:
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    path = save_bc_checkpoint(
        path=tmp_path / "bc.pt",
        actor=actor,
        architecture="gru",
        rules=RULES,
        bc_config={},
        seed_split={},
        metrics={},
        dataset_metadata={},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["rules_fingerprint"] = "sha1:0000"
    torch.save(payload, path)

    with pytest.raises(IncompatibleCheckpointError, match="rules_fingerprint"):
        load_bc_checkpoint(path, rules=RULES)


# --- the recurrent arena agent ----------------------------------------------


def test_the_agent_keeps_one_memory_per_seat() -> None:
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    agent = RecurrentNeuralAgent(actor)
    assert agent.states_are_separate()
    for seat in range(NUM_SEATS):
        assert torch.equal(agent.hidden_state(seat), zero_hidden_state(1, actor.spec))
        assert agent.consumed_event_count(seat) == 0


def test_reset_clears_every_seats_memory() -> None:
    """The arena calls ``reset`` per deal; that is the deal boundary."""
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    agent = RecurrentNeuralAgent(actor)
    engine = TarneebEngine(RULES)
    views = PublicViewBuilder(engine)
    state = engine.new_deal(0, deal_index=0, match_seed=99)
    engine.deal(state)
    for _ in range(6):
        seat = state.current_player_seat
        assert seat is not None
        engine.step(state, agent.act(views.build(state, seat)))

    assert any(agent.consumed_event_count(seat) > 0 for seat in range(NUM_SEATS))
    agent.reset()
    for seat in range(NUM_SEATS):
        assert agent.consumed_event_count(seat) == 0
        assert float(agent.hidden_state(seat).abs().sum()) == 0.0


def test_the_agent_consumes_each_event_exactly_once() -> None:
    """Its cursor must track the deal's event stream without gap or repeat."""
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    agent = RecurrentNeuralAgent(actor)
    engine = TarneebEngine(RULES)
    views = PublicViewBuilder(engine)
    state = engine.new_deal(0, deal_index=0, match_seed=101)
    engine.deal(state)

    while state.phase in (Phase.BID, Phase.DECLARE, Phase.PLAY):
        seat = state.current_player_seat
        assert seat is not None
        before = len(state.events)
        agent.act(views.build(state, seat))
        assert agent.consumed_event_count(seat) == before
        engine.step(state, StrongHeuristicAgent().act(views.build(state, seat)))
        if state.phase is Phase.DEAL:
            break


def test_the_agent_plays_legally_in_the_frozen_arena() -> None:
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    agent = RecurrentNeuralAgent(actor, deterministic=True)
    arena = Arena()

    evaluation = arena.evaluate(
        agent,
        RandomAgent(seed=0),
        paired_deal_count=6,
        evaluation_seed=4_000_000,
        bootstrap_resamples=50,
    )

    assert arena.illegal_action_count == 0
    assert evaluation.metrics["illegal_action_rate"] == 0.0
    assert evaluation.metrics["no_trump_action_rate"] == 0.0
    assert evaluation.metrics["replay_integrity_rate"] == 1.0


def test_the_agent_is_deterministic_when_asked_to_be() -> None:
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    arena = Arena()
    first = arena.evaluate(
        RecurrentNeuralAgent(actor, deterministic=True),
        RandomAgent(seed=0),
        paired_deal_count=4,
        evaluation_seed=4_000_000,
        bootstrap_resamples=50,
    )
    second = arena.evaluate(
        RecurrentNeuralAgent(actor, deterministic=True),
        RandomAgent(seed=0),
        paired_deal_count=4,
        evaluation_seed=4_000_000,
        bootstrap_resamples=50,
    )
    assert first.metrics["mean_paired_score_difference"] == pytest.approx(
        second.metrics["mean_paired_score_difference"]
    )


def test_the_agent_never_receives_the_centralized_critic_vector() -> None:
    """Structural: it builds a 692-wide observation and holds no critic."""
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    agent = RecurrentNeuralAgent(actor)
    engine = TarneebEngine(RULES)
    views = PublicViewBuilder(engine)
    state = engine.new_deal(0, deal_index=0, match_seed=7)
    engine.deal(state)
    seat = state.current_player_seat
    assert seat is not None

    observation, segment_ids, segment_length = agent.encode(views.build(state, seat))

    assert observation.shape == (1, D_STATIC)
    assert observation.shape[-1] != D_CRITIC
    assert segment_ids.shape == (1, K_MAX, 7)
    assert int(segment_length.item()) <= K_MAX
    assert not hasattr(agent, "critic")


def test_the_agent_output_is_invariant_to_the_other_hands() -> None:
    """Invariant I7 for the deployment path."""
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    agent = RecurrentNeuralAgent(actor)
    engine = TarneebEngine(RULES)
    views = PublicViewBuilder(engine)
    state = engine.new_deal(0, deal_index=0, match_seed=2024)
    engine.deal(state)
    teacher = StrongHeuristicAgent()
    for _ in range(10):
        seat = state.current_player_seat
        assert seat is not None
        engine.step(state, teacher.act(views.build(state, seat)))

    observer = state.current_player_seat
    assert observer is not None
    before = agent.encode(views.build(state, observer))[0].clone()

    from tarneeb.core.cards import cards_in_hand

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

    after = agent.encode(views.build(state, observer))[0]
    assert torch.equal(before, after)


def test_action_sixty_four_is_never_selected() -> None:
    actor, _critic = build_recurrent_actor_and_critic(seed=1)
    agent = RecurrentNeuralAgent(actor, deterministic=False, seed=5)
    engine = TarneebEngine(RULES)
    views = PublicViewBuilder(engine)
    for match_seed in (11, 12, 13):
        state = engine.new_deal(0, deal_index=0, match_seed=match_seed)
        engine.deal(state)
        agent.reset()
        while state.phase in (Phase.BID, Phase.DECLARE, Phase.PLAY):
            seat = state.current_player_seat
            assert seat is not None
            action = agent.act(views.build(state, seat))
            assert action != NO_TRUMP_ACTION
            engine.step(state, action)
            if state.phase is Phase.DEAL:
                break
