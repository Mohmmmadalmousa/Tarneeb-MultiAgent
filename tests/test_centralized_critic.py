"""Ablation A2: the centralized critic, and the boundary it must not cross.

The central proof in this file is a paired positive/negative control:

    same actor-visible state, different hidden opponent hands
        -> the actor output is IDENTICAL          (negative control)
        -> the centralized critic value DIFFERS   (positive control)

Both halves are needed. Without the first, a centralized critic could be leaking
into the policy. Without the second, the privileged input might be wired up but
ignored, and A2 would be comparing a treatment against itself.

Everything else here guards the properties Phase 10 inherits rather than
re-derives: G6, epoch-zero, isolation, determinism and the action-space safety
invariants.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.core.actions import NO_TRUMP_ACTION
from tarneeb.core.cards import NUM_SEATS, cards_in_hand
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.public_view import PublicViewBuilder
from tarneeb.env.spaces import D_CRITIC, D_STATIC, K_MAX, N_EVENT_FIELDS
from tarneeb.env.vec import VecTarneeb
from tarneeb.models.centralized_critic import (
    ACTOR_PARAMETERS,
    CENTRALIZED_CRITIC_PARAMETERS,
    CENTRALIZED_EXTRA_PARAMETERS,
    CRITIC_MODE_CENTRALIZED,
    CRITIC_MODE_DECENTRALIZED,
    CRITIC_MODES,
    DECENTRALIZED_CRITIC_PARAMETERS,
    architecture_spec_for,
    build_actor_and_critic_for,
    critic_observation_dim_for,
    describe_condition,
    expected_critic_parameters,
    verify_parameter_counts,
)
from tarneeb.models.nets import count_parameters, parameters_are_disjoint
from tarneeb.models.recurrent import zero_hidden_state
from tarneeb.training.centralized_rollout import (
    CentralizedCriticCollector,
    DecentralizedCriticCollector,
    assert_actor_input_is_actor_safe,
)
from tarneeb.training.recurrent_state import LEARNER_POLICY_ID

RULES = RulesConfig()
ENVIRONMENTS = 4
BUDGET = 40


def midgame_state(match_seed: int = 7788):
    """Return an engine, builder and a state partway through the play phase."""
    engine = TarneebEngine(RULES)
    builder = ObservationBuilder(engine)
    views = PublicViewBuilder(engine)
    teacher = StrongHeuristicAgent()
    state = engine.new_deal(0, deal_index=0, match_seed=match_seed)
    engine.deal(state)
    for _ in range(400):
        if state.phase not in (Phase.BID, Phase.DECLARE, Phase.PLAY):
            break
        if state.phase is Phase.PLAY and state.trick_number >= 3:
            break
        seat = state.current_player_seat
        assert seat is not None
        engine.step(state, teacher.act(views.build(state, seat)))
    return engine, builder, state


def permute_hidden_hands(state, observer: int, seed: int = 0) -> None:
    """Redistribute the three unseen hands among the other seats, in place."""
    others = [seat for seat in range(NUM_SEATS) if seat != observer]
    pooled = [card for seat in others for card in cards_in_hand(state.hands[seat])]
    np.random.default_rng(seed).shuffle(pooled)
    cursor = 0
    for seat in others:
        size = len(cards_in_hand(state.hands[seat]))
        hand = 0
        for card in pooled[cursor : cursor + size]:
            hand |= 1 << card
        state.hands[seat] = hand
        cursor += size


def build_collector(
    collector_class, critic_mode, seed=5_000_000, environments=ENVIRONMENTS, budget=BUDGET
):
    vector_environments = VecTarneeb(num_envs=environments, root_seed=seed)
    actor, critic = build_actor_and_critic_for(critic_mode, seed=1)
    generator = torch.Generator()
    generator.manual_seed(2)
    return collector_class(
        environments=vector_environments,
        actor=actor,
        critic=critic,
        rollout_length=budget,
        action_sampling_generator=generator,
        opponents=[BasicHeuristicAgent() for _ in range(environments)],
    )


# --- the central privileged-information control -----------------------------


def test_hidden_hands_move_the_centralized_critic_but_not_the_actor() -> None:
    """The paired control that defines Phase 10.

    One state, two versions differing ONLY in how the unseen cards are
    distributed among the other three seats. The actor must not notice. The
    centralized critic must.
    """
    _engine, builder, state = midgame_state()
    observer = state.current_player_seat
    assert observer is not None

    actor, centralized = build_actor_and_critic_for(CRITIC_MODE_CENTRALIZED, seed=3)
    _actor2, decentralized = build_actor_and_critic_for(CRITIC_MODE_DECENTRALIZED, seed=3)

    empty_ids = torch.zeros((1, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
    empty_lengths = torch.zeros(1, dtype=torch.int64)

    def observe():
        actor_view = torch.from_numpy(builder.static_actor(state, observer)).unsqueeze(0)
        critic_view = torch.from_numpy(builder.static_critic(state, observer)).unsqueeze(0)
        return actor_view, critic_view

    before_actor_input, before_critic_input = observe()
    with torch.no_grad():
        before_logits, _ = actor(
            before_actor_input, empty_ids, empty_lengths, zero_hidden_state(1, actor.spec)
        )
        before_value, _ = centralized(
            before_critic_input,
            empty_ids,
            empty_lengths,
            zero_hidden_state(1, centralized.spec),
        )
        before_plain, _ = decentralized(
            before_actor_input,
            empty_ids,
            empty_lengths,
            zero_hidden_state(1, decentralized.spec),
        )

    permute_hidden_hands(state, observer, seed=1)
    after_actor_input, after_critic_input = observe()
    with torch.no_grad():
        after_logits, _ = actor(
            after_actor_input, empty_ids, empty_lengths, zero_hidden_state(1, actor.spec)
        )
        after_value, _ = centralized(
            after_critic_input,
            empty_ids,
            empty_lengths,
            zero_hidden_state(1, centralized.spec),
        )
        after_plain, _ = decentralized(
            after_actor_input,
            empty_ids,
            empty_lengths,
            zero_hidden_state(1, decentralized.spec),
        )

    # Negative control: the actor sees nothing, so nothing it produces moves.
    assert torch.equal(before_actor_input, after_actor_input)
    assert torch.equal(before_logits, after_logits)
    assert torch.equal(before_plain, after_plain)

    # Positive control: the critic input changed, and the value moved with it.
    assert not torch.equal(before_critic_input, after_critic_input)
    assert float((before_value - after_value).abs().max()) > 1e-6


def test_the_critic_only_block_is_where_the_two_inputs_differ() -> None:
    """The 856 vector is the 692 vector plus 164 privileged features."""
    _engine, builder, state = midgame_state(match_seed=4242)
    observer = state.current_player_seat
    assert observer is not None

    actor_view = builder.static_actor(state, observer)
    critic_view = builder.static_critic(state, observer)

    assert actor_view.shape == (D_STATIC,)
    assert critic_view.shape == (D_CRITIC,)
    assert np.array_equal(critic_view[:D_STATIC], actor_view)
    assert D_CRITIC - D_STATIC == 164


# --- the actor boundary is unchanged ----------------------------------------


def test_the_actor_still_rejects_the_856_observation() -> None:
    """Structural: the actor owns no parameter shaped to accept 856 inputs."""
    actor, _critic = build_actor_and_critic_for(CRITIC_MODE_CENTRALIZED, seed=1)
    empty_ids = torch.zeros((1, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
    empty_lengths = torch.zeros(1, dtype=torch.int64)
    with pytest.raises(ValueError, match="692 features"):
        actor(
            torch.zeros((1, D_CRITIC)),
            empty_ids,
            empty_lengths,
            zero_hidden_state(1, actor.spec),
        )


def test_the_centralized_critic_rejects_the_692_observation() -> None:
    _actor, critic = build_actor_and_critic_for(CRITIC_MODE_CENTRALIZED, seed=1)
    empty_ids = torch.zeros((1, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
    empty_lengths = torch.zeros(1, dtype=torch.int64)
    with pytest.raises(ValueError, match="856 features"):
        critic(
            torch.zeros((1, D_STATIC)),
            empty_ids,
            empty_lengths,
            zero_hidden_state(1, critic.spec),
        )


def test_the_actor_is_identical_across_both_conditions() -> None:
    """A2 varies the critic. If the actor differed, it would not be an ablation."""
    control_actor, _c = build_actor_and_critic_for(CRITIC_MODE_DECENTRALIZED, seed=11)
    treatment_actor, _t = build_actor_and_critic_for(CRITIC_MODE_CENTRALIZED, seed=11)

    assert control_actor.spec.actor_observation_dim == treatment_actor.spec.actor_observation_dim
    for left, right in zip(control_actor.parameters(), treatment_actor.parameters(), strict=True):
        assert torch.equal(left, right)


# --- widths and parameter counts --------------------------------------------


def test_parameter_counts_match_the_contract() -> None:
    for mode in CRITIC_MODES:
        actor, critic = build_actor_and_critic_for(mode, seed=1)
        counts = verify_parameter_counts(actor, critic, mode)
        assert counts["actor"] == ACTOR_PARAMETERS == 956_617
        assert counts["critic"] == expected_critic_parameters(mode)
        assert counts["actor_observation_dim"] == D_STATIC


def test_the_centralized_critic_costs_exactly_the_extra_block() -> None:
    """1,024,137 = 940,169 + 164 * 512, the blueprint figure, computed here."""
    assert CENTRALIZED_EXTRA_PARAMETERS == (D_CRITIC - D_STATIC) * 512 == 83_968
    assert (
        CENTRALIZED_CRITIC_PARAMETERS
        == DECENTRALIZED_CRITIC_PARAMETERS + CENTRALIZED_EXTRA_PARAMETERS
        == 1_024_137
    )
    _actor, critic = build_actor_and_critic_for(CRITIC_MODE_CENTRALIZED, seed=1)
    assert count_parameters(critic) == 1_024_137


def test_the_spec_factory_moves_only_the_critic_width() -> None:
    control = architecture_spec_for(CRITIC_MODE_DECENTRALIZED)
    treatment = architecture_spec_for(CRITIC_MODE_CENTRALIZED)
    assert control.actor_observation_dim == treatment.actor_observation_dim == D_STATIC
    assert control.critic_observation_dim == D_STATIC
    assert treatment.critic_observation_dim == D_CRITIC
    assert control.hidden_size == treatment.hidden_size
    assert control.gru_layer_count == treatment.gru_layer_count
    assert control.event_embedding_dims == treatment.event_embedding_dims


def test_an_unknown_critic_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="critic_mode must be one of"):
        critic_observation_dim_for("omniscient")
    with pytest.raises(ValueError, match="critic_mode must be one of"):
        expected_critic_parameters("omniscient")


def test_verification_catches_a_mismatched_configuration() -> None:
    """A configuration mistake must stop a run, not silently change the question."""
    actor, critic = build_actor_and_critic_for(CRITIC_MODE_DECENTRALIZED, seed=1)
    with pytest.raises(ValueError, match="centralized critic must have"):
        verify_parameter_counts(actor, critic, CRITIC_MODE_CENTRALIZED)


def test_the_condition_description_records_the_privileged_flag() -> None:
    control = describe_condition(CRITIC_MODE_DECENTRALIZED)
    treatment = describe_condition(CRITIC_MODE_CENTRALIZED)
    assert control["privileged"] is False
    assert treatment["privileged"] is True
    assert control["actor_observation_dim"] == treatment["actor_observation_dim"] == 692
    assert treatment["critic_observation_dim"] == 856


def test_actor_and_critic_parameters_are_disjoint() -> None:
    for mode in CRITIC_MODES:
        actor, critic = build_actor_and_critic_for(mode, seed=1)
        assert parameters_are_disjoint(actor, critic)


# --- collection keeps the two inputs separate -------------------------------


@pytest.fixture(scope="module")
def centralized_run():
    collector = build_collector(CentralizedCriticCollector, CRITIC_MODE_CENTRALIZED)
    buffer, metrics = collector.collect()
    return collector, buffer, metrics


@pytest.fixture(scope="module")
def decentralized_run():
    collector = build_collector(DecentralizedCriticCollector, CRITIC_MODE_DECENTRALIZED)
    buffer, metrics = collector.collect()
    return collector, buffer, metrics


def test_the_stored_inputs_have_the_right_widths(centralized_run, decentralized_run) -> None:
    """The one wiring mistake that would matter, checked by shape."""
    for run, critic_dim in ((centralized_run, D_CRITIC), (decentralized_run, D_STATIC)):
        collector, buffer, _metrics = run
        assert assert_actor_input_is_actor_safe(buffer, D_STATIC, critic_dim)
        for sequence in buffer.sequences.values():
            for transition in sequence:
                assert transition.static.shape == (D_STATIC,)
                assert transition.critic_static.shape == (critic_dim,)
        del collector


def test_the_centralized_batch_carries_both_widths(centralized_run) -> None:
    collector, buffer, _metrics = centralized_run
    batch = buffer.to_chunks(collector.bootstrap_values(), 1.0, 0.95)
    assert batch.static.shape[-1] == D_STATIC
    assert batch.critic_static.shape[-1] == D_CRITIC
    assert batch.is_finite()


def test_the_decentralized_condition_stores_the_actor_vector_twice(decentralized_run) -> None:
    """The control must be byte-identical to the Phase 9 behaviour."""
    _collector, buffer, _metrics = decentralized_run
    for sequence in buffer.sequences.values():
        for transition in sequence:
            assert np.array_equal(transition.static, transition.critic_static)


def test_the_centralized_condition_stores_a_wider_critic_vector(centralized_run) -> None:
    _collector, buffer, _metrics = centralized_run
    for sequence in buffer.sequences.values():
        for transition in sequence:
            assert transition.critic_static.shape[0] > transition.static.shape[0]
            assert np.array_equal(transition.critic_static[:D_STATIC], transition.static)


def test_both_conditions_collect_the_same_amount(centralized_run, decentralized_run) -> None:
    """Same actor, same seeds, same schedule: only the critic input differs."""
    central_collector, _cb, central_metrics = centralized_run
    plain_collector, _pb, plain_metrics = decentralized_run
    assert central_collector.last_collection_steps == plain_collector.last_collection_steps
    assert central_metrics.completed_deals == plain_metrics.completed_deals
    assert central_metrics.learner_decision_count == plain_metrics.learner_decision_count


# --- inherited Phase 9 guarantees --------------------------------------------


def test_g6_is_inherited(centralized_run) -> None:
    """No unfinished recurrent state may survive an optimizer update."""
    collector, buffer, _metrics = centralized_run
    assert collector.no_unfinished_learner_sequence(buffer)
    assert collector.recurrent_state_is_zero()
    for sequence in buffer.sequences.values():
        assert sequence[-1].done


def test_epoch_zero_holds_for_the_centralized_critic(centralized_run) -> None:
    from tarneeb.training.recurrent_ppo import (
        RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE,
        replay_chunks,
    )

    collector, buffer, _metrics = centralized_run
    batch = buffer.to_chunks(collector.bootstrap_values(), 1.0, 0.95)
    replayed = replay_chunks(collector.actor, collector.critic, batch)
    assert (
        replayed.max_log_probability_deviation(batch)
        < RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
    )
    assert replayed.max_value_deviation(batch) < 1e-3


def test_a_corrupted_initial_state_still_breaks_epoch_zero(centralized_run) -> None:
    """The negative control for the gate above."""
    from dataclasses import replace as dataclass_replace

    from tarneeb.training.recurrent_ppo import (
        RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE,
        replay_chunks,
    )

    collector, buffer, _metrics = centralized_run
    batch = buffer.to_chunks(collector.bootstrap_values(), 1.0, 0.95)
    corrupted = dataclass_replace(batch, initial_actor_hidden=batch.initial_actor_hidden + 1.0)
    replayed = replay_chunks(collector.actor, collector.critic, corrupted)
    assert (
        replayed.max_log_probability_deviation(corrupted)
        > RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
    )


def test_seat_and_environment_isolation(centralized_run) -> None:
    collector, _buffer, _metrics = centralized_run
    actor_state = collector.state.actor_hidden[LEARNER_POLICY_ID]
    critic_state = collector.state.critic_hidden[LEARNER_POLICY_ID]
    assert actor_state.data_ptr() != critic_state.data_ptr()
    actor_state[1, 2] = 7.0
    assert float(actor_state[0].abs().sum()) == 0.0
    assert float(actor_state[1, 0].abs().sum()) == 0.0
    actor_state.zero_()


def test_event_consumption_stays_exact(centralized_run) -> None:
    collector, buffer, metrics = centralized_run
    collector.assert_consumption_invariant()
    assert metrics.total_events_consumed % NUM_SEATS == 0
    for sequence in buffer.sequences.values():
        assert sequence[0].previous_event_length == 0


def test_no_illegal_action_and_no_no_trump_action(centralized_run) -> None:
    _collector, buffer, metrics = centralized_run
    assert metrics.illegal_action_count == 0
    assert metrics.no_trump_action_count == 0
    for sequence in buffer.sequences.values():
        for transition in sequence:
            assert bool(transition.legal_action_mask[transition.action])
            assert not bool(transition.legal_action_mask[NO_TRUMP_ACTION])


def test_the_centralized_rollout_is_deterministic() -> None:
    first, _ = build_collector(
        CentralizedCriticCollector, CRITIC_MODE_CENTRALIZED, seed=5_100_000
    ).collect()
    second, _ = build_collector(
        CentralizedCriticCollector, CRITIC_MODE_CENTRALIZED, seed=5_100_000
    ).collect()
    assert first.sequences.keys() == second.sequences.keys()
    for key, sequence in first.sequences.items():
        for left, right in zip(sequence, second.sequences[key], strict=True):
            assert left.action == right.action
            assert left.value == right.value


def test_a_different_seed_diverges() -> None:
    first, _ = build_collector(
        CentralizedCriticCollector, CRITIC_MODE_CENTRALIZED, seed=5_100_000
    ).collect()
    second, _ = build_collector(
        CentralizedCriticCollector, CRITIC_MODE_CENTRALIZED, seed=5_200_000
    ).collect()
    left = [t.action for k in sorted(first.sequences) for t in first.sequences[k]]
    right = [t.action for k in sorted(second.sequences) for t in second.sequences[k]]
    assert left != right
