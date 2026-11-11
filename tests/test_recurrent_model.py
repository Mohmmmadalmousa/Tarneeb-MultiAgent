"""Shapes, parameter counts, masking and event-encoding for the Event-GRU nets.

These are the tests that must pass before any recurrent state management is
trusted, because every later test assumes the network itself is the shape the
contract says it is.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarneeb.core.actions import NO_TRUMP_ACTION, NUM_ACTIONS
from tarneeb.env.event_features import EVENT_FIELD_CARDINALITIES
from tarneeb.env.spaces import D_CRITIC, D_STATIC, K_MAX, N_EVENT_FIELDS
from tarneeb.models.masking import MaskedCategorical
from tarneeb.models.nets import (
    ARCHITECTURE_SPEC_VERSION,
    count_parameters,
    parameters_are_disjoint,
)
from tarneeb.models.recurrent import (
    EVENT_CONCAT_DIM,
    EVENT_EMBEDDING_DIMS,
    GRU_RESET_DEAL,
    RECURRENT_ARCHITECTURE_SPEC_VERSION,
    RecurrentActorNetwork,
    RecurrentArchitectureSpec,
    RecurrentCriticNetwork,
    build_recurrent_actor_and_critic,
    zero_hidden_state,
)

BATCH = 5


@pytest.fixture(scope="module")
def networks() -> tuple[RecurrentActorNetwork, RecurrentCriticNetwork]:
    return build_recurrent_actor_and_critic(seed=17)


def random_segments(
    batch: int, lengths: list[int], generator: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(ids (B, K_MAX, 7), lengths (B,))`` with in-range random ids."""
    ids = np.zeros((batch, K_MAX, N_EVENT_FIELDS), dtype=np.int64)
    for row, length in enumerate(lengths):
        for field_index, cardinality in enumerate(EVENT_FIELD_CARDINALITIES):
            ids[row, :length, field_index] = generator.integers(0, cardinality, size=length)
    return torch.from_numpy(ids), torch.tensor(lengths, dtype=torch.int64)


def legal_masks(batch: int, generator: np.random.Generator) -> torch.Tensor:
    """Return plausible legal-action masks: never empty, never action 64."""
    masks = np.zeros((batch, NUM_ACTIONS), dtype=bool)
    for row in range(batch):
        chosen = generator.choice(NO_TRUMP_ACTION, size=6, replace=False)
        masks[row, chosen] = True
    return torch.from_numpy(masks)


# --- versions and dimensions -----------------------------------------------


def test_recurrent_architecture_version_is_distinct_from_the_feedforward_one() -> None:
    assert RECURRENT_ARCHITECTURE_SPEC_VERSION == "event_gru_actor_critic_v1"
    assert RECURRENT_ARCHITECTURE_SPEC_VERSION != ARCHITECTURE_SPEC_VERSION


def test_event_embedding_widths_match_the_blueprint_specification() -> None:
    """Blueprint section 5.4 gives these exactly; they are not a free choice."""
    assert EVENT_EMBEDDING_DIMS == (16, 8, 8, 8, 32, 8, 8)
    assert EVENT_CONCAT_DIM == 88
    assert len(EVENT_EMBEDDING_DIMS) == N_EVENT_FIELDS


def test_event_encoder_parameter_count_confirms_the_embedding_split() -> None:
    """The 7,816 total is what pins this split rather than any other summing to 88.

    Tables cost ``sum(cardinality * width)`` and the projection costs
    ``88 * 64 + 64``. Only the blueprint's widths reproduce section 6.3's figure.
    """
    table_parameters = sum(
        cardinality * width
        for cardinality, width in zip(EVENT_FIELD_CARDINALITIES, EVENT_EMBEDDING_DIMS, strict=True)
    )
    assert table_parameters == 2_120
    assert EVENT_CONCAT_DIM * 64 + 64 == 5_696
    assert table_parameters + 5_696 == 7_816


def test_parameter_counts_are_exactly_the_phase_eight_contract(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    """Computed, not copied. The critic figure is Phase 8's, not the blueprint's.

    The blueprint's 1,024,137 critic is the 856-dim centralized one, which is
    Phase 10. Phase 8's critic reads the 692-dim actor-safe observation, so its
    static encoder is 83,968 parameters smaller.
    """
    actor, critic = networks
    assert count_parameters(actor) == 956_617
    assert count_parameters(critic) == 940_169
    assert count_parameters(actor) + count_parameters(critic) == 1_896_786

    centralized_static_cost = (D_CRITIC - D_STATIC) * 512
    assert centralized_static_cost == 83_968
    assert count_parameters(critic) + centralized_static_cost == 1_024_137


def test_history_path_parameter_count_matches_the_blueprint(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    actor, _critic = networks
    encoder = actor.trunk.history_encoder
    embedding_parameters = sum(p.numel() for p in encoder.embedding.parameters())
    gru_parameters = sum(p.numel() for p in encoder.gru.parameters())
    assert embedding_parameters == 7_816
    assert gru_parameters == 247_296
    assert embedding_parameters + gru_parameters == 255_112


def test_actor_and_critic_parameters_are_disjoint(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    actor, critic = networks
    assert parameters_are_disjoint(actor, critic)


def test_actor_and_critic_do_not_share_a_gru_or_an_embedding_table(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    """Disjoint parameter sets stated at the module level, checked at the layer."""
    actor, critic = networks
    assert actor.trunk.history_encoder is not critic.trunk.history_encoder
    assert parameters_are_disjoint(
        actor.trunk.history_encoder.gru, critic.trunk.history_encoder.gru
    )
    assert parameters_are_disjoint(
        actor.trunk.history_encoder.embedding, critic.trunk.history_encoder.embedding
    )


# --- forward shapes ---------------------------------------------------------


def test_actor_returns_logits_and_advanced_state_of_the_right_shape(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    actor, _critic = networks
    generator = np.random.default_rng(0)
    ids, lengths = random_segments(BATCH, [0, 1, 5, 16, 3], generator)
    observations = torch.zeros((BATCH, D_STATIC))
    hidden = zero_hidden_state(BATCH, actor.spec)

    logits, advanced = actor(observations, ids, lengths, hidden)

    assert logits.shape == (BATCH, NUM_ACTIONS)
    assert advanced.shape == (actor.spec.gru_layer_count, BATCH, actor.spec.hidden_size)
    assert bool(torch.isfinite(logits).all())


def test_critic_returns_scalar_values_and_advanced_state(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    _actor, critic = networks
    generator = np.random.default_rng(1)
    ids, lengths = random_segments(BATCH, [2, 0, 7, 1, 0], generator)
    observations = torch.zeros((BATCH, critic.spec.critic_observation_dim))
    hidden = zero_hidden_state(BATCH, critic.spec)

    values, advanced = critic(observations, ids, lengths, hidden)

    assert values.shape == (BATCH,)
    assert advanced.shape == (critic.spec.gru_layer_count, BATCH, critic.spec.hidden_size)


def test_actor_refuses_an_observation_of_the_wrong_width(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    """A wiring mistake that fed the actor the 856-dim critic vector must raise."""
    actor, _critic = networks
    generator = np.random.default_rng(2)
    ids, lengths = random_segments(1, [0], generator)
    with pytest.raises(ValueError, match="692 features"):
        actor(torch.zeros((1, D_CRITIC)), ids, lengths, zero_hidden_state(1, actor.spec))


# --- the empty-segment no-op ------------------------------------------------


def test_an_empty_segment_leaves_the_hidden_state_exactly_unchanged(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    """Rule R2 and R4 both depend on this being exact, not merely close."""
    actor, _critic = networks
    encoder = actor.trunk.history_encoder
    hidden = torch.randn(actor.spec.gru_layer_count, 4, actor.spec.hidden_size)
    ids = torch.zeros((4, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
    lengths = torch.zeros(4, dtype=torch.int64)

    advanced = encoder.advance(ids, lengths, hidden)

    assert torch.equal(advanced, hidden)


def test_a_mixed_batch_leaves_only_the_empty_rows_unchanged(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    """The no-op must be per row, not all-or-nothing for the batch."""
    actor, _critic = networks
    encoder = actor.trunk.history_encoder
    generator = np.random.default_rng(3)
    ids, lengths = random_segments(4, [0, 3, 0, 5], generator)
    hidden = torch.randn(actor.spec.gru_layer_count, 4, actor.spec.hidden_size)

    advanced = encoder.advance(ids, lengths, hidden)

    assert torch.equal(advanced[:, 0], hidden[:, 0])
    assert torch.equal(advanced[:, 2], hidden[:, 2])
    assert not torch.equal(advanced[:, 1], hidden[:, 1])
    assert not torch.equal(advanced[:, 3], hidden[:, 3])


def test_padding_inside_a_segment_cannot_change_the_result(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    """Rows beyond ``prev_ev_len`` are excluded by packing, so noise there is inert."""
    actor, _critic = networks
    encoder = actor.trunk.history_encoder
    generator = np.random.default_rng(4)
    ids, lengths = random_segments(3, [2, 6, 1], generator)
    hidden = torch.randn(actor.spec.gru_layer_count, 3, actor.spec.hidden_size)
    clean = encoder.advance(ids, lengths, hidden)

    noisy = ids.clone()
    for row, length in enumerate(lengths.tolist()):
        for field_index, cardinality in enumerate(EVENT_FIELD_CARDINALITIES):
            noisy[row, length:, field_index] = int(cardinality) - 1

    assert torch.equal(encoder.advance(noisy, lengths, hidden), clean)


def test_a_segment_longer_than_k_max_is_refused(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    actor, _critic = networks
    encoder = actor.trunk.history_encoder
    ids = torch.zeros((1, K_MAX + 4, N_EVENT_FIELDS), dtype=torch.int64)
    lengths = torch.tensor([K_MAX + 1], dtype=torch.int64)
    with pytest.raises(ValueError, match="K_MAX"):
        encoder.advance(ids, lengths, zero_hidden_state(1, actor.spec))


def test_an_out_of_range_event_id_is_refused(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    actor, _critic = networks
    ids = torch.zeros((1, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
    ids[0, 0, 4] = EVENT_FIELD_CARDINALITIES[4]  # one past the card field
    lengths = torch.tensor([1], dtype=torch.int64)
    with pytest.raises(ValueError, match="out of range"):
        actor.trunk.history_encoder.advance(ids, lengths, zero_hidden_state(1, actor.spec))


# --- incremental versus batch equivalence -----------------------------------


def test_processing_events_one_at_a_time_matches_processing_them_at_once(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    """The single most important recurrent property (blueprint T-E5).

    The rollout advances a seat's state a few events at a time as the engine
    emits them; replay reconstructs the same state from stored segments. If these
    two paths disagreed, every reconstructed decision would be evaluated under a
    state the policy never actually had.
    """
    actor, _critic = networks
    encoder = actor.trunk.history_encoder
    generator = np.random.default_rng(5)
    total = 12
    ids, _lengths = random_segments(1, [total], generator)

    at_once = encoder.advance(
        ids, torch.tensor([total], dtype=torch.int64), zero_hidden_state(1, actor.spec)
    )

    incremental = zero_hidden_state(1, actor.spec)
    for position in range(total):
        single = torch.zeros((1, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
        single[0, 0] = ids[0, position]
        incremental = encoder.advance(single, torch.tensor([1], dtype=torch.int64), incremental)

    assert torch.allclose(at_once, incremental, atol=1e-5)


def test_arbitrary_segmentations_of_one_stream_agree(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    """Any split of the same ordered stream must reach the same state."""
    actor, _critic = networks
    encoder = actor.trunk.history_encoder
    generator = np.random.default_rng(6)
    total = 14
    ids, _lengths = random_segments(1, [total], generator)

    def consume(splits: list[int]) -> torch.Tensor:
        hidden = zero_hidden_state(1, actor.spec)
        cursor = 0
        for size in splits:
            block = torch.zeros((1, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
            block[0, :size] = ids[0, cursor : cursor + size]
            hidden = encoder.advance(block, torch.tensor([size], dtype=torch.int64), hidden)
            cursor += size
        return hidden

    reference = consume([total])
    for splits in ([7, 7], [1, 2, 3, 4, 4], [13, 1], [4, 0, 6, 0, 4]):
        assert torch.allclose(consume(splits), reference, atol=1e-5)


def test_event_order_changes_the_encoded_state(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    """The encoder is order-sensitive by construction.

    This asserts the *input* distinction is preserved, not that a trained network
    exploits it -- guardrail G7. Whether the policy uses auction order is an
    empirical question measured by ablation A7, not a unit test.
    """
    actor, _critic = networks
    encoder = actor.trunk.history_encoder
    generator = np.random.default_rng(7)
    ids, _lengths = random_segments(1, [6], generator)
    reversed_ids = ids.clone()
    reversed_ids[0, :6] = ids[0, :6].flip(0)

    forward = encoder.advance(ids, torch.tensor([6]), zero_hidden_state(1, actor.spec))
    backward = encoder.advance(reversed_ids, torch.tensor([6]), zero_hidden_state(1, actor.spec))

    assert not torch.allclose(forward, backward, atol=1e-6)


# --- masking ----------------------------------------------------------------


def test_illegal_actions_receive_exactly_zero_probability(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    actor, _critic = networks
    generator = np.random.default_rng(8)
    ids, lengths = random_segments(BATCH, [0, 2, 4, 1, 0], generator)
    masks = legal_masks(BATCH, generator)
    logits, _advanced = actor(
        torch.randn((BATCH, D_STATIC)), ids, lengths, zero_hidden_state(BATCH, actor.spec)
    )

    probabilities = MaskedCategorical.from_logits(logits, masks).probabilities

    assert torch.equal(probabilities[~masks], torch.zeros_like(probabilities[~masks]))
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(BATCH), atol=1e-6)


def test_action_sixty_four_is_always_impossible(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    """Reserved no-trump. The engine never makes it legal, so it is never sampled."""
    actor, _critic = networks
    generator = np.random.default_rng(9)
    ids, lengths = random_segments(BATCH, [0, 0, 3, 0, 2], generator)
    masks = legal_masks(BATCH, generator)
    assert not bool(masks[:, NO_TRUMP_ACTION].any())

    with torch.no_grad():
        logits, _advanced = actor(
            torch.randn((BATCH, D_STATIC)), ids, lengths, zero_hidden_state(BATCH, actor.spec)
        )
    distribution = MaskedCategorical.from_logits(logits, masks)

    assert float(distribution.probabilities[:, NO_TRUMP_ACTION].max()) == 0.0
    sampling_generator = torch.Generator()
    sampling_generator.manual_seed(0)
    for _ in range(50):
        sampled = distribution.sample(sampling_generator)
        assert not bool((sampled == NO_TRUMP_ACTION).any())


def test_an_all_false_mask_is_refused(
    networks: tuple[RecurrentActorNetwork, RecurrentCriticNetwork],
) -> None:
    """A seat with no legal action must never be asked to act."""
    actor, _critic = networks
    generator = np.random.default_rng(10)
    ids, lengths = random_segments(1, [0], generator)
    logits, _advanced = actor(
        torch.zeros((1, D_STATIC)), ids, lengths, zero_hidden_state(1, actor.spec)
    )
    with pytest.raises(ValueError, match="no legal action"):
        MaskedCategorical.from_logits(logits, torch.zeros((1, NUM_ACTIONS), dtype=torch.bool))


# --- spec validation --------------------------------------------------------


def test_the_spec_rejects_an_unimplemented_reset_mode() -> None:
    with pytest.raises(ValueError, match="gru_reset"):
        RecurrentArchitectureSpec(gru_reset="match")


def test_the_spec_rejects_a_changed_action_count() -> None:
    with pytest.raises(ValueError, match="frozen at 65"):
        RecurrentArchitectureSpec(action_count=64)


def test_the_spec_round_trips_through_a_dictionary() -> None:
    spec = RecurrentArchitectureSpec()
    assert RecurrentArchitectureSpec.from_dict(spec.as_dict()) == spec
    assert spec.gru_reset == GRU_RESET_DEAL


def test_seeded_construction_is_reproducible_and_leaves_global_rng_alone() -> None:
    torch.manual_seed(1234)
    before = torch.random.get_rng_state()
    first_actor, _first_critic = build_recurrent_actor_and_critic(seed=99)
    assert torch.equal(before, torch.random.get_rng_state())
    second_actor, _second_critic = build_recurrent_actor_and_critic(seed=99)
    for left, right in zip(first_actor.parameters(), second_actor.parameters(), strict=True):
        assert torch.equal(left, right)
