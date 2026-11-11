"""Shapes, parameter counts and the history encoder of the Phase 6 networks.

Expected parameter counts are derived by hand from the layer table, not read
back from the implementation, so an accidental architecture edit fails here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarneeb.core.actions import NUM_ACTIONS
from tarneeb.env.event_features import EVENT_FIELD_CARDINALITIES, expand_event_id_rows
from tarneeb.env.spaces import D_CRITIC, D_EVENT_ONEHOT, D_STATIC, L_MAX, N_EVENT_FIELDS
from tarneeb.models.nets import (
    ARCHITECTURE_SPEC_VERSION,
    ActorNetwork,
    ArchitectureSpec,
    CriticNetwork,
    PaddedFeedforwardHistoryEncoder,
    build_actor_and_critic,
    count_parameters,
    parameters_are_disjoint,
)

# --- hand-derived parameter counts ------------------------------------------
#
#   static encoder : 692*512 + 512  +  2*512  +  512*256 + 256  +  2*256
#   history        :  99*16  + 16   +  (80*16)*256 + 256
#   fusion         : 512*256 + 256  +  256*256 + 256
#   policy head    : 256*65  + 65
#   value head     : 256*1   + 1
#
# The static and history subtotals below match blueprint section 6.3 exactly for
# the feedforward variant.
STATIC_ENCODER_PARAMETERS_692 = 692 * 512 + 512 + 2 * 512 + 512 * 256 + 256 + 2 * 256
FEEDFORWARD_HISTORY_PARAMETERS = 99 * 16 + 16 + (80 * 16) * 256 + 256
FUSION_PARAMETERS = 512 * 256 + 256 + 256 * 256 + 256
POLICY_HEAD_PARAMETERS = 256 * NUM_ACTIONS + NUM_ACTIONS
VALUE_HEAD_PARAMETERS = 256 * 1 + 1

EXPECTED_ACTOR_PARAMETERS = (
    STATIC_ENCODER_PARAMETERS_692
    + FEEDFORWARD_HISTORY_PARAMETERS
    + FUSION_PARAMETERS
    + POLICY_HEAD_PARAMETERS
)
EXPECTED_CRITIC_PARAMETERS = (
    STATIC_ENCODER_PARAMETERS_692
    + FEEDFORWARD_HISTORY_PARAMETERS
    + FUSION_PARAMETERS
    + VALUE_HEAD_PARAMETERS
)


def random_history(
    batch_size: int, rng: np.random.Generator, valid_events: int = 12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a batch of legal padded histories with ``valid_events`` real rows."""
    ids = np.zeros((batch_size, L_MAX, N_EVENT_FIELDS), dtype=np.int64)
    masks = np.zeros((batch_size, L_MAX), dtype=bool)
    for row in range(batch_size):
        for position in range(valid_events):
            ids[row, position] = [
                rng.integers(0, cardinality) for cardinality in EVENT_FIELD_CARDINALITIES
            ]
        masks[row, :valid_events] = True
    return torch.from_numpy(ids), torch.from_numpy(masks)


def test_architecture_defaults_match_the_frozen_contract() -> None:
    spec = ArchitectureSpec()
    assert spec.actor_observation_dim == D_STATIC == 692
    assert spec.action_count == NUM_ACTIONS == 65
    assert spec.history_length == L_MAX == 80
    assert spec.event_field_count == N_EVENT_FIELDS == 7
    assert spec.event_onehot_dim == D_EVENT_ONEHOT == 99
    assert ARCHITECTURE_SPEC_VERSION == "ff_actor_critic_v1"


def test_phase_six_critic_is_decentralized_not_centralized() -> None:
    """The Phase 6 critic reads the actor-safe observation, not the 856-dim one."""
    spec = ArchitectureSpec()
    assert spec.critic_observation_dim == D_STATIC
    assert spec.critic_observation_dim != D_CRITIC


def test_recurrent_history_encoder_is_rejected() -> None:
    with pytest.raises(ValueError, match="history"):
        ArchitectureSpec(history_encoder="gru")


def test_parameter_counts_match_the_hand_derived_totals() -> None:
    actor, critic = build_actor_and_critic(seed=0)
    assert count_parameters(actor) == EXPECTED_ACTOR_PARAMETERS == 1_031_041
    assert count_parameters(critic) == EXPECTED_CRITIC_PARAMETERS == 1_014_593


def test_actor_and_critic_share_no_parameter() -> None:
    actor, critic = build_actor_and_critic(seed=0)
    assert parameters_are_disjoint(actor, critic)
    assert parameters_are_disjoint(critic, actor)


def test_building_from_a_seed_is_reproducible_and_leaves_the_global_rng_alone() -> None:
    torch.manual_seed(999)
    state_before = torch.random.get_rng_state()

    first_actor, _ = build_actor_and_critic(seed=42)
    second_actor, _ = build_actor_and_critic(seed=42)
    different_actor, _ = build_actor_and_critic(seed=43)

    for first, second in zip(first_actor.parameters(), second_actor.parameters(), strict=True):
        assert torch.equal(first, second)
    assert any(
        not torch.equal(first, other)
        for first, other in zip(first_actor.parameters(), different_actor.parameters(), strict=True)
    )
    assert torch.equal(state_before, torch.random.get_rng_state())


@pytest.mark.parametrize("batch_size", [1, 3, 17])
def test_forward_shapes_and_finiteness(batch_size: int) -> None:
    rng = np.random.default_rng(batch_size)
    actor, critic = build_actor_and_critic(seed=1)
    observations = torch.from_numpy(rng.random((batch_size, D_STATIC), dtype=np.float32))
    history_ids, history_masks = random_history(batch_size, rng)

    logits = actor(observations, history_ids, history_masks)
    values = critic(observations, history_ids, history_masks)

    assert logits.shape == (batch_size, NUM_ACTIONS)
    assert values.shape == (batch_size,)
    assert bool(torch.isfinite(logits).all())
    assert bool(torch.isfinite(values).all())


def test_actor_rejects_an_observation_of_the_wrong_width() -> None:
    """Handing the actor the 856-dim critic observation must fail loudly."""
    rng = np.random.default_rng(5)
    actor, _ = build_actor_and_critic(seed=1)
    critic_observations = torch.from_numpy(rng.random((2, D_CRITIC), dtype=np.float32))
    history_ids, history_masks = random_history(2, rng)
    with pytest.raises((ValueError, RuntimeError)):
        actor(critic_observations, history_ids, history_masks)


def test_on_device_one_hot_matches_the_numpy_event_encoder() -> None:
    """The network's expansion and the buffer's encoder must agree exactly."""
    rng = np.random.default_rng(7)
    encoder = PaddedFeedforwardHistoryEncoder(ArchitectureSpec())
    history_ids, _ = random_history(4, rng, valid_events=L_MAX)

    expanded = encoder.expand_history_ids(history_ids).numpy()
    for row in range(history_ids.shape[0]):
        expected = expand_event_id_rows(history_ids[row].numpy().astype(np.int16))
        assert np.array_equal(expanded[row], expected)


def test_one_hot_rejects_an_out_of_range_field_id() -> None:
    encoder = PaddedFeedforwardHistoryEncoder(ArchitectureSpec())
    ids = torch.zeros((1, L_MAX, N_EVENT_FIELDS), dtype=torch.long)
    ids[0, 0, 0] = EVENT_FIELD_CARDINALITIES[0]
    with pytest.raises(ValueError, match="out of range"):
        encoder.expand_history_ids(ids)


def test_padded_rows_contribute_exactly_nothing() -> None:
    """Changing a masked-out row must not move the encoder's output at all.

    This is what the "zero after the projection" rule buys: the projection bias
    would otherwise make every padded row contribute a constant.
    """
    rng = np.random.default_rng(11)
    encoder = PaddedFeedforwardHistoryEncoder(ArchitectureSpec())
    history_ids, history_masks = random_history(2, rng, valid_events=9)

    baseline = encoder(history_ids, history_masks)

    polluted_ids = history_ids.clone()
    polluted_ids[:, 9:, :] = torch.from_numpy(
        rng.integers(
            0,
            np.array(EVENT_FIELD_CARDINALITIES),
            size=(2, L_MAX - 9, N_EVENT_FIELDS),
        )
    )
    polluted = encoder(polluted_ids, history_masks)

    assert torch.equal(baseline, polluted)


def test_history_encoder_rejects_a_mismatched_mask() -> None:
    rng = np.random.default_rng(13)
    encoder = PaddedFeedforwardHistoryEncoder(ArchitectureSpec())
    history_ids, _ = random_history(2, rng)
    with pytest.raises(ValueError, match="history_masks"):
        encoder(history_ids, torch.ones((2, L_MAX - 1), dtype=torch.bool))


def test_actor_network_is_deterministic_for_identical_inputs() -> None:
    rng = np.random.default_rng(17)
    actor = ActorNetwork()
    actor.eval()
    observations = torch.from_numpy(rng.random((5, D_STATIC), dtype=np.float32))
    history_ids, history_masks = random_history(5, rng)
    with torch.no_grad():
        first = actor(observations, history_ids, history_masks)
        second = actor(observations, history_ids, history_masks)
    assert torch.equal(first, second)


def test_critic_can_be_widened_without_touching_the_module() -> None:
    """A later centralized critic is a configuration change, not a rewrite."""
    spec = ArchitectureSpec(critic_observation_dim=D_CRITIC)
    critic = CriticNetwork(spec)
    expected = (
        (D_CRITIC * 512 + 512 + 2 * 512 + 512 * 256 + 256 + 2 * 256)
        + FEEDFORWARD_HISTORY_PARAMETERS
        + FUSION_PARAMETERS
        + VALUE_HEAD_PARAMETERS
    )
    assert count_parameters(critic) == expected
