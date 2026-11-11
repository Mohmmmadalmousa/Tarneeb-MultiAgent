"""League-aware recurrent rollout: G5, G6, masking, and policy grouping."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarneeb.core.actions import PASS_ACTION
from tarneeb.core.rules import RulesConfig
from tarneeb.env.vec import VecTarneeb
from tarneeb.league.cache import HistoricalActorCache
from tarneeb.league.registry import PolicyRegistry
from tarneeb.league.sampling import (
    LatestOnlyProvider,
    PFSPConfig,
    PFSPPerformance,
    PFSPProvider,
)
from tarneeb.models.recurrent import (
    RecurrentActorNetwork,
    RecurrentArchitectureSpec,
    RecurrentCriticNetwork,
    build_recurrent_actor_and_critic,
    zero_hidden_state,
)
from tarneeb.models.recurrent_checkpoint import save_recurrent_checkpoint
from tarneeb.training.league_rollout import LeagueRecurrentCollector, NeuralRequest
from tarneeb.training.recurrent_rollout import RecurrentRolloutMetrics, relativized_event_ids
from tarneeb.training.recurrent_state import LEARNER_POLICY_ID

HIDDEN_TOLERANCE = 1e-5
LEARNER_SPEC = RecurrentArchitectureSpec(
    critic_observation_dim=856,
    hidden_size=32,
    static_hidden_dim=64,
)
HISTORICAL_SPEC = RecurrentArchitectureSpec(
    critic_observation_dim=692,
    hidden_size=32,
    static_hidden_dim=64,
)


class SequenceOpponentProvider:
    """Deterministic test provider that changes only on each sample call."""

    def __init__(self, policy_ids: tuple[str, ...]) -> None:
        self.policy_ids = policy_ids
        self.sample_count = 0

    def sample(self) -> str:
        policy_id = self.policy_ids[self.sample_count % len(self.policy_ids)]
        self.sample_count += 1
        return policy_id

    def probabilities(self) -> dict[str, float]:
        return {policy_id: 1.0 / len(self.policy_ids) for policy_id in self.policy_ids}


class RecordingLeagueCollector(LeagueRecurrentCollector):
    """Collector that records outputs for the batched/reference comparison."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.actor_output_history: dict[tuple[str, NeuralRequest], list[torch.Tensor]] = {}
        self.actor_hidden_history: dict[tuple[str, NeuralRequest], list[torch.Tensor]] = {}

    def _actor_decision_outputs(
        self,
        policy_id: str,
        actor: RecurrentActorNetwork,
        requests: list[NeuralRequest],
        observations: dict[NeuralRequest, np.ndarray],
    ) -> dict[NeuralRequest, torch.Tensor]:
        outputs = super()._actor_decision_outputs(policy_id, actor, requests, observations)
        for request, logits in outputs.items():
            self.actor_output_history.setdefault((policy_id, request), []).append(logits.clone())
        return outputs

    def _advance_actor_group(
        self,
        policy_id: str,
        actor: RecurrentActorNetwork,
        environment_indices: np.ndarray,
        seats: np.ndarray,
        event_ids: torch.Tensor,
        event_lengths: torch.Tensor,
    ) -> None:
        super()._advance_actor_group(
            policy_id, actor, environment_indices, seats, event_ids, event_lengths
        )
        for environment_index, seat in zip(environment_indices, seats, strict=True):
            request = (int(environment_index), int(seat))
            hidden_state = self.state.actor_slot(*request, policy_id=policy_id).detach().clone()
            self.actor_hidden_history.setdefault((policy_id, request), []).append(hidden_state)


class PerRequestRecordingReferenceCollector(RecordingLeagueCollector):
    """Unbatched actor/event path with identical seating and sampling order."""

    def _actor_decision_outputs(
        self,
        policy_id: str,
        actor: RecurrentActorNetwork,
        requests: list[NeuralRequest],
        observations: dict[NeuralRequest, np.ndarray],
    ) -> dict[NeuralRequest, torch.Tensor]:
        outputs: dict[NeuralRequest, torch.Tensor] = {}
        for request in requests:
            outputs.update(
                RecordingLeagueCollector._actor_decision_outputs(
                    self, policy_id, actor, [request], observations
                )
            )
        return outputs

    def _learner_critic_values(self, requests: list[NeuralRequest]) -> dict[NeuralRequest, float]:
        values: dict[NeuralRequest, float] = {}
        for request in requests:
            values.update(super()._learner_critic_values([request]))
        return values

    def _advance_actor_group(
        self,
        policy_id: str,
        actor: RecurrentActorNetwork,
        environment_indices: np.ndarray,
        seats: np.ndarray,
        event_ids: torch.Tensor,
        event_lengths: torch.Tensor,
    ) -> None:
        for row in range(environment_indices.size):
            super()._advance_actor_group(
                policy_id,
                actor,
                environment_indices[row : row + 1],
                seats[row : row + 1],
                event_ids[row : row + 1],
                event_lengths[row : row + 1],
            )


@pytest.fixture(scope="module")
def policy_registry(tmp_path_factory: pytest.TempPathFactory) -> PolicyRegistry:
    checkpoint_directory = tmp_path_factory.mktemp("league-rollout-checkpoints")
    registry = PolicyRegistry()
    for seed, policy_id, category in (
        (101, "historical_a", "champion"),
        (102, "historical_b", "snapshot"),
        (103, "historical_c", "snapshot"),
        (104, "always_pass", "snapshot"),
    ):
        actor, critic = build_recurrent_actor_and_critic(HISTORICAL_SPEC, seed=seed)
        with torch.no_grad():
            actor.policy_head.bias[PASS_ACTION] = 10.0 if policy_id == "always_pass" else 0.0
            actor.policy_head.bias[52] = 5.0
        checkpoint_path = checkpoint_directory / f"{policy_id}.pt"
        save_recurrent_checkpoint(
            checkpoint_path,
            actor,
            critic,
            None,
            RulesConfig(),
            {"source_phase": "phase11_test"},
            0,
            0,
            0.0,
            {"seed_index": seed},
        )
        registry.register_checkpoint(
            policy_id,
            checkpoint_path,
            source_phase="phase11_test",
            category=category,
            seed=seed,
            update=0,
        )
    registry.register_anchor("strong", "strong")
    registry.register_anchor("random", "random")
    return registry


def build_pfsp_provider(registry: PolicyRegistry, seed: int = 17) -> PFSPProvider:
    return PFSPProvider(
        policies_by_category={
            "champion": ("historical_a",),
            "recent": ("historical_b",),
            "older": ("historical_c",),
            "strong": ("strong",),
            "random": ("random",),
        },
        performance={
            "historical_a": PFSPPerformance(0.2, 500),
            "historical_b": PFSPPerformance(0.5, 500),
            "historical_c": PFSPPerformance(0.8, 500),
        },
        rng=np.random.default_rng(seed),
        config=PFSPConfig(
            ema_beta=0.2,
            unobserved_learner_win_rate=0.5,
        ),
    )


def build_collector(
    policy_registry: PolicyRegistry,
    opponent_provider: object,
    *,
    collector_class: type[LeagueRecurrentCollector] = LeagueRecurrentCollector,
    environment_count: int = 4,
    rollout_length: int = 4,
    seed: int = 51,
    learner_actor: RecurrentActorNetwork | None = None,
    learner_critic: RecurrentCriticNetwork | None = None,
) -> LeagueRecurrentCollector:
    environments = VecTarneeb(num_envs=environment_count, root_seed=seed)
    if learner_actor is None or learner_critic is None:
        built_actor, built_critic = build_recurrent_actor_and_critic(LEARNER_SPEC, seed=77)
        with torch.no_grad():
            built_actor.policy_head.bias[52] = 5.0
        learner_actor = built_actor
        learner_critic = built_critic
    action_sampling_generator = torch.Generator()
    action_sampling_generator.manual_seed(seed + 1)
    return collector_class(
        environments=environments,
        actor=learner_actor,
        critic=learner_critic,
        rollout_length=rollout_length,
        action_sampling_generator=action_sampling_generator,
        opponent_provider=opponent_provider,  # type: ignore[arg-type]
        actor_cache=HistoricalActorCache(policy_registry),
        opponent_seed=seed + 2,
    )


def test_seating_assigns_exactly_one_coherent_opposing_policy(
    policy_registry: PolicyRegistry,
) -> None:
    collector = build_collector(policy_registry, LatestOnlyProvider("historical_a"))
    buffer, metrics = collector.collect()
    del metrics

    assignments = {
        (environment_index, generation): policy_id
        for environment_index, generation, policy_id in collector.policy_assignment_history
    }
    assert assignments
    covered_learner_seats: set[int] = set()
    for environment_index in range(collector.environments.num_envs):
        learner_seats = collector.learner_seats(environment_index)
        assert len(learner_seats) == 2
        assert len({seat % 2 for seat in learner_seats}) == 1
        covered_learner_seats.update(learner_seats)
    assert covered_learner_seats == {0, 1, 2, 3}

    for environment_index, generation, seat, policy_id in collector.decision_policy_history:
        if collector.is_learner_seat(environment_index, seat):
            assert policy_id == LEARNER_POLICY_ID
        else:
            assert policy_id == assignments[(environment_index, generation)]
    for environment_index, seat in buffer.sequences:
        assert collector.is_learner_seat(environment_index, seat)
    for (
        _environment_index,
        _seat,
        _policy_id,
        action,
        legal_action_mask,
    ) in collector.action_history:
        assert bool(legal_action_mask[action])
        assert action != 64


def test_latest_only_and_pfsp_use_the_same_collector_path(
    policy_registry: PolicyRegistry,
) -> None:
    latest = build_collector(policy_registry, LatestOnlyProvider("historical_a"))
    latest_buffer, latest_metrics = latest.collect()
    league = build_collector(policy_registry, build_pfsp_provider(policy_registry))
    league_buffer, league_metrics = league.collect()

    assert isinstance(latest, LeagueRecurrentCollector)
    assert isinstance(league, LeagueRecurrentCollector)
    assert latest.no_unfinished_learner_sequence(latest_buffer)
    assert league.no_unfinished_learner_sequence(league_buffer)
    assert latest_metrics.illegal_action_count == league_metrics.illegal_action_count == 0
    assert latest_metrics.no_trump_action_count == league_metrics.no_trump_action_count == 0
    assert set(latest.sampled_opponent_policy_ids) == {"historical_a"}
    assert set(league.sampled_opponent_policy_ids) <= {
        "historical_a",
        "historical_b",
        "historical_c",
        "strong",
        "random",
    }


def test_policy_assignment_cannot_change_with_live_recurrent_state(
    policy_registry: PolicyRegistry,
) -> None:
    provider = SequenceOpponentProvider(("historical_a", "historical_b"))
    collector = build_collector(policy_registry, provider, environment_count=1)

    collector._assign_opponent_policy(0)
    assert collector.opponent_policy_ids[0] == "historical_a"
    collector.state.actor_hidden["historical_a"][0, 1] = 1.0
    with pytest.raises(AssertionError, match="cannot switch while recurrent state is live"):
        collector._assign_opponent_policy(0)

    collector.state.reset_environment(0)
    collector._assign_opponent_policy(0)
    assert collector.opponent_policy_ids[0] == "historical_b"
    assert not collector.state.actor_hidden["historical_a"][0].any()
    assert not collector.state.actor_hidden["historical_b"][0].any()


def test_registered_policy_refresh_is_rejected_with_live_recurrent_state(
    policy_registry: PolicyRegistry,
) -> None:
    collector = build_collector(
        policy_registry, LatestOnlyProvider("historical_a"), environment_count=1
    )
    collector.state.actor_hidden["learner"][0, 0] = 1.0
    with pytest.raises(AssertionError, match="zero-state boundary"):
        collector.refresh_registered_policies()

    collector.state.reset_all()
    collector.refresh_registered_policies()
    assert collector.recurrent_state_is_zero()


def test_state_and_event_advancement_follow_each_seat_policy(
    policy_registry: PolicyRegistry,
) -> None:
    collector = build_collector(
        policy_registry,
        SequenceOpponentProvider(("historical_a", "historical_b")),
        environment_count=2,
    )
    collector._assign_opponent_policy(0)
    collector._assign_opponent_policy(1)
    actions = np.array(
        [
            int(np.flatnonzero(environment.legal_action_mask())[0])
            for environment in collector.environments.environments
        ],
        dtype=np.int64,
    )
    vector_step = collector.environments.step(actions)
    assert all(vector_step.events)
    metrics = RecurrentRolloutMetrics()
    collector._advance_policy_states(vector_step.events, metrics)

    for environment_index, events in enumerate(vector_step.events):
        for seat in range(4):
            policy_id = collector.policy_id_for_seat(environment_index, seat)
            actor = (
                collector.actor
                if policy_id == collector.policy_id
                else collector.actor_cache.get(policy_id)
            )
            event_ids = np.stack([relativized_event_ids(events, seat)]).astype(np.int64)
            padded = torch.zeros((1, actor.spec.max_segment_length, event_ids.shape[-1]))
            padded[0, : len(events)] = torch.from_numpy(event_ids[0])
            expected = actor.trunk.history_encoder.advance(
                padded,
                torch.tensor([len(events)], dtype=torch.int64),
                zero_hidden_state(1, actor.spec),
            )
            actual = collector.state.actor_hidden[policy_id][environment_index, seat].unsqueeze(1)
            assert torch.allclose(actual, expected, atol=HIDDEN_TOLERANCE)
            assert collector.state.last_seq_consumed[environment_index, seat] == events[-1].seq

    assert (
        collector.state.actor_hidden["historical_a"].data_ptr()
        != collector.state.actor_hidden["historical_b"].data_ptr()
    )
    assert not collector.state.critic_hidden["historical_a"].any()
    assert not collector.state.critic_hidden["historical_b"].any()
    assert all(
        not parameter.requires_grad
        for actor in collector._historical_actors.values()
        for parameter in actor.parameters()
    )


def test_batched_multi_policy_collection_matches_per_request_reference(
    policy_registry: PolicyRegistry,
) -> None:
    provider_b = LatestOnlyProvider("historical_a")
    reference = build_collector(
        policy_registry,
        provider_b,
        collector_class=PerRequestRecordingReferenceCollector,
        environment_count=4,
        rollout_length=12,
        seed=118,
    )
    batched = build_collector(
        policy_registry,
        LatestOnlyProvider("historical_a"),
        collector_class=RecordingLeagueCollector,
        environment_count=4,
        rollout_length=12,
        seed=118,
    )
    batched_buffer, batched_metrics = batched.collect()
    reference_buffer, reference_metrics = reference.collect()

    assert batched.policy_assignment_history == reference.policy_assignment_history
    assert batched.decision_policy_history == reference.decision_policy_history
    assert batched.actor_output_history.keys() == reference.actor_output_history.keys()
    for key, logits in batched.actor_output_history.items():
        assert len(logits) == len(reference.actor_output_history[key])
        for left, right in zip(logits, reference.actor_output_history[key], strict=True):
            assert torch.allclose(left, right, atol=HIDDEN_TOLERANCE)
    assert batched.actor_hidden_history.keys() == reference.actor_hidden_history.keys()
    for key, hidden_states in batched.actor_hidden_history.items():
        assert len(hidden_states) == len(reference.actor_hidden_history[key])
        for left, right in zip(hidden_states, reference.actor_hidden_history[key], strict=True):
            assert torch.allclose(left, right, atol=HIDDEN_TOLERANCE)
    assert len(batched.action_history) == len(reference.action_history)
    for left, right in zip(batched.action_history, reference.action_history, strict=True):
        assert left[:4] == right[:4]
        assert np.array_equal(left[4], right[4])
    assert batched_buffer.sequences.keys() == reference_buffer.sequences.keys()
    for key, sequence in batched_buffer.sequences.items():
        reference_sequence = reference_buffer.sequences[key]
        assert len(sequence) == len(reference_sequence)
        for left, right in zip(sequence, reference_sequence, strict=True):
            assert left.action == right.action
            assert np.array_equal(left.legal_action_mask, right.legal_action_mask)
            assert np.array_equal(left.static, right.static)
            assert left.old_log_probability == pytest.approx(
                right.old_log_probability, abs=HIDDEN_TOLERANCE
            )
            assert left.value == pytest.approx(right.value, abs=HIDDEN_TOLERANCE)
            assert left.first_event_seq == right.first_event_seq
            assert left.previous_event_length == right.previous_event_length
            assert np.array_equal(left.previous_event_ids, right.previous_event_ids)
            if left.is_chunk_head:
                assert np.allclose(
                    left.actor_hidden_head, right.actor_hidden_head, atol=HIDDEN_TOLERANCE
                )
                assert np.allclose(
                    left.critic_hidden_head, right.critic_hidden_head, atol=HIDDEN_TOLERANCE
                )
    assert batched_metrics.completed_deals == reference_metrics.completed_deals
    assert batched_metrics.passed_out_deals == reference_metrics.passed_out_deals
    assert batched_metrics.illegal_action_count == reference_metrics.illegal_action_count == 0
    assert batched_metrics.no_trump_action_count == reference_metrics.no_trump_action_count == 0
    assert batched.no_unfinished_learner_sequence(batched_buffer)
    assert batched.recurrent_state_is_zero()
    assert batched.last_collection_statistics["neural_policy_group_count"] > 0


def test_group_statistics_show_multi_request_actor_batches(
    policy_registry: PolicyRegistry,
) -> None:
    collector = build_collector(
        policy_registry,
        LatestOnlyProvider("historical_a"),
        environment_count=4,
        rollout_length=8,
        seed=211,
    )
    collector.collect()

    assert collector.neural_policy_group_count > 0
    assert any(
        request_count > 1
        for policy_groups in collector.neural_policy_group_sizes
        for _policy_id, request_count in policy_groups
    )


def test_seeded_pfsp_rollout_samples_multiple_historical_neural_policies(
    policy_registry: PolicyRegistry,
) -> None:
    collector = build_collector(
        policy_registry,
        build_pfsp_provider(policy_registry, seed=723),
        environment_count=8,
        rollout_length=1,
        seed=717,
    )
    collector.collect()
    sampled_historical_policy_ids = {
        policy_id
        for policy_id in collector.sampled_opponent_policy_ids
        if policy_id.startswith("historical_")
    }
    assert len(sampled_historical_policy_ids) >= 2
    assert any(
        len({policy_id for policy_id, _request_count in policy_groups}) >= 2
        for policy_groups in collector.neural_policy_group_sizes
    )


def test_heuristic_opponents_use_existing_agents_not_recurrent_actors(
    policy_registry: PolicyRegistry,
) -> None:
    collector = build_collector(
        policy_registry, LatestOnlyProvider("strong"), environment_count=2, rollout_length=2
    )
    buffer, metrics = collector.collect()

    assert collector.sampled_opponent_policy_ids
    assert set(collector.sampled_opponent_policy_ids) == {"strong"}
    assert collector.opponent_neural_decision_count == 0
    assert collector.cache_load_count == 0
    assert metrics.completed_deals > 0
    assert all(
        collector.is_learner_seat(environment_index, seat)
        for environment_index, seat in buffer.sequences
    )


def test_latest_current_learner_actor_can_control_both_partnerships(
    policy_registry: PolicyRegistry,
) -> None:
    collector = build_collector(
        policy_registry, LatestOnlyProvider(LEARNER_POLICY_ID), environment_count=2
    )
    buffer, _metrics = collector.collect()
    assert set(collector.sampled_opponent_policy_ids) == {LEARNER_POLICY_ID}
    assert any(
        not collector.is_learner_seat(environment_index, seat)
        for environment_index, _generation, seat, _policy_id in collector.decision_policy_history
    )
    assert all(
        collector.is_learner_seat(environment_index, seat)
        for environment_index, seat in buffer.sequences
    )


def test_all_pass_redeal_resets_policy_state_before_new_assignment(
    policy_registry: PolicyRegistry,
) -> None:
    learner_actor, learner_critic = build_recurrent_actor_and_critic(LEARNER_SPEC, seed=909)
    with torch.no_grad():
        learner_actor.policy_head.bias[PASS_ACTION] = 10.0
    provider = SequenceOpponentProvider(("always_pass", "strong"))
    collector = build_collector(
        policy_registry,
        provider,
        environment_count=1,
        rollout_length=1,
        seed=911,
        learner_actor=learner_actor,
        learner_critic=learner_critic,
    )
    buffer, metrics = collector.collect()

    assert metrics.passed_out_deals >= 1
    assert metrics.completed_deals >= 1
    assert (
        sum(transition.done for sequence in buffer.sequences.values() for transition in sequence)
        == 2 * metrics.completed_deals
    )
    assert collector.policy_assignment_history[0][2] == "always_pass"
    assert any(
        policy_id == "strong"
        for _environment_index, _generation, policy_id in collector.policy_assignment_history
    )
    assert collector.no_unfinished_learner_sequence(buffer)
    assert collector.recurrent_state_is_zero()
    assert collector.last_collection_statistics["non_finite_count"] == 0


def test_collection_statistics_report_cache_and_g6_state(
    policy_registry: PolicyRegistry,
) -> None:
    collector = build_collector(
        policy_registry, LatestOnlyProvider("historical_a"), environment_count=2, rollout_length=2
    )
    buffer, metrics = collector.collect()
    statistics = collector.last_collection_statistics

    assert statistics["completed_deals"] == metrics.completed_deals
    assert statistics["opponent_neural_decision_count"] > 0
    assert statistics["cache_load_count"] == 1
    assert statistics["cache_hit_count"] >= 1
    assert statistics["decisions_per_second"] > 0.0
    assert statistics["no_unfinished_learner_sequence"]
    assert statistics["recurrent_state_is_zero"]
    assert statistics["illegal_action_count"] == 0
    assert statistics["no_trump_action_count"] == 0
    assert all(
        transition.legal_action_mask[transition.action]
        for sequence in buffer.sequences.values()
        for transition in sequence
    )
