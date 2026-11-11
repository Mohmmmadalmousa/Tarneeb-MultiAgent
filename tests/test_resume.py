"""Save, load and resume: an interrupted run must continue, not restart.

A round-trip that only restores weights is not a resume. If the environments
restart, the sampling stream rewinds, or the update counter resets, the resumed
run silently trains on a different data distribution with a different
learning-rate schedule -- and nothing raises.

The test here is the strong one: run N updates uninterrupted; separately run K
updates, checkpoint, load into a fresh process-equivalent set of objects, and
continue to N. The two must agree update for update on every PPO diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tarneeb.models.checkpoint import IncompatibleCheckpointError, load_checkpoint, save_checkpoint
from tarneeb.models.nets import build_actor_and_critic
from tarneeb.training.config import EvaluationSchedule, TrainingConfig
from tarneeb.training.ppo import PPOConfig, build_optimizer
from tarneeb.training.train import train

COMPARED_METRICS = (
    "update",
    "global_environment_steps",
    "policy_loss",
    "value_loss",
    "entropy",
    "approximate_kl_divergence",
    "clip_fraction",
    "explained_variance",
    "gradient_norm",
    "learning_rate",
    "completed_deals",
    "learner_decision_count",
    "mean_bid",
    "contract_success_rate",
)


def resume_config(
    checkpoint_directory: Path, total_updates: int, checkpoint_every: int
) -> TrainingConfig:
    """A small but fully representative run: real opponents, real deals."""
    rollout_length = 40
    environment_count = 4
    return TrainingConfig(
        run_name="resume",
        environment_count=environment_count,
        rollout_length=rollout_length,
        total_environment_steps=rollout_length * environment_count * total_updates,
        environment_root_seed=5,
        training_opponent="basic",
        checkpoint_directory=checkpoint_directory,
        checkpoint_every_updates=checkpoint_every,
        ppo=PPOConfig(update_epochs=2, minibatch_count=2),
        evaluation=EvaluationSchedule(
            every_updates=10_000, paired_deal_count=2, bootstrap_resamples=50
        ),
    )


def comparable(records: list[dict]) -> list[tuple]:
    return [tuple(record[name] for name in COMPARED_METRICS) for record in records]


def checkpoint_at_update(result, update: int) -> Path:
    """Return the checkpoint a run wrote after ``update`` updates."""
    for path in result.checkpoint_paths:
        if Path(path).stem.endswith(f"update{update}"):
            return Path(path)
    raise AssertionError(f"no checkpoint at update {update} in {result.checkpoint_paths}")


def test_a_resumed_run_reproduces_the_uninterrupted_run(tmp_path: Path) -> None:
    """The gate: resume from mid-run and land exactly where the run would have.

    The reference run is the uninterrupted one and also writes the checkpoint the
    resume starts from, so both share one configuration -- including the
    learning-rate annealing horizon. Comparing a differently-configured short run
    would test the schedule, not the resume.
    """
    reference = train(resume_config(tmp_path / "reference", total_updates=4, checkpoint_every=2))
    assert len(reference.update_records) == 4

    checkpoint_path = checkpoint_at_update(reference, 2)
    resumed = train(
        resume_config(tmp_path / "resumed", total_updates=4, checkpoint_every=99),
        resume_from=checkpoint_path,
    )

    assert resumed.resumed_from == str(checkpoint_path)
    assert resumed.resumed_at_update == 2
    assert len(resumed.update_records) == 2, "a resume must not repeat completed updates"
    assert comparable(reference.update_records[2:]) == comparable(resumed.update_records)


def test_a_resume_that_lost_its_state_would_be_caught(tmp_path: Path) -> None:
    """Negative control: restoring only the weights must NOT reproduce the run.

    Without this, a resume that quietly restarted the environments and the
    sampling stream could still pass the test above by coincidence.
    """
    reference = train(resume_config(tmp_path / "reference", total_updates=4, checkpoint_every=2))
    checkpoint_path = checkpoint_at_update(reference, 2)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    crippled = tmp_path / "weights_only.pt"
    payload["resume"]["action_sampling_generator_state"] = (
        torch.Generator().manual_seed(999).get_state()
    )
    torch.save(payload, crippled)

    resumed = train(
        resume_config(tmp_path / "resumed", total_updates=4, checkpoint_every=99),
        resume_from=crippled,
    )
    assert comparable(reference.update_records[2:]) != comparable(resumed.update_records)


def test_the_resumed_run_continues_the_step_count(tmp_path: Path) -> None:
    reference = train(resume_config(tmp_path / "reference", total_updates=4, checkpoint_every=2))
    resumed = train(
        resume_config(tmp_path / "resumed", total_updates=4, checkpoint_every=99),
        resume_from=checkpoint_at_update(reference, 2),
    )
    assert resumed.update_records[0]["global_environment_steps"] == 40 * 4 * 3
    assert resumed.total_environment_steps == 40 * 4 * 4


def test_the_learning_rate_schedule_continues_rather_than_restarting(tmp_path: Path) -> None:
    """A reset update counter would silently re-raise the annealed learning rate."""
    reference = train(resume_config(tmp_path / "reference", total_updates=4, checkpoint_every=2))
    resumed = train(
        resume_config(tmp_path / "resumed", total_updates=4, checkpoint_every=99),
        resume_from=checkpoint_at_update(reference, 2),
    )
    assert resumed.update_records[0]["learning_rate"] == pytest.approx(
        reference.update_records[2]["learning_rate"]
    )
    assert resumed.update_records[0]["learning_rate"] < reference.update_records[0]["learning_rate"]


# --- what exactly is restored ----------------------------------------------


def test_the_checkpoint_carries_every_piece_of_resume_state(tmp_path: Path) -> None:
    result = train(resume_config(tmp_path / "run", total_updates=2, checkpoint_every=2))
    payload = load_checkpoint(Path(result.checkpoint_paths[-1])).payload

    assert payload["optimizer_state_dict"] is not None
    assert payload["critic_state_dict"] is not None
    resume = payload["resume"]
    assert set(resume) == {
        "completed_updates",
        "total_environment_steps",
        "action_sampling_generator_state",
        "minibatch_shuffle_generator_state",
        "environment_states",
        "environment_deal_indices",
        "environment_match_seeds",
        "opponent_generator_states",
        "collector_redeal_counts",
    }
    assert resume["completed_updates"] == 2
    assert len(resume["environment_states"]) == 4
    assert len(resume["opponent_generator_states"]) == 4
    assert all(state is not None for state in resume["environment_states"])


def test_the_restored_environments_carry_a_deal_in_progress(tmp_path: Path) -> None:
    """Restarting the deals would be a different data distribution, not a resume."""
    result = train(resume_config(tmp_path / "run", total_updates=2, checkpoint_every=2))
    resume = load_checkpoint(Path(result.checkpoint_paths[-1])).payload["resume"]
    for state in resume["environment_states"]:
        assert state.events, "a mid-rollout environment should hold a deal in progress"


def test_resuming_a_checkpoint_without_resume_state_is_refused(tmp_path: Path) -> None:
    actor, critic = build_actor_and_critic(seed=1)
    path = save_checkpoint(
        path=tmp_path / "deploy_only.pt",
        actor=actor,
        critic=critic,
        optimizer=build_optimizer(actor, critic, PPOConfig()),
        rules=__import__("tarneeb.core.rules", fromlist=["RulesConfig"]).RulesConfig(),
        training_config={},
        global_environment_steps=0,
        completed_updates=0,
        wall_clock_seconds=0.0,
        seeds={},
    )
    with pytest.raises(ValueError, match="cannot be resumed"):
        train(
            resume_config(tmp_path / "run", total_updates=2, checkpoint_every=99), resume_from=path
        )


def test_resuming_into_a_different_environment_count_is_refused(tmp_path: Path) -> None:
    first_segment = train(resume_config(tmp_path / "segment", total_updates=2, checkpoint_every=2))
    wider = resume_config(tmp_path / "wider", total_updates=4, checkpoint_every=99).with_overrides(
        environment_count=8
    )
    with pytest.raises(ValueError, match="same environment count"):
        train(wider, resume_from=Path(first_segment.checkpoint_paths[-1]))


def test_resuming_an_incompatible_checkpoint_is_refused(tmp_path: Path) -> None:
    result = train(resume_config(tmp_path / "run", total_updates=2, checkpoint_every=2))
    path = Path(result.checkpoint_paths[-1])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["buffer_contract_version"] = "chunk_v2"
    torch.save(payload, path)

    with pytest.raises(IncompatibleCheckpointError, match="buffer_contract_version"):
        train(
            resume_config(tmp_path / "run2", total_updates=4, checkpoint_every=99), resume_from=path
        )
