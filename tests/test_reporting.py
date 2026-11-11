"""The observability boundary, enforced rather than asserted in prose.

project requirements requires that logging be optional and removable: disabling it must not
change learning behaviour, PPO and rollout code must not own a reporter, and
metrics leaving the training algorithm must be detached scalars.

The load-bearing test here is
:func:`test_logging_does_not_change_learning`: the same run with both sinks
attached and with none must agree on every PPO diagnostic, to full precision.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
import torch

from tarneeb.training import ppo as ppo_module
from tarneeb.training import rollout as rollout_module
from tarneeb.training.config import EvaluationSchedule, TrainingConfig
from tarneeb.training.ppo import PPOConfig
from tarneeb.training.reporting import (
    SCALAR_METRIC_NAMES,
    CompositeReporter,
    JsonlReporter,
    TensorBoardReporter,
    build_default_reporter,
    derived_safety_rates,
)
from tarneeb.training.train import train

COMPARED_METRICS = (
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
    "epoch_zero_log_probability_max_deviation",
)


def tiny_config(checkpoint_directory: Path) -> TrainingConfig:
    return TrainingConfig(
        run_name="reporting",
        environment_count=2,
        rollout_length=32,
        total_environment_steps=128,
        environment_root_seed=9,
        training_opponent="basic",
        checkpoint_directory=checkpoint_directory,
        checkpoint_every_updates=99,
        ppo=PPOConfig(update_epochs=1, minibatch_count=2),
        evaluation=EvaluationSchedule(
            every_updates=99, paired_deal_count=2, bootstrap_resamples=50
        ),
    )


# --- the boundary itself ----------------------------------------------------


def test_logging_does_not_change_learning(tmp_path: Path) -> None:
    """The gate: identical learning with both sinks attached and with none."""
    with build_default_reporter(tmp_path / "logged") as reporter:
        logged = train(tiny_config(tmp_path / "a"), report=reporter)
    silent = train(tiny_config(tmp_path / "b"), report=None, report_evaluation=None)

    assert len(logged.update_records) == len(silent.update_records)
    for left, right in zip(logged.update_records, silent.update_records, strict=True):
        for name in COMPARED_METRICS:
            assert left[name] == pytest.approx(right[name], rel=0, abs=0), name


def test_ppo_and_rollout_take_no_logger_or_verbose_argument() -> None:
    """The algorithm must not be able to own a reporter, even by accident."""
    forbidden = {"logger", "log", "writer", "reporter", "verbose", "tensorboard", "summary_writer"}
    for callable_object in (
        ppo_module.ppo_update,
        rollout_module.RolloutCollector.__init__,
        rollout_module.RolloutCollector.collect,
    ):
        parameters = set(inspect.signature(callable_object).parameters)
        assert not (parameters & forbidden), (callable_object, parameters & forbidden)


def test_neither_ppo_nor_rollout_imports_a_writer() -> None:
    """Checked on the import graph, not on prose that may mention reporting."""
    for module in (ppo_module, rollout_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any("tensorboard" in name for name in imported), (module, imported)
        assert not any(name.endswith("reporting") for name in imported), (module, imported)
        assert "SummaryWriter" not in source


def test_every_metric_leaving_an_update_is_a_plain_scalar(tmp_path: Path) -> None:
    result = train(tiny_config(tmp_path))
    for record in result.update_records:
        for key, value in record.items():
            assert not isinstance(value, torch.Tensor), key
            assert value is None or isinstance(value, int | float | str | bool), key


# --- JSONL ------------------------------------------------------------------


def test_jsonl_records_one_line_per_update_and_is_readable(tmp_path: Path) -> None:
    path = tmp_path / "updates.jsonl"
    with JsonlReporter(path) as reporter:
        result = train(tiny_config(tmp_path / "checkpoints"), report=reporter)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(result.update_records)
    records = [json.loads(line) for line in lines]
    assert [record["update"] for record in records] == [1, 2]
    for record in records:
        assert record["illegal_action_rate"] == 0.0
        assert record["no_trump_action_rate"] == 0.0
        for name in ("policy_loss", "value_loss", "entropy", "explained_variance"):
            assert isinstance(record[name], float)


def test_jsonl_is_flushed_as_the_run_proceeds(tmp_path: Path) -> None:
    """An interrupted run must still leave the records it had already produced."""
    path = tmp_path / "updates.jsonl"
    reporter = JsonlReporter(path)
    reporter({"update": 1, "policy_loss": -0.5, "learner_decision_count": 10})
    assert path.read_text(encoding="utf-8").count("\n") == 1
    reporter.close()


def test_the_safety_rates_are_derived_from_the_counts() -> None:
    record = {"learner_decision_count": 200, "illegal_action_count": 0, "no_trump_action_count": 0}
    enriched = derived_safety_rates(record)
    assert enriched["illegal_action_rate"] == 0.0
    assert enriched["no_trump_action_rate"] == 0.0

    dirty = {"learner_decision_count": 200, "illegal_action_count": 2, "no_trump_action_count": 1}
    enriched = derived_safety_rates(dirty)
    assert enriched["illegal_action_rate"] == pytest.approx(0.01)
    assert enriched["no_trump_action_rate"] == pytest.approx(0.005)


def test_deriving_rates_leaves_the_original_record_untouched() -> None:
    record = {"learner_decision_count": 10, "illegal_action_count": 0}
    derived_safety_rates(record)
    assert "illegal_action_rate" not in record


# --- TensorBoard ------------------------------------------------------------


def test_tensorboard_writes_an_event_file(tmp_path: Path) -> None:
    log_directory = tmp_path / "tensorboard"
    with TensorBoardReporter(log_directory) as reporter:
        train(tiny_config(tmp_path / "checkpoints"), report=reporter)
    event_files = list(log_directory.glob("events.out.tfevents.*"))
    assert event_files, "no TensorBoard event file was produced"
    assert event_files[0].stat().st_size > 0


def test_tensorboard_records_the_expected_scalars(tmp_path: Path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_directory = tmp_path / "tensorboard"
    with TensorBoardReporter(log_directory) as reporter:
        train(tiny_config(tmp_path / "checkpoints"), report=reporter)

    accumulator = EventAccumulator(str(log_directory))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])

    for required in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approximate_kl_divergence",
        "clip_fraction",
        "gradient_norm",
        "explained_variance",
        "learning_rate",
        "illegal_action_rate",
        "no_trump_action_rate",
        "epoch_zero_log_probability_max_deviation",
        "environment_steps_per_second",
    ):
        assert f"train/{required}" in tags, required

    illegal = accumulator.Scalars("train/illegal_action_rate")
    assert all(point.value == 0.0 for point in illegal)


def test_tensorboard_skips_a_metric_that_has_no_value(tmp_path: Path) -> None:
    """``None`` means "no deal finished", which must not be plotted as a zero."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_directory = tmp_path / "tensorboard"
    with TensorBoardReporter(log_directory) as reporter:
        reporter(
            {
                "update": 1,
                "global_environment_steps": 10,
                "learner_decision_count": 5,
                "policy_loss": -0.1,
                "mean_bid": None,
            }
        )
    accumulator = EventAccumulator(str(log_directory))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert "train/policy_loss" in tags
    assert "train/mean_bid" not in tags


def test_every_advertised_scalar_is_a_real_metric_name(tmp_path: Path) -> None:
    """No decorative metrics: each plotted name must be produced by the run."""
    result = train(tiny_config(tmp_path))
    produced = set(result.update_records[0]) | {"illegal_action_rate", "no_trump_action_rate"}
    assert set(SCALAR_METRIC_NAMES) <= produced


def test_evaluation_scalars_are_recorded_separately(tmp_path: Path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_directory = tmp_path / "tensorboard"
    with TensorBoardReporter(log_directory) as reporter:
        reporter.add_evaluation(
            {
                "global_environment_steps": 100,
                "opponent": "random",
                "mean_paired_score_difference": 1.5,
                "win_rate": 0.8,
                "paired_score_difference_ci": {"lower_bound": 1.0, "upper_bound": 2.0},
                "illegal_action_rate": 0.0,
            }
        )
    accumulator = EventAccumulator(str(log_directory))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert "eval/random/mean_paired_score_difference" in tags
    assert "eval/random/win_rate" in tags


# --- composition ------------------------------------------------------------


def test_a_composite_reporter_fans_out_to_every_sink(tmp_path: Path) -> None:
    seen: list[dict] = []
    composite = CompositeReporter([seen.append, JsonlReporter(tmp_path / "a.jsonl")])
    composite({"update": 1, "learner_decision_count": 4})
    composite.close()

    assert len(seen) == 1
    assert (tmp_path / "a.jsonl").read_text(encoding="utf-8").strip()


def test_the_default_reporter_produces_both_sinks(tmp_path: Path) -> None:
    with build_default_reporter(tmp_path) as reporter:
        reporter({"update": 1, "global_environment_steps": 1, "learner_decision_count": 1})
    assert (tmp_path / "updates.jsonl").exists()
    assert list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))


def test_the_default_reporter_can_omit_tensorboard(tmp_path: Path) -> None:
    with build_default_reporter(tmp_path, tensorboard=False) as reporter:
        reporter({"update": 1, "global_environment_steps": 1, "learner_decision_count": 1})
    assert (tmp_path / "updates.jsonl").exists()
    assert not (tmp_path / "tensorboard").exists()
