"""Optional metric sinks: JSONL on disk and TensorBoard scalars.

This module is the *only* place in the training stack that writes anything, and
nothing in it is required for training to run. That is the observability
boundary recorded in project requirements, stated concretely:

- PPO and rollout code compute metrics and return them as values. They own no
  writer, no reporter and no renderer, and take no ``logger`` or ``verbose``
  argument.
- :func:`~tarneeb.training.train.train` accepts an optional ``report`` callable.
  Passing ``None`` disables every sink here, and a run with logging disabled
  produces bit-identical learning -- asserted by
  ``tests/test_reporting.py::test_logging_does_not_change_learning``.
- Every value that arrives here is already a detached Python scalar. No autograd
  tensor can reach a writer, because none ever leaves the update.
- Nothing here renders a game. Watch/render mode is a separate evaluation path
  and belongs to a later phase.

A sink is any callable taking one metrics dictionary. :class:`CompositeReporter`
fans one record out to several, so a caller chooses its sinks and the training
loop stays unaware of which, or whether any, exist.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

MetricsRecord = dict[str, Any]

SCALAR_METRIC_NAMES: tuple[str, ...] = (
    # PPO objective and its diagnostics
    "policy_loss",
    "value_loss",
    "entropy",
    "approximate_kl_divergence",
    "clip_fraction",
    "explained_variance",
    "gradient_norm",
    "learning_rate",
    "epoch_zero_log_probability_max_deviation",
    # safety invariants: both must stay exactly zero
    "illegal_action_rate",
    "no_trump_action_rate",
    # what the policy is actually doing
    "completed_deals",
    "passed_out_deals",
    "mean_bid",
    "pass_rate",
    "learner_declare_rate",
    "contract_success_rate",
    "mean_learner_reward",
    "mean_absolute_score_difference",
    # throughput
    "environment_steps_per_second",
    "decisions_per_second",
    "update_seconds",
)
"""Metrics worth plotting. Every one is either part of the PPO objective, a
documented diagnostic of it, a safety invariant, or throughput. Counters that are
merely restated (``environment_steps``, ``decision_count``) are recorded in the
JSONL record but are not plotted as curves."""


class Reporter(Protocol):
    """Anything that can accept one metrics record."""

    def __call__(self, record: MetricsRecord) -> None: ...


def derived_safety_rates(record: MetricsRecord) -> MetricsRecord:
    """Return the record with the two safety invariants expressed as rates.

    The collector counts illegal and reserved-action selections; the gate is
    stated as a rate, and a rate of exactly ``0.0`` is what must be reported.
    """
    decisions = record.get("learner_decision_count") or 0
    illegal = record.get("illegal_action_count") or 0
    no_trump = record.get("no_trump_action_count") or 0
    enriched = dict(record)
    enriched["illegal_action_rate"] = (illegal / decisions) if decisions else 0.0
    enriched["no_trump_action_rate"] = (no_trump / decisions) if decisions else 0.0
    return enriched


class JsonlReporter:
    """Append one JSON object per update to a ``.jsonl`` file.

    Written and flushed as each update completes, so a run that is interrupted
    still leaves every record it had already produced.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")

    def __call__(self, record: MetricsRecord) -> None:
        self._stream.write(json.dumps(derived_safety_rates(record), default=str) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> JsonlReporter:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class TensorBoardReporter:
    """Write the scalar metrics to a TensorBoard event directory.

    ``torch.utils.tensorboard`` is imported lazily so that neither the engine nor
    a training run without logging ever pulls it in.
    """

    def __init__(self, log_directory: Path, step_key: str = "global_environment_steps") -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.log_directory = log_directory
        self.step_key = step_key
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(log_directory))

    def __call__(self, record: MetricsRecord) -> None:
        enriched = derived_safety_rates(record)
        step = int(enriched.get(self.step_key) or enriched.get("update") or 0)
        for name in SCALAR_METRIC_NAMES:
            value = enriched.get(name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                # ``None`` is a real answer -- "no deal finished, so there is no
                # mean bid" -- and must not be plotted as a zero.
                continue
            self._writer.add_scalar(f"train/{name}", float(value), step)
        self._writer.flush()

    def add_evaluation(self, record: MetricsRecord) -> None:
        """Record one isolated evaluation result against a frozen baseline."""
        step = int(record.get("global_environment_steps") or 0)
        opponent = record.get("opponent", "baseline")
        interval = record.get("paired_score_difference_ci") or {}
        for name, value in (
            ("mean_paired_score_difference", record.get("mean_paired_score_difference")),
            ("win_rate", record.get("win_rate")),
            ("paired_score_difference_ci_lower", interval.get("lower_bound")),
            ("paired_score_difference_ci_upper", interval.get("upper_bound")),
            ("illegal_action_rate", record.get("illegal_action_rate")),
        ):
            if isinstance(value, int | float) and not isinstance(value, bool):
                self._writer.add_scalar(f"eval/{opponent}/{name}", float(value), step)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> TensorBoardReporter:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class CompositeReporter:
    """Fan one record out to several sinks, in order."""

    def __init__(self, sinks: list[Callable[[MetricsRecord], None]]) -> None:
        self.sinks = sinks

    def __call__(self, record: MetricsRecord) -> None:
        for sink in self.sinks:
            sink(record)

    def close(self) -> None:
        for sink in self.sinks:
            close = getattr(sink, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> CompositeReporter:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def build_default_reporter(output_directory: Path, tensorboard: bool = True) -> CompositeReporter:
    """Return the JSONL (+ optional TensorBoard) sinks for one run directory."""
    sinks: list[Callable[[MetricsRecord], None]] = [
        JsonlReporter(output_directory / "updates.jsonl")
    ]
    if tensorboard:
        sinks.append(TensorBoardReporter(output_directory / "tensorboard"))
    return CompositeReporter(sinks)
