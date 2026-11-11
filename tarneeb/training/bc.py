"""Behaviour cloning from the frozen Strong heuristic (Phase 9, Part B).

Two jobs: turn the Strong heuristic's play into demonstration sequences, and
train a policy to imitate them.

**Why the information boundary is inherited rather than argued.**
``StrongHeuristicAgent.act`` takes a :class:`~tarneeb.env.public_view.PublicView`
and nothing else. That view is actor-safe by construction and already proved so
by the frozen Phase 4/5 leakage tests -- it carries the observer's own hand and
the public record, never another seat's cards. So the teacher physically cannot
condition on information the student is forbidden, and the labels cannot encode
anything the student may not see. Phase 9 adds tests that check this holds for
the generated dataset too, rather than relying on the argument alone.

**Why the student stays recurrent.** It would be much simpler to flatten BC into
a 65-way classification over independent decisions. That would also make the BC
initialization useless for PPO, because the GRU would never learn to carry a
useful state and the recurrent policy would start from weights tuned for a
context it will not have. So BC trains over whole ``(deal, seat)`` sequences,
resetting at the deal boundary exactly as the recurrent contract requires.

**The critic is not cloned.** The Strong heuristic has no value function, and its
internal evaluations are not a value estimate of the frozen reward. PPO after BC
therefore starts from a BC actor and a freshly initialized critic.

The training objective is masked cross-entropy over the legal actions. Illegal
actions keep probability exactly zero through the frozen mask; legality is never
taught through a penalty.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.buffers.bc import (
    BC_DATASET_VERSION,
    TEACHER_IDENTITY,
    BCBatch,
    BCDataset,
    BCDatasetMetadata,
    BCDealRecord,
    BCSequence,
    collate_sequences,
)
from tarneeb.core.actions import ACTION_SPEC_VERSION, NO_TRUMP_ACTION, NUM_ACTIONS
from tarneeb.core.cards import NUM_SEATS
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import EVENT_SCHEMA_VERSION, Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.env.event_features import EVENT_ID_DTYPE, event_ids
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.public_view import PublicViewBuilder
from tarneeb.env.spaces import K_MAX, L_MAX, N_EVENT_FIELDS, OBS_SPEC_VERSION
from tarneeb.models.masking import MaskedCategorical
from tarneeb.models.nets import ActorNetwork
from tarneeb.models.recurrent import RecurrentActorNetwork, zero_hidden_state

MAX_DECISIONS_PER_DEAL = 200
"""Safety stop. A legal deal needs far fewer; exceeding it means a driver bug."""

LIVE_PHASES = (Phase.BID, Phase.DECLARE, Phase.PLAY)


# --- dataset generation -----------------------------------------------------


def generate_bc_dataset(
    seeds: range,
    split_name: str,
    rules: RulesConfig | None = None,
    dealer_rotation: int = NUM_SEATS,
) -> BCDataset:
    """Play ``seeds`` deals with the Strong heuristic in all four seats.

    Only the seed, the dealer and the teacher's action sequence are recorded;
    everything else is regenerated deterministically at load time by
    :func:`regenerate_sequences`. A passed-out deal is kept -- passing is a real
    decision the teacher made, and dropping those hands would bias the auction
    labels toward hands worth bidding.
    """
    resolved_rules = rules if rules is not None else RulesConfig()
    engine = TarneebEngine(resolved_rules)
    views = PublicViewBuilder(engine)
    teacher = StrongHeuristicAgent()

    records: list[BCDealRecord] = []
    decision_total = 0

    for index, match_seed in enumerate(seeds):
        dealer_seat = index % dealer_rotation
        state = engine.new_deal(dealer_seat, deal_index=0, match_seed=match_seed)
        engine.deal(state)
        teacher.reset()

        actions: list[int] = []
        passed_out = False
        while state.phase in LIVE_PHASES:
            if len(actions) > MAX_DECISIONS_PER_DEAL:  # pragma: no cover - safety stop
                raise AssertionError("deal exceeded the maximum decision count")
            acting_seat = state.current_player_seat
            assert acting_seat is not None
            action = teacher.act(views.build(state, acting_seat))
            actions.append(int(action))
            engine.step(state, action)
            if state.phase is Phase.DEAL:
                passed_out = True
                break

        records.append(
            BCDealRecord(
                match_seed=match_seed,
                dealer_seat=dealer_seat,
                deal_index=0,
                teacher_actions=tuple(actions),
                passed_out=passed_out,
            )
        )
        decision_total += len(actions)

    metadata = BCDatasetMetadata(
        dataset_version=BC_DATASET_VERSION,
        teacher=TEACHER_IDENTITY,
        rules_version=resolved_rules.version,
        rules_fingerprint=resolved_rules.fingerprint(),
        obs_spec_version=OBS_SPEC_VERSION,
        action_spec_version=ACTION_SPEC_VERSION,
        event_schema_version=EVENT_SCHEMA_VERSION,
        split_name=split_name,
        seed_start=seeds.start,
        seed_stop=seeds.stop,
        deal_count=len(records),
        decision_count=decision_total,
    )
    return BCDataset(metadata=metadata, deals=tuple(records))


def regenerate_sequences(
    record: BCDealRecord,
    rules: RulesConfig | None = None,
    engine: TarneebEngine | None = None,
    observations: ObservationBuilder | None = None,
) -> list[BCSequence]:
    """Replay one recorded deal and return its four ``(deal, seat)`` sequences.

    The deal is reconstructed by stepping the frozen engine with the stored
    teacher actions, so every observation comes from the frozen builder at
    training time and cannot drift from a stored copy.

    Segment semantics, restated because it differs from ``chunk_v2``: a BC
    sequence starts at the deal boundary where the hidden state is zero, so
    decision 0 consumes the events from the deal start up to that seat's first
    action. There is no stored ``h0`` to summarize them.
    """
    resolved_rules = rules if rules is not None else RulesConfig()
    resolved_engine = engine if engine is not None else TarneebEngine(resolved_rules)
    builder = observations if observations is not None else ObservationBuilder(resolved_engine)

    state = resolved_engine.new_deal(
        record.dealer_seat, deal_index=record.deal_index, match_seed=record.match_seed
    )
    resolved_engine.deal(state)

    per_seat: list[dict[str, list[Any]]] = [
        {
            "statics": [],
            "segment_ids": [],
            "segment_lengths": [],
            "history_ids": [],
            "history_masks": [],
            "legal": [],
            "actions": [],
            "phases": [],
        }
        for _ in range(NUM_SEATS)
    ]
    # How many events this seat has already folded into its own sequence.
    consumed_by_seat = [0] * NUM_SEATS

    for action in record.teacher_actions:
        if state.phase not in LIVE_PHASES:  # pragma: no cover - record implies liveness
            break
        acting_seat = state.current_player_seat
        assert acting_seat is not None

        seat_record = per_seat[acting_seat]
        events_before = len(state.events)
        start = consumed_by_seat[acting_seat]
        segment_length = events_before - start
        if segment_length > K_MAX:  # pragma: no cover - the proved bound is 14
            raise AssertionError(
                f"seat {acting_seat} faced a {segment_length}-event segment, "
                f"which exceeds K_MAX={K_MAX}"
            )

        segment = np.zeros((K_MAX, N_EVENT_FIELDS), dtype=EVENT_ID_DTYPE)
        for position, event in enumerate(state.events[start:events_before]):
            segment[position] = event_ids(event, acting_seat)

        history_ids, history_masks = builder.padded_history(state, acting_seat)

        seat_record["statics"].append(builder.static_actor(state, acting_seat))
        seat_record["segment_ids"].append(segment)
        seat_record["segment_lengths"].append(segment_length)
        seat_record["history_ids"].append(history_ids)
        seat_record["history_masks"].append(history_masks)
        seat_record["legal"].append(resolved_engine.legal_actions(state))
        seat_record["actions"].append(int(action))
        seat_record["phases"].append(int(state.phase))
        consumed_by_seat[acting_seat] = events_before

        resolved_engine.step(state, int(action))
        if state.phase is Phase.DEAL:
            break

    sequences: list[BCSequence] = []
    for seat in range(NUM_SEATS):
        seat_record = per_seat[seat]
        if not seat_record["actions"]:
            continue
        sequences.append(
            BCSequence(
                deal_id=record.match_seed,
                seat=seat,
                match_seed=record.match_seed,
                statics=np.stack(seat_record["statics"]).astype(np.float32),
                segment_ids=np.stack(seat_record["segment_ids"]),
                segment_lengths=np.array(seat_record["segment_lengths"], dtype=np.int64),
                history_ids=np.stack(seat_record["history_ids"]),
                history_masks=np.stack(seat_record["history_masks"]),
                legal_action_masks=np.stack(seat_record["legal"]),
                teacher_actions=np.array(seat_record["actions"], dtype=np.int64),
                phases=np.array(seat_record["phases"], dtype=np.int64),
            )
        )
    return sequences


def iterate_batches(
    dataset: BCDataset,
    batch_sequences: int,
    shuffle_generator: np.random.Generator | None = None,
    rules: RulesConfig | None = None,
    architecture: str = "both",
) -> list[BCBatch]:
    """Regenerate the whole dataset and group it into padded batches.

    Sequences are grouped by length before batching, which keeps padding low
    without ever mixing a sequence's positions -- a short auction-only sequence
    and a full 13-trick one would otherwise pad to the longer length.
    """
    resolved_rules = rules if rules is not None else RulesConfig()
    engine = TarneebEngine(resolved_rules)
    builder = ObservationBuilder(engine)

    sequences: list[BCSequence] = []
    for record in dataset.deals:
        sequences.extend(regenerate_sequences(record, resolved_rules, engine, builder))

    sequences.sort(key=lambda sequence: sequence.decision_count)
    batches = [
        collate_sequences(sequences[start : start + batch_sequences], architecture)
        for start in range(0, len(sequences), batch_sequences)
    ]
    if shuffle_generator is not None:
        order = shuffle_generator.permutation(len(batches))
        batches = [batches[index] for index in order]
    return batches


# --- training ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BCConfig:
    """Behaviour-cloning hyper-parameters."""

    learning_rate: float = 1e-3
    epochs: int = 3
    batch_sequences: int = 64
    max_gradient_norm: float = 0.5
    weight_decay: float = 0.0
    shuffle_seed: int = 11
    architecture: str = "gru"
    """``"gru"`` or ``"ff"`` -- A1 clones both from the same demonstrations."""

    def __post_init__(self) -> None:
        if self.architecture not in ("gru", "ff"):
            raise ValueError(f"architecture must be 'gru' or 'ff', got {self.architecture!r}")
        if self.epochs < 1 or self.batch_sequences < 1:
            raise ValueError("epochs and batch_sequences must be at least 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BCMetrics:
    """Accuracy and likelihood, overall and split by game phase."""

    decision_count: int = 0
    correct: int = 0
    negative_log_likelihood_total: float = 0.0
    by_phase_count: dict[int, int] = field(default_factory=dict)
    by_phase_correct: dict[int, int] = field(default_factory=dict)
    max_illegal_probability: float = 0.0
    max_no_trump_probability: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.decision_count if self.decision_count else 0.0

    @property
    def negative_log_likelihood(self) -> float:
        if not self.decision_count:
            return 0.0
        return self.negative_log_likelihood_total / self.decision_count

    def phase_accuracy(self) -> dict[str, float | None]:
        names = {int(Phase.BID): "BID", int(Phase.DECLARE): "DECLARE", int(Phase.PLAY): "PLAY"}
        result: dict[str, float | None] = {}
        for phase_value, name in names.items():
            count = self.by_phase_count.get(phase_value, 0)
            result[name] = self.by_phase_correct.get(phase_value, 0) / count if count else None
        return result

    def phase_counts(self) -> dict[str, int]:
        names = {int(Phase.BID): "BID", int(Phase.DECLARE): "DECLARE", int(Phase.PLAY): "PLAY"}
        return {name: self.by_phase_count.get(value, 0) for value, name in names.items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_count": self.decision_count,
            "accuracy": self.accuracy,
            "negative_log_likelihood": self.negative_log_likelihood,
            "phase_accuracy": self.phase_accuracy(),
            "phase_counts": self.phase_counts(),
            "max_illegal_probability": self.max_illegal_probability,
            "max_no_trump_probability": self.max_no_trump_probability,
        }


def _to_tensors(batch: BCBatch) -> dict[str, Tensor]:
    return {
        "statics": torch.from_numpy(batch.statics),
        "segment_ids": torch.from_numpy(batch.segment_ids.astype(np.int64)),
        "segment_lengths": torch.from_numpy(batch.segment_lengths),
        "history_ids": torch.from_numpy(batch.history_ids.astype(np.int64)),
        "history_masks": torch.from_numpy(batch.history_masks),
        "legal": torch.from_numpy(batch.legal_action_masks),
        "targets": torch.from_numpy(batch.teacher_actions),
        "valid": torch.from_numpy(batch.valid),
    }


def recurrent_batch_log_probabilities(
    actor: RecurrentActorNetwork, tensors: dict[str, Tensor]
) -> Tensor:
    """Return ``(T, B, 65)`` masked log-probabilities for a recurrent actor.

    Walks the sequence in order carrying hidden state, consuming each position's
    segment before evaluating it -- the same rule PPO replay uses, with a zero
    start because a BC sequence begins at a deal boundary.
    """
    batch_size = tensors["statics"].shape[1]
    hidden = zero_hidden_state(batch_size, actor.spec)
    rows: list[Tensor] = []
    for position in range(tensors["statics"].shape[0]):
        logits, hidden = actor(
            tensors["statics"][position],
            tensors["segment_ids"][position],
            tensors["segment_lengths"][position],
            hidden,
        )
        rows.append(
            MaskedCategorical.from_logits(logits, tensors["legal"][position]).log_probabilities
        )
    return torch.stack(rows)


def feedforward_batch_log_probabilities(actor: ActorNetwork, tensors: dict[str, Tensor]) -> Tensor:
    """Return ``(T, B, 65)`` masked log-probabilities for the feedforward actor.

    Flattened to one batch because a feedforward policy carries no state, so
    position order is irrelevant to its output.
    """
    sequence_length, batch_size = tensors["statics"].shape[:2]
    flat_logits = actor(
        tensors["statics"].reshape(sequence_length * batch_size, -1),
        tensors["history_ids"].reshape(sequence_length * batch_size, L_MAX, N_EVENT_FIELDS),
        tensors["history_masks"].reshape(sequence_length * batch_size, L_MAX),
    )
    flat_legal = tensors["legal"].reshape(sequence_length * batch_size, NUM_ACTIONS)
    log_probabilities = MaskedCategorical.from_logits(flat_logits, flat_legal).log_probabilities
    return log_probabilities.reshape(sequence_length, batch_size, NUM_ACTIONS)


def batch_log_probabilities(
    actor: RecurrentActorNetwork | ActorNetwork, tensors: dict[str, Tensor]
) -> Tensor:
    if isinstance(actor, RecurrentActorNetwork):
        return recurrent_batch_log_probabilities(actor, tensors)
    return feedforward_batch_log_probabilities(actor, tensors)


def masked_cross_entropy(log_probabilities: Tensor, targets: Tensor, valid: Tensor) -> Tensor:
    """Return the mean negative log-likelihood over the valid positions only.

    The log-probabilities already come from the frozen masked distribution, so an
    illegal action carries probability exactly zero and legality is never taught
    through a penalty term.
    """
    chosen = log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    weights = valid.to(chosen.dtype)
    total = weights.sum()
    if float(total) == 0.0:  # pragma: no cover - empty batches are not produced
        return (chosen * weights).sum() * 0.0
    return -(chosen * weights).sum() / total


def accumulate_metrics(
    metrics: BCMetrics,
    log_probabilities: Tensor,
    batch: BCBatch,
    valid: Tensor,
) -> None:
    """Fold one batch's predictions into the running report."""
    with torch.no_grad():
        probabilities = log_probabilities.exp()
        predicted = log_probabilities.argmax(dim=-1)
        targets = torch.from_numpy(batch.teacher_actions)
        correct = (predicted == targets) & valid
        chosen = log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        metrics.decision_count += int(valid.sum())
        metrics.correct += int(correct.sum())
        metrics.negative_log_likelihood_total += float(-(chosen * valid).sum())

        illegal = ~torch.from_numpy(batch.legal_action_masks)
        valid_illegal = probabilities * illegal * valid.unsqueeze(-1)
        metrics.max_illegal_probability = max(
            metrics.max_illegal_probability, float(valid_illegal.max())
        )
        metrics.max_no_trump_probability = max(
            metrics.max_no_trump_probability,
            float((probabilities[..., NO_TRUMP_ACTION] * valid).max()),
        )

        phases = batch.phases
        valid_numpy = valid.numpy()
        correct_numpy = correct.numpy()
        for phase_value in np.unique(phases[valid_numpy]):
            selected = (phases == phase_value) & valid_numpy
            key = int(phase_value)
            metrics.by_phase_count[key] = metrics.by_phase_count.get(key, 0) + int(selected.sum())
            metrics.by_phase_correct[key] = metrics.by_phase_correct.get(key, 0) + int(
                (correct_numpy & selected).sum()
            )


def evaluate_bc(actor: RecurrentActorNetwork | ActorNetwork, batches: list[BCBatch]) -> BCMetrics:
    """Return accuracy and likelihood on a held-out split, without training."""
    metrics = BCMetrics()
    was_training = actor.training
    actor.eval()
    with torch.no_grad():
        for batch in batches:
            tensors = _to_tensors(batch)
            log_probabilities = batch_log_probabilities(actor, tensors)
            accumulate_metrics(metrics, log_probabilities, batch, tensors["valid"])
    if was_training:
        actor.train()
    return metrics


@dataclass
class BCResult:
    """What one behaviour-cloning run produced."""

    architecture: str
    epochs_completed: int = 0
    train_metrics: list[BCMetrics] = field(default_factory=list)
    validation_metrics: list[BCMetrics] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    optimizer_steps: int = 0

    @property
    def best_validation_accuracy(self) -> float:
        return max((m.accuracy for m in self.validation_metrics), default=0.0)


def train_behaviour_cloning(
    actor: RecurrentActorNetwork | ActorNetwork,
    train_batches: list[BCBatch],
    validation_batches: list[BCBatch],
    config: BCConfig,
) -> BCResult:
    """Clone the teacher into ``actor`` and return the per-epoch report."""
    optimizer = torch.optim.Adam(
        actor.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    shuffle_generator = np.random.default_rng(config.shuffle_seed)
    result = BCResult(architecture=config.architecture)
    start_time = time.perf_counter()

    for _epoch in range(config.epochs):
        actor.train()
        epoch_metrics = BCMetrics()
        order = shuffle_generator.permutation(len(train_batches))
        for index in order:
            batch = train_batches[int(index)]
            tensors = _to_tensors(batch)
            log_probabilities = batch_log_probabilities(actor, tensors)
            loss = masked_cross_entropy(log_probabilities, tensors["targets"], tensors["valid"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(actor.parameters(), config.max_gradient_norm)
            optimizer.step()
            result.optimizer_steps += 1

            accumulate_metrics(epoch_metrics, log_probabilities.detach(), batch, tensors["valid"])

        result.train_metrics.append(epoch_metrics)
        result.validation_metrics.append(evaluate_bc(actor, validation_batches))
        result.epochs_completed += 1

    result.wall_clock_seconds = time.perf_counter() - start_time
    return result


def dataset_path(directory: Path, split_name: str) -> Path:
    """Return the conventional file name for one dataset split."""
    return directory / f"bc_{split_name}.npz"
