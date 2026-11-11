"""The behaviour-cloning dataset contract (``bc_v1``) and its storage.

**The design decision that shapes everything here: the dataset stores seeds and
teacher actions, not observations.**

A naive dataset would store the 692-float observation for every decision. At the
sizes Phase 9 needs that is hundreds of megabytes, and worse, it freezes a *copy*
of the observation builder's output. If the frozen builder and the stored copy
ever disagreed, training would silently use the stale one.

Instead a dataset file holds, per deal, only:

- the match seed, dealer seat and deal index;
- the teacher's action at every decision, in order.

Everything else is **regenerated** by replaying the deal through the frozen
engine. The engine is deterministic, so this reproduces the identical deal; and
because observations come from the frozen ``ObservationBuilder`` at training
time, they cannot drift from what the environment actually produces. The files
are small enough to be trivial, and a test proves regeneration is exact.

**Sequence semantics.** One sequence is one ``(deal, seat)``. It begins at the
deal boundary, where the recurrent contract says the hidden state is zero, so a
BC sequence needs no stored ``h0`` -- unlike ``chunk_v2``, whose chunks start
mid-deal and must carry one.

That difference changes where the first event segment goes. In ``chunk_v2`` a
chunk head stores an *empty* segment, because the events before it are already
summarized in the stored ``h0`` (rule R4). Here there is no stored ``h0``: the
sequence starts at zero, so decision 0 must consume the events that happened
between the deal starting and that seat first acting. In general:

```text
segment[i] = events strictly after this seat's decision i-1
             and strictly before its decision i
segment[0] = events from the deal boundary up to the seat's first decision
```

Replay is then ``h = 0``; for each ``i``, consume ``segment[i]``, then evaluate
decision ``i``. That is rule R5 with a zero start, and it reconstructs exactly
the state the seat would have held under R1/R2 during a live rollout.

**One dataset, both architectures.** Because each decision records how many
events preceded it, the same file yields the GRU's between-decision segments
*and* the feedforward baseline's padded 80-row history. A1 therefore compares two
architectures on identical demonstration data rather than on two datasets that
happen to look similar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tarneeb.core.actions import ACTION_SPEC_VERSION, NUM_ACTIONS
from tarneeb.core.events import EVENT_SCHEMA_VERSION
from tarneeb.core.rules import RulesConfig
from tarneeb.env.event_features import EVENT_ID_DTYPE
from tarneeb.env.spaces import K_MAX, L_MAX, N_EVENT_FIELDS, OBS_SPEC_VERSION

BC_DATASET_VERSION = "bc_v1"
"""The demonstration-dataset schema. Recorded in the file and in a checkpoint."""

TEACHER_IDENTITY = "StrongHeuristicAgent"
"""Whose decisions these are. A different teacher is a different dataset."""


@dataclass(frozen=True, slots=True)
class BCSequence:
    """One ``(deal, seat)`` demonstration sequence, ready for the networks.

    ``segment_ids``/``segment_lengths`` drive the recurrent model;
    ``history_ids``/``history_masks`` drive the feedforward model. Both are
    derived from the same event stream, so the two architectures see exactly the
    same information.
    """

    deal_id: int
    seat: int
    match_seed: int
    statics: np.ndarray
    """``(T, 692)`` float32 actor-safe observations."""

    segment_ids: np.ndarray
    """``(T, K_MAX, 7)`` int16 -- events since this seat's previous decision."""

    segment_lengths: np.ndarray
    """``(T,)`` int64, each ``0..K_MAX``."""

    history_ids: np.ndarray
    """``(T, 80, 7)`` int16 -- the padded history, for the feedforward condition."""

    history_masks: np.ndarray
    """``(T, 80)`` bool."""

    legal_action_masks: np.ndarray
    """``(T, 65)`` bool, straight from the frozen engine."""

    teacher_actions: np.ndarray
    """``(T,)`` int64 -- what the Strong heuristic chose."""

    phases: np.ndarray
    """``(T,)`` int64 -- BID / DECLARE / PLAY, for the phase-split report."""

    @property
    def decision_count(self) -> int:
        return int(self.statics.shape[0])


@dataclass(frozen=True, slots=True)
class BCDatasetMetadata:
    """Everything needed to prove a dataset matches the running contracts."""

    dataset_version: str
    teacher: str
    rules_version: str
    rules_fingerprint: str
    obs_spec_version: str
    action_spec_version: str
    event_schema_version: str
    split_name: str
    seed_start: int
    seed_stop: int
    deal_count: int
    decision_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "teacher": self.teacher,
            "rules_version": self.rules_version,
            "rules_fingerprint": self.rules_fingerprint,
            "obs_spec_version": self.obs_spec_version,
            "action_spec_version": self.action_spec_version,
            "event_schema_version": self.event_schema_version,
            "split_name": self.split_name,
            "seed_start": self.seed_start,
            "seed_stop": self.seed_stop,
            "deal_count": self.deal_count,
            "decision_count": self.decision_count,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> BCDatasetMetadata:
        return cls(**values)

    def check_compatible(self, rules: RulesConfig) -> None:
        """Raise if this dataset was built under different frozen contracts."""
        expected = {
            "dataset_version": BC_DATASET_VERSION,
            "teacher": TEACHER_IDENTITY,
            "rules_version": rules.version,
            "rules_fingerprint": rules.fingerprint(),
            "obs_spec_version": OBS_SPEC_VERSION,
            "action_spec_version": ACTION_SPEC_VERSION,
            "event_schema_version": EVENT_SCHEMA_VERSION,
        }
        mismatches = [
            f"{field}: dataset has {getattr(self, field)!r}, code has {value!r}"
            for field, value in expected.items()
            if getattr(self, field) != value
        ]
        if mismatches:
            raise ValueError(
                "refusing to use a BC dataset built under different contracts:\n  "
                + "\n  ".join(mismatches)
            )


@dataclass(frozen=True, slots=True)
class BCDealRecord:
    """The compact on-disk record of one demonstrated deal."""

    match_seed: int
    dealer_seat: int
    deal_index: int
    teacher_actions: tuple[int, ...]
    """Every action taken in the deal, in order, across all four seats."""

    passed_out: bool


@dataclass(frozen=True, slots=True)
class BCDataset:
    """A set of demonstrated deals plus the metadata proving what they are."""

    metadata: BCDatasetMetadata
    deals: tuple[BCDealRecord, ...]

    def __len__(self) -> int:
        return len(self.deals)

    def save(self, path: Path) -> Path:
        """Write the dataset as one compact ``.npz``.

        Actions are stored as one flat array with per-deal offsets, which keeps
        the file to a few hundred kilobytes for thousands of deals.
        """
        lengths = np.array([len(deal.teacher_actions) for deal in self.deals], dtype=np.int64)
        flat_actions = (
            np.concatenate([np.array(deal.teacher_actions, dtype=np.int16) for deal in self.deals])
            if self.deals
            else np.zeros(0, dtype=np.int16)
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            metadata=np.array(json.dumps(self.metadata.as_dict())),
            match_seeds=np.array([d.match_seed for d in self.deals], dtype=np.int64),
            dealer_seats=np.array([d.dealer_seat for d in self.deals], dtype=np.int8),
            deal_indices=np.array([d.deal_index for d in self.deals], dtype=np.int64),
            passed_out=np.array([d.passed_out for d in self.deals], dtype=bool),
            action_lengths=lengths,
            actions=flat_actions,
        )
        return path

    @classmethod
    def load(cls, path: Path, rules: RulesConfig | None = None) -> BCDataset:
        """Load a dataset, refusing one built under different frozen contracts."""
        with np.load(path, allow_pickle=False) as stored:
            metadata = BCDatasetMetadata.from_dict(json.loads(str(stored["metadata"])))
            metadata.check_compatible(rules if rules is not None else RulesConfig())

            lengths = stored["action_lengths"]
            actions = stored["actions"]
            offsets = np.concatenate(([0], np.cumsum(lengths)))
            deals = tuple(
                BCDealRecord(
                    match_seed=int(stored["match_seeds"][index]),
                    dealer_seat=int(stored["dealer_seats"][index]),
                    deal_index=int(stored["deal_indices"][index]),
                    teacher_actions=tuple(
                        int(a) for a in actions[offsets[index] : offsets[index + 1]]
                    ),
                    passed_out=bool(stored["passed_out"][index]),
                )
                for index in range(len(lengths))
            )
        return cls(metadata=metadata, deals=deals)


@dataclass(frozen=True, slots=True)
class BCBatch:
    """A padded batch of sequences, time-major like ``chunk_v2``.

    Time-major because recurrent replay walks positions in order carrying hidden
    state. ``valid`` marks the real positions, and every loss is masked by it, so
    padding contributes exactly zero.
    """

    statics: np.ndarray
    """``(T, B, 692)`` float32."""

    segment_ids: np.ndarray
    """``(T, B, K_MAX, 7)`` int16; zero-width when the batch is feedforward-only."""

    segment_lengths: np.ndarray
    """``(T, B)`` int64."""

    history_ids: np.ndarray
    """``(T, B, 80, 7)`` int16; zero-width when the batch is recurrent-only."""

    history_masks: np.ndarray
    """``(T, B, 80)`` bool."""

    legal_action_masks: np.ndarray
    """``(T, B, 65)`` bool. Padded positions carry the single-legal-action sentinel."""

    teacher_actions: np.ndarray
    """``(T, B)`` int64."""

    phases: np.ndarray
    """``(T, B)`` int64; ``-1`` where padded."""

    valid: np.ndarray
    """``(T, B)`` bool."""

    @property
    def sequence_length(self) -> int:
        return int(self.statics.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.statics.shape[1])

    @property
    def valid_decision_count(self) -> int:
        return int(self.valid.sum())


PAD_LEGAL_ACTION = 0
"""The single legal action on a padded position's sentinel mask.

The frozen masking utility rejects an all-False mask, correctly: a distribution
over nothing is not a meaningful object. Nothing a padded position produces
reaches a loss, because every term is masked by ``valid``.
"""


def collate_sequences(sequences: list[BCSequence], architecture: str = "both") -> BCBatch:
    """Pad a list of sequences into one time-major batch.

    ``architecture`` selects which history modality to materialize. The two are
    derived from the same event stream and are interchangeable information, but
    materializing both doubles memory for nothing: a recurrent actor never reads
    the padded 80-row block and a feedforward actor never reads the segments.
    The unused modality becomes a zero-width array rather than a wrong-shaped
    one, so reading it fails loudly instead of silently returning zeros.
    """
    if not sequences:
        raise ValueError("cannot collate an empty list of sequences")
    if architecture not in ("gru", "ff", "both"):
        raise ValueError(f"architecture must be 'gru', 'ff' or 'both', got {architecture!r}")

    length = max(sequence.decision_count for sequence in sequences)
    batch_size = len(sequences)
    observation_dim = sequences[0].statics.shape[1]
    want_segments = architecture in ("gru", "both")
    want_history = architecture in ("ff", "both")

    statics = np.zeros((length, batch_size, observation_dim), dtype=np.float32)
    segment_ids = np.zeros(
        (length, batch_size, K_MAX if want_segments else 0, N_EVENT_FIELDS), dtype=np.int16
    )
    segment_lengths = np.zeros((length, batch_size), dtype=np.int64)
    history_ids = np.zeros(
        (length, batch_size, L_MAX if want_history else 0, N_EVENT_FIELDS), dtype=np.int16
    )
    history_masks = np.zeros((length, batch_size, L_MAX if want_history else 0), dtype=bool)
    legal_action_masks = np.zeros((length, batch_size, NUM_ACTIONS), dtype=bool)
    legal_action_masks[..., PAD_LEGAL_ACTION] = True
    teacher_actions = np.zeros((length, batch_size), dtype=np.int64)
    phases = np.full((length, batch_size), -1, dtype=np.int64)
    valid = np.zeros((length, batch_size), dtype=bool)

    for column, sequence in enumerate(sequences):
        count = sequence.decision_count
        statics[:count, column] = sequence.statics
        segment_lengths[:count, column] = sequence.segment_lengths
        if want_segments:
            segment_ids[:count, column] = sequence.segment_ids
        if want_history:
            history_ids[:count, column] = sequence.history_ids
            history_masks[:count, column] = sequence.history_masks
        legal_action_masks[:count, column] = sequence.legal_action_masks
        teacher_actions[:count, column] = sequence.teacher_actions
        phases[:count, column] = sequence.phases
        valid[:count, column] = True

    return BCBatch(
        statics=statics,
        segment_ids=segment_ids,
        segment_lengths=segment_lengths,
        history_ids=history_ids,
        history_masks=history_masks,
        legal_action_masks=legal_action_masks,
        teacher_actions=teacher_actions,
        phases=phases,
        valid=valid,
    )


def empty_segment_block() -> np.ndarray:
    """Return the all-zero ``(K_MAX, 7)`` block used for an empty segment."""
    return np.zeros((K_MAX, N_EVENT_FIELDS), dtype=EVENT_ID_DTYPE)
