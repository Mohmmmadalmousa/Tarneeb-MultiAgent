"""Authenticate the completed TRAIN audit; never sample worlds or execute policies."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SOURCE = "phase11b_retention_train_value_audit_v2"
ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path("results/phase11b/retention_train_value_audit_v2")
PLAN_SHA = "76b920cd4bfcd3b46b0e12fa1572a5b16f13576f1193f6d8c4fa3f3fafa9913d"
COMPLETION_SHA = "967090df904e388a373906a75d80424a743456bb42a896b70bbfc59780160887"
EXCLUDED = frozenset((2592, 252, 1692, 1302))
EXPECTED = frozenset(
    [
        (row, action, 59)
        for row, actions in (
            (2412, (55, 56, 57)),
            (2502, (55, 56)),
            (2562, (55, 56, 57)),
            (2682, (55, 56, 57, 58)),
            (2652, (55, 56, 57, 58)),
            (2802, (55, 56, 57, 58)),
            (2832, (55, 56, 57, 58)),
        )
        for action in actions
    ]
    + [(2682, 58, 57), (2652, 58, 57)]
    + [(942, 59, action) for action in (55, 56, 57, 58)]
)
_AUTHORITY = object()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PairwiseEvidence:
    row: int  # Original TRAIN archive row, never minibatch position.
    preferred: int
    other: int
    source: str


@dataclass(frozen=True)
class RowBinding:
    fingerprint: str
    actions: tuple[int, ...]
    teacher: int


@dataclass(frozen=True)
class AuthenticatedTrainEvidence:
    pairs: tuple[PairwiseEvidence, ...]
    rows: tuple[RowBinding, ...]
    provenance: tuple[tuple[str, str], ...]
    _token: object = field(default=None, repr=False, compare=False)
    _checksum: str = field(default="", repr=False)

    def _identity(self):
        return _digest(
            (
                [(e.row, e.preferred, e.other, e.source) for e in self.pairs],
                [(b.fingerprint, b.actions, b.teacher) for b in self.rows],
                self.provenance,
            )
        )

    def validate(self):
        _require(
            self._token is _AUTHORITY and self._checksum == self._identity(),
            "Unauthenticated or modified TRAIN evidence",
        )


def _seal(pairs, rows, provenance):
    """Internal issuer, called only after verification (synthetic fixtures in tests)."""
    bundle = AuthenticatedTrainEvidence(tuple(pairs), tuple(rows), tuple(provenance), _AUTHORITY)
    object.__setattr__(bundle, "_checksum", bundle._identity())
    return bundle


def fingerprints_for(examples):
    """Same complete public-input encoding as the frozen selection manifest."""
    return tuple(
        hashlib.sha256(
            np.asarray(static, dtype="<f4").tobytes()
            + np.asarray(history, dtype="<i8").tobytes()
            + np.asarray([length], dtype="<i8").tobytes()
            + np.asarray(legal, dtype=bool).tobytes()
        ).hexdigest()
        for static, history, length, legal in zip(
            examples.statics, examples.histories, examples.lengths, examples.legal, strict=True
        )
    )


def _supported_comparisons(case, result, config):
    """Use existing simultaneous bands in ALL four policy/belief combinations."""
    actions, teacher = case["actions"], case["teacher"]
    row = case["training_selection"]["training_row_id"]
    gate = (
        result["ess"] >= config["minimum_ess"]
        and result["max_weight"] <= config["maximum_normalized_weight"]
        and result["weighted_lapse_responsibility"] <= config["maximum_lapse_responsibility"]
    )
    _require(gate == result["posterior_support_gate"], "Posterior-support gate mismatch")
    summaries, classes = [], set()
    _require(set(result["summaries"]) == set(config["continuations"]), "Missing policy")
    for policy in config["continuations"]:
        _require(set(result["summaries"][policy]) == set(config["beliefs"]), "Missing belief")
        for belief in config["beliefs"]:
            s = result["summaries"][policy][belief]
            _require([x["action"] for x in s["actions"]] == actions, "Legal comparisons changed")
            by_action = {x["action"]: x for x in s["actions"]}
            radius = s["band_radius"]
            _require(math.isfinite(radius) and radius >= 0, "Invalid confidence radius")
            for a in actions:
                x = by_action[a]
                for reference, key in ((59, "advantage"), (teacher, "teacher_advantage")):
                    delta = x["mean_margin"] - by_action[reference]["mean_margin"]
                    ci = [0, 0] if a == reference else [delta - radius, delta + radius]
                    ci_key = (
                        "advantage_simultaneous_95ci"
                        if key == "advantage"
                        else key + "_simultaneous_95ci"
                    )
                    _require(
                        np.allclose(x[ci_key], ci, rtol=0, atol=1e-10), "Confidence bound mismatch"
                    )
            bids = [by_action[a]["advantage_simultaneous_95ci"] for a in actions if a != 59]
            classification = (
                "raise_favored"
                if any(lo > 0 for lo, _ in bids)
                else "pass_favored"
                if all(hi < 0 for _, hi in bids)
                else "uncertain"
            )
            _require(s["classification"] == classification, "Classification mismatch")
            classes.add(classification)
            summaries.append(by_action)
    robust = next(iter(classes)) if gate and len(classes) == 1 else "uncertain"
    _require(result["robust_classification"] == robust, "Robust classification mismatch")
    if robust == "uncertain":
        return set()
    _require(row not in EXCLUDED, "Uncertain row cannot supply evidence")
    found = set()
    for action in actions:
        if action == 59:
            continue
        if all(s[action]["advantage_simultaneous_95ci"][0] > 0 for s in summaries):
            found.add((row, action, 59))
        if all(s[action]["advantage_simultaneous_95ci"][1] < 0 for s in summaries):
            found.add((row, 59, action))
        if (
            teacher != 59
            and action != teacher
            and all(s[action]["teacher_advantage_simultaneous_95ci"][0] > 0 for s in summaries)
        ):
            found.add((row, action, teacher))
    return found


def load_train_evidence() -> AuthenticatedTrainEvidence:
    from scripts import audit_phase11b_train_evidence as audit

    from tarneeb.auction_repair.data import PublicExamples

    root = ROOT.resolve()
    _require(Path.cwd().resolve() == root, "Run from repository root")
    folder = root / AUDIT
    for name, expected in (("preregistration.json", PLAN_SHA), ("completion.json", COMPLETION_SHA)):
        _require(audit.selection.sha256(folder / name) == expected, f"Pinned {name} hash mismatch")
    plan = audit.r.read(folder / "preregistration.json")
    completion = audit.r.read(folder / "completion.json")
    audit.verify_plan(plan)  # Sources, checkpoint, seeds, selection, exact public reconstruction.
    _require(
        completion["plan_identity"] == plan["identity"]
        and completion["states"] == 12
        and completion["worlds"] == 768
        and completion["status"] == "complete_audit",
        "Incomplete audit",
    )
    with np.load(audit.selection.TRAIN, allow_pickle=False) as data:
        examples = PublicExamples(
            *(data[k].copy() for k in ("statics", "histories", "lengths", "legal", "labels")),
            tuple(json.loads(str(data["metadata_json"]))),
        )
    fingerprints = fingerprints_for(examples)
    rows = tuple(
        RowBinding(fp, tuple(np.flatnonzero(mask).tolist()), int(label))
        for fp, mask, label in zip(fingerprints, examples.legal, examples.labels, strict=True)
    )
    pairs = set()
    for case, result_hash in zip(plan["cases"], completion["result_sha256"], strict=True):
        directory = folder / "states" / case["id"]
        result = audit.r.read(directory / "result.json")
        _require(
            _digest(result) == result_hash
            and _digest({k: v for k, v in result.items() if k != "result_sha256"})
            == result["result_sha256"],
            "Result hash mismatch",
        )
        _require(
            result["plan_identity"] == plan["identity"]
            and result["state_identity"] == _digest(case)
            and result["status"] == "complete_state_result"
            and result["completed_worlds"] == 64,
            "Result identity/completion mismatch",
        )
        row = case["training_selection"]["training_row_id"]
        binding = rows[row]
        _require(
            binding.fingerprint == case["training_selection"]["observation_history_mask_sha256"]
            and sorted(case["actions"]) == list(binding.actions)
            and case["teacher"] == binding.teacher,
            "TRAIN fingerprint, legality or teacher mismatch",
        )
        _require(
            len(result["world_sha256"]) == 64 and len(list(directory.glob("world_*.json"))) == 64,
            "Incomplete saved worlds",
        )
        for index, expected in enumerate(result["world_sha256"]):
            world = audit.r.read(directory / f"world_{index:03d}.json")
            payload = world["payload"]
            _require(world["sha256"] == expected == _digest(payload), "World checksum mismatch")
            _require(
                payload["world_index"] == index
                and payload["plan_identity"] == plan["identity"]
                and payload["state_identity"] == _digest(case)
                and payload["actions"] == case["actions"],
                "World identity/pairing mismatch",
            )
        pairs.update(_supported_comparisons(case, result, plan["scientific_config"]))
    _require(
        pairs == EXPECTED and len(pairs) == 30 and len({x[0] for x in pairs}) == 8,
        "Authenticated comparisons differ from the expected TRAIN mapping",
    )
    return _seal(
        [PairwiseEvidence(*x, SOURCE) for x in sorted(pairs)],
        rows,
        (
            ("source", SOURCE),
            ("plan_sha256", PLAN_SHA),
            ("completion_sha256", COMPLETION_SHA),
            ("selection_sha256", audit.MANIFEST_SHA),
            ("train_sha256", audit.selection.TRAIN_SHA),
        ),
    )
