"""Prepare independent public auction controls only; no continuation entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tarneeb import auction_value_audit as audit
from tarneeb.agents.heuristic_strong import best_trump_candidate
from tarneeb.auction_repair.data import encode_public_view, public_teacher_label, strength_bucket
from tarneeb.core.cards import card_name, hand_from_cards
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase

ROOT = Path("results/phase11b/independent_controls_v3")
OLD = Path("results/phase11b")
TRAIN = OLD / "auction_bc_smoke_s8300001_v1/train_examples.npz"
DEV = OLD / "auction_bc_smoke_s8300001_v1/dev_examples.npz"
INPUTS = OLD / "neural_policy_comparison_v2/inputs.json"
PLAN = OLD / "counterfactual_value_audit_v2/preregistration.json"
CATEGORIES = (
    "partner_raise_candidate",
    "partner_pass_candidate",
    "opening_13_candidate",
    "near_sweep_lower_candidate",
)
PREFIXES = ((54, 59), (52, 54, 59), (53, 54, 59), (59, 54, 59))
SHAPES = (
    (5, 4, 4, 0),
    (6, 4, 3, 0),
    (7, 3, 3, 0),
    (7, 4, 2, 0),
    (6, 5, 2, 0),
    (5, 5, 3, 0),
    (4, 4, 3, 2),
    (7, 3, 2, 1),
    (8, 3, 2, 0),
)
PERMS = np.array(list(permutations(range(4))))
BITS = np.array([i.bit_count() for i in range(8192)])
DESIGN = {
    "version": "phase11b_independent_controls_v3_candidates",
    "categories": list(CATEGORIES),
    "per_category": 4,
    "selection_seeds": list(range(8700000, 8700016)),
    "reserved_world_seeds": list(range(8710000, 8710016)),
    "reserved_bootstrap_seeds": list(range(8720000, 8720016)),
    "max_attempts_per_case": 5000,
    "selection_wall_seconds": 120,
    "minimum_card_replacements_under_any_suit_permutation": 3,
    "partner_prefixes": [list(x) for x in PREFIXES],
    "opening_prefixes": [[], [59], [59, 59], [59, 59, 59]],
    "selection": {
        CATEGORIES[
            0
        ]: "7-9 cards in a random suit including AKQJ; at least one side ace; side cards rank-weighted exp(rank/2).",
        CATEGORIES[
            1
        ]: "Uniform own-hand draw; longest suit <=5, at most one ace and at most three A/K/Q cards.",
        CATEGORIES[
            2
        ]: "Random listed shape/suit assignment; each occupied suit includes ace and, if length>=2, king; remaining ranks weighted exp(rank/2); at least three aces, three kings and nine A/K/Q/J cards.",
        CATEGORIES[
            3
        ]: "Same shape proposal, rank-weighted cards, longest suit deliberately lacks its ace; one or two aces total and at least seven A/K/Q/J cards.",
    },
    "opening_shape_proposals": [list(x) for x in SHAPES],
    "acceptance": "First candidate meeting own-hand category and overlap gates, in fixed category/slot order. No model outputs, value estimates or teacher-action filters. Stop on exhaustion; never relax gates.",
    "labels": "Categories are hypotheses. Teacher recommendations are heuristic-only. All counterfactual classifications are unavailable until separately authorized evaluation.",
    "hidden_information": "No deal or hidden-world sampler. Sorted unseen-card placeholders only encode legal public prefixes; alternate reversed placeholders verify invariance. Neither allocation is used for evaluation or selection.",
    "historical_limit": "Original BC and Phase11 PPO per-decision actor hands are unavailable here; no exhaustive independence claim for those histories. Seed separation does not prove such independence.",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def fingerprint(static, history, legal):
    return hashlib.sha256(
        np.asarray(static, dtype=np.float32).tobytes()
        + np.asarray(history, dtype=np.int64).reshape(-1, 7).tobytes()
        + np.asarray(legal, dtype=bool).tobytes()
    ).hexdigest()


def history_hash(history):
    return hashlib.sha256(np.asarray(history, dtype=np.int64).reshape(-1, 7).tobytes()).hexdigest()


def suit_masks(hand):
    return tuple(sum(1 << (c % 13) for c in hand if c // 13 == s) for s in range(4))


def nearest(hand, references):
    """Exact minimum replacements over all 24 suit bijections, retaining ranks."""
    if not references:
        return {"minimum_replacements": 13, "nearest_id": None}
    masks = np.array([x[1] for x in references])
    intersections = BITS[np.array(suit_masks(hand))[None, :, None] & masks[:, None, :]]
    overlaps = intersections[:, np.arange(4)[None, :], PERMS].sum(axis=2)
    per_hand = overlaps.max(axis=1)
    index = int(per_hand.argmax())
    return {"minimum_replacements": 13 - int(per_hand[index]), "nearest_id": references[index][0]}


def load_references():
    groups, inputs, seeds, counts = {}, set(), set(), {}
    static_counts, history_counts = Counter(), Counter()
    for label, path in (("auction_bc_train", TRAIN), ("auction_bc_dev", DEV)):
        with np.load(path, allow_pickle=False) as f:
            d = {k: f[k] for k in f.files}
        metadata = json.loads(str(d["metadata_json"]))
        unique, templates = {}, {}
        for i, meta in enumerate(metadata):
            hand = np.flatnonzero(d["statics"][i, :52]).tolist()
            mask = suit_masks(hand)
            unique.setdefault(mask, f"{label}:{i}")
            if meta["case_family"] == "constructed":
                templates.setdefault(mask, f"{label}:{i}")
            inputs.add(
                fingerprint(d["statics"][i], d["histories"][i, : d["lengths"][i]], d["legal"][i])
            )
            static_counts[hashlib.sha256(d["statics"][i].tobytes()).hexdigest()] += 1
            history_counts[history_hash(d["histories"][i, : d["lengths"][i]])] += 1
            seeds.add(meta["seed"])
        groups[label] = [(v, k) for k, v in unique.items()]
        if templates:
            groups["constructed_templates"] = [(v, k) for k, v in templates.items()]
        counts[label] = {"rows": len(metadata), "unique_hands": len(unique)}
    previous = read(INPUTS)
    cases = read(PLAN)["cases"]
    groups["audit_v2"] = [(c["id"], suit_masks(c["own_hand"])) for c in cases]
    for v in previous.values():
        static_counts[
            hashlib.sha256(np.array(v["observation"], np.float32).tobytes()).hexdigest()
        ] += 1
        history_counts[history_hash(v["event_ids"])] += 1
        inputs.add(
            fingerprint(
                np.array(v["observation"], np.float32),
                np.array(v["event_ids"], np.int64).reshape(-1, 7),
                np.array(v["legal_action_mask"], bool),
            )
        )
    for c in cases:
        seeds.update((c["world_seed"], c["bootstrap_seed"]))
    seeds.update((8400000, 8400100, 8300001, 8600001))
    counts["audit_v2"] = {"rows": len(cases)}
    counts["static_fingerprints"] = dict(static_counts)
    counts["public_history_fingerprints"] = dict(history_counts)
    return groups, inputs, seeds, counts


def weighted_cards(rng, pool, n):
    pool = np.array(sorted(pool))
    weights = np.exp((pool % 13) / 2.0)
    return rng.choice(pool, n, replace=False, p=weights / weights.sum()).tolist()


def propose_hand(rng, category):
    if category == CATEGORIES[1]:
        return sorted(rng.choice(52, 13, replace=False).tolist())
    if category == CATEGORIES[0]:
        suit, length = int(rng.integers(4)), int(rng.integers(7, 10))
        hand = [suit * 13 + r for r in (9, 10, 11, 12)]
        hand += rng.choice([suit * 13 + r for r in range(9)], length - 4, replace=False).tolist()
        hand += weighted_cards(rng, [c for c in range(52) if c // 13 != suit], 13 - length)
        return sorted(hand)
    lengths = SHAPES[int(rng.integers(len(SHAPES)))]
    suits = rng.permutation(4).tolist()
    hand = []
    for j, length in enumerate(lengths):
        suit = suits[j]
        mandatory = (
            ([12] + ([11] if length >= 2 else [])) if length and category == CATEGORIES[2] else []
        )
        pool = [
            suit * 13 + r
            for r in range(13)
            if r not in mandatory and not (category == CATEGORIES[3] and j == 0 and r == 12)
        ]
        hand += [suit * 13 + r for r in mandatory]
        hand += weighted_cards(rng, pool, length - len(mandatory))
    return sorted(hand)


def category_accepts(hand, category):
    lengths = [sum(c // 13 == s for c in hand) for s in range(4)]
    aces, kings = sum(c % 13 == 12 for c in hand), sum(c % 13 == 11 for c in hand)
    if category == CATEGORIES[0]:
        return aces >= 2 and 7 <= max(lengths) <= 9
    if category == CATEGORIES[1]:
        return max(lengths) <= 5 and aces <= 1 and sum(c % 13 >= 10 for c in hand) <= 3
    if category == CATEGORIES[2]:
        return aces >= 3 and kings >= 3 and sum(c % 13 >= 9 for c in hand) >= 9
    return 1 <= aces <= 2 and sum(c % 13 >= 9 for c in hand) >= 7


def encode(case, reverse=False):
    unseen = sorted(set(range(52)) - set(case["own_hand"]), reverse=reverse)
    hands = [hand_from_cards(case["own_hand"])] + [
        hand_from_cards(unseen[i : i + 13]) for i in range(0, 39, 13)
    ]
    view = audit.decision_view(case, hands)
    static, history, length = encode_public_view(view)
    return view, static, history[:length], view.legal_action_mask.copy()


def select(references):
    groups, old_inputs, old_seeds, counts = references
    reserved = (
        DESIGN["selection_seeds"]
        + DESIGN["reserved_world_seeds"]
        + DESIGN["reserved_bootstrap_seeds"]
    )
    assert len(set(reserved)) == 48 and 20250910 not in reserved and not set(reserved) & old_seeds
    cases, inputs, selected, evidence = [], {}, [], {}
    started = time.monotonic()
    for index in range(16):
        category, slot = CATEGORIES[index // 4], index % 4
        rng = np.random.default_rng(DESIGN["selection_seeds"][index])
        rejected = Counter()
        for attempt in range(1, DESIGN["max_attempts_per_case"] + 1):
            if time.monotonic() - started > DESIGN["selection_wall_seconds"]:
                raise TimeoutError("Selection cap; no relaxation or automatic retry")
            hand = propose_hand(rng, category)
            if not category_accepts(hand, category):
                rejected["category"] += 1
                continue
            distances = {k: nearest(hand, v) for k, v in groups.items()}
            distances["earlier_v3_cases"] = nearest(hand, selected)
            if min(x["minimum_replacements"] for x in distances.values()) < 3:
                rejected["hand_family"] += 1
                continue
            prefix = list(PREFIXES[slot]) if index < 8 else [59] * slot
            case = {
                "id": f"V3_{index + 1:02d}",
                "index": index,
                "category": category,
                "own_hand": hand,
                "cards": [card_name(c) for c in hand],
                "prefix": prefix,
                "dealer": (-1 - len(prefix)) % 4,
                "actor_seat": 0,
                "high_bid": 9 if index < 8 else None,
                "owner": "partner" if index < 8 else "none",
                "selection_seed": reserved[index],
                "world_seed": DESIGN["reserved_world_seeds"][index],
                "bootstrap_seed": DESIGN["reserved_bootstrap_seeds"][index],
                "hand_origin": "own-hand-only structural proposal, unseen family",
                "classification": None,
                "teacher_evidence": "heuristic_only",
            }
            view, static, history, legal = encode(case)
            key = fingerprint(static, history, legal)
            if key in old_inputs or key in {x["input_sha256"] for x in evidence.values()}:
                rejected["exact_input"] += 1
                continue
            other, static2, history2, legal2 = encode(case, reverse=True)
            assert (
                np.array_equal(static, static2)
                and np.array_equal(history, history2)
                and np.array_equal(legal, legal2)
            )
            assert public_teacher_label(view) == public_teacher_label(other)
            assert view.phase is Phase.BID and view.current_high_bid == case["high_bid"]
            assert index >= 8 or view.current_high_bidder == 2
            suit, estimate = best_trump_candidate(view)
            case.update(
                actions=[59] + [int(a) for a in np.flatnonzero(legal) if a != 59],
                teacher=public_teacher_label(view),
                teacher_trump=suit,
                estimate=estimate,
                heuristic_strength=strength_bucket(estimate),
                suit_lengths=list(view.own_suit_lengths()),
            )
            cases.append(case)
            inputs[case["id"]] = {
                "observation": static.tolist(),
                "event_ids": history.tolist(),
                "event_count": len(history),
                "legal_action_mask": legal.tolist(),
            }
            evidence[case["id"]] = {
                "accepted_attempt": attempt,
                "rejected": dict(rejected),
                "input_sha256": key,
                "comparisons": distances,
                "hidden_placeholder_invariance": True,
                "canonical_suit_family": sorted(suit_masks(hand)),
                "exact_actor_observation_matches": counts["static_fingerprints"].get(
                    hashlib.sha256(static.tobytes()).hexdigest(), 0
                ),
                "public_history_only_matches": counts["public_history_fingerprints"].get(
                    history_hash(history), 0
                ),
                "history_note": "Shared public auction patterns are intentional; the full information set and hand family must be new.",
            }
            selected.append((case["id"], suit_masks(hand)))
            break
        else:
            raise RuntimeError(f"No qualifying hand for {category}/{slot}; do not relax gates")
    return {
        "cases": cases,
        "inputs": inputs,
        "overlap": evidence,
        "reference_counts": counts,
        "verified_seed_overlap": [],
        "statistical_results": None,
    }


def protected_files():
    paths = set()
    for directory in (
        "tarneeb",
        "scripts",
        "tests",
        "docs",
        "configs",
        "checkpoints",
        "results/phase11b",
    ):
        paths.update(
            p
            for p in Path(directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and ROOT not in p.parents
        )
    return sorted(paths)


def prepare():
    if ROOT.exists():
        raise FileExistsError("V3 output already exists; never overwrite")
    ROOT.mkdir()
    write(
        ROOT / "selection_preregistration.json",
        {
            "design": DESIGN,
            "source_hashes": {
                str(p): sha(p) for p in [Path(__file__), Path("tests/test_phase11b_controls_v3.py")]
            },
            "input_hashes": {str(p): sha(p) for p in [TRAIN, DEV, INPUTS, PLAN]},
        },
    )
    print("Selection rules pinned; hashing protected files.", flush=True)
    before = {str(p): sha(p) for p in protected_files()}
    write(ROOT / "preservation_before.json", before)
    with (
        patch.object(TarneebEngine, "deal", side_effect=AssertionError("No deals")),
        patch.object(audit, "sample_hands", side_effect=AssertionError("No hidden worlds")),
        patch.object(audit, "play_candidate", side_effect=AssertionError("No continuations")),
    ):
        result = select(load_references())
    for key, name in (
        ("cases", "cases.json"),
        ("inputs", "inputs.json"),
        ("overlap", "overlap.json"),
    ):
        write(ROOT / name, result[key])
    write(
        ROOT / "verification.json",
        {k: v for k, v in result.items() if k not in ("cases", "inputs", "overlap")},
    )
    print("16 cases prepared; verifying protected files.", flush=True)
    changed = [p for p, value in before.items() if sha(p) != value]
    write(ROOT / "preservation_after.json", {"checked": len(before), "changed": changed})
    if changed:
        raise AssertionError(changed)
    write(
        ROOT / "completion.json",
        {
            "status": "candidate_dataset_only",
            "cases": 16,
            "simulated_worlds": 0,
            "model_executions": 0,
            "hashes": {p.name: sha(p) for p in ROOT.iterdir() if p.is_file()},
        },
    )
    print(
        json.dumps(
            {"prepared": 16, "protected_files_unchanged": len(before), "evaluation_run": False}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.parse_args()
    prepare()
