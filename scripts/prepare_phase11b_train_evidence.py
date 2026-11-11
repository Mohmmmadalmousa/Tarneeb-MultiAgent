"""Select saved TRAIN candidates only. Default is read-only; no value labels yet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tarneeb.agents.heuristic_strong import best_trump_candidate
from tarneeb.auction_repair.artifacts import sha256, write_json_new
from tarneeb.auction_repair.data import encode_public_view, public_teacher_label, strength_bucket
from tarneeb.auction_value_audit import make_state
from tarneeb.core.actions import PASS_ACTION, action_for_bid
from tarneeb.core.cards import card_name, hand_from_cards
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.env.public_view import PublicViewBuilder

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "checkpoints/phase11/phase11_a3_pfsp_s0_final_u250.pt"
TRAIN = ROOT / "results/phase11b/auction_bc_smoke_s8300001_v1/train_examples.npz"
DEV = TRAIN.with_name("dev_examples.npz")
OUTPUT = ROOT / "results/phase11b/retention_train_evidence_v1/selection.json"
SOURCE_SHA = "317fd320fa33e98f435f1eb4b790b3ad5804635f59a6e86c73d8c35fda23ba7a"
TRAIN_SHA = "5d4cf535e7ec2f76a9da048d798d2ad11a96f95d65c43ac4a346be5646f3cfdf"
VERSION = "phase11b_retention_train_evidence_v1"
CATEGORIES = (
    ("BID10", "constructed", action_for_bid(10), None, 2),
    ("BID11", "constructed", action_for_bid(11), None, 2),
    ("BID12", "constructed", action_for_bid(12), None, 2),
    ("BID13", "constructed", action_for_bid(13), None, 2),
    ("PASS_WEAK", "natural", PASS_ACTION, "weak", 2),
    ("PASS_MEDIUM", "natural", PASS_ACTION, "medium", 1),
    ("PASS_STRONG", "natural", PASS_ACTION, "strong", 1),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def fingerprint(data, row):
    return hashlib.sha256(
        np.asarray(data["statics"][row], dtype="<f4").tobytes()
        + np.asarray(data["histories"][row], dtype="<i8").tobytes()
        + np.asarray([data["lengths"][row]], dtype="<i8").tobytes()
        + np.asarray(data["legal"][row], dtype=bool).tobytes()
    ).hexdigest()


def reconstruct(data, row, meta, hand, reverse=False):
    """Replay public bids with deterministic placeholders, never actual hidden cards."""
    actor = meta["seat"]
    unseen = sorted(set(range(52)) - set(hand), reverse=reverse)
    hands = [0] * 4
    hands[actor] = hand_from_cards(hand)
    for index, seat in enumerate(s for s in range(4) if s != actor):
        hands[seat] = hand_from_cards(unseen[index * 13 : (index + 1) * 13])
    engine, state = make_state({"dealer": meta["seed"] % 4, "world_seed": meta["seed"]}, hands)
    for action in meta["prefix"]:
        require(state.phase is Phase.BID, "Prefix left auction phase")
        engine.step(state, action)
    require(state.phase is Phase.BID and state.current_player_seat == actor, "Actor mismatch")
    view = PublicViewBuilder(engine).build(state, actor)
    static, history, length = encode_public_view(view)
    for key, actual in (
        ("statics", static),
        ("histories", history),
        ("lengths", length),
        ("legal", view.legal_action_mask),
        ("labels", public_teacher_label(view)),
    ):
        require(np.array_equal(data[key][row], actual), f"Row {row}: {key} mismatch")
    suit, estimate = best_trump_candidate(view)
    expected = {
        "learner_team": actor % 2,
        "current_high_bid": view.current_high_bid,
        "high_bidder_is_partner": view.current_high_bidder == view.partner_seat,
        "estimate": estimate,
        "strength": strength_bucket(estimate),
        "suit_lengths": list(view.own_suit_lengths()),
        "best_suit": suit,
    }
    require(all(meta[k] == v for k, v in expected.items()), f"Row {row}: metadata mismatch")
    require(
        view.current_high_bid == 9 and view.current_high_bidder == view.partner_seat,
        f"Row {row}: not partner BID 9",
    )
    return view.current_high_bidder


def build_selection():
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in (SOURCE, TRAIN, DEV)}
    require(hashes[str(SOURCE.relative_to(ROOT))] == SOURCE_SHA, "Checkpoint SHA256 mismatch")
    require(hashes[str(TRAIN.relative_to(ROOT))] == TRAIN_SHA, "TRAIN SHA256 mismatch")
    with np.load(TRAIN, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(data["metadata_json"]))
    with np.load(DEV, allow_pickle=False) as archive:
        dev_metadata = json.loads(str(archive["metadata_json"]))
    train_seeds = {m["seed"] for m in metadata}
    dev_seeds = {m["seed"] for m in dev_metadata}
    future_seeds = set(range(8910000, 8910012)) | set(range(8920000, 8920012))
    require(train_seeds.isdisjoint(dev_seeds), "TRAIN/DEV seeds overlap")
    require(20250910 not in train_seeds | dev_seeds | future_seeds, "Sealed seed present")
    require(future_seeds.isdisjoint(train_seeds | dev_seeds), "Future audit seeds overlap")
    cases, counts, used_rows, used_hands = [], {}, set(), set()
    for category, family, label, strength, count in CATEGORIES:
        candidates = []
        for row, meta in enumerate(metadata):
            if (
                meta["case_family"] == family
                and meta["current_high_bid"] == 9
                and meta["high_bidder_is_partner"]
                and data["labels"][row] == label
                and (strength is None or meta["strength"] == strength)
            ):
                rank = digest([VERSION, category, row, fingerprint(data, row)])
                candidates.append((rank, row))
        counts[category] = len(candidates)
        selected = 0
        for rank, row in sorted(candidates):
            hand = tuple(np.flatnonzero(data["statics"][row, :52]).tolist())
            if row in used_rows or hand in used_hands:
                continue
            require(
                len(hand) == 13 and np.isin(data["statics"][row, :52], [0, 1]).all(),
                f"Row {row}: invalid own hand",
            )
            meta = metadata[row]
            owner = reconstruct(data, row, meta, hand)
            reconstruct(data, row, meta, hand, reverse=True)
            selected += 1
            index = len(cases)
            cases.append(
                {
                    "id": f"TRAIN_{category}_{selected:02d}",
                    "training_row_id": row,
                    "category": category,
                    "metadata": meta,
                    "ranking_sha256": rank,
                    "observation_history_mask_sha256": fingerprint(data, row),
                    "hand_sha256": digest(hand),
                    "own_hand": list(hand),
                    "hand_names": [card_name(card) for card in hand],
                    "dealer": meta["seed"] % 4,
                    "actor": meta["seat"],
                    "current_high_bidder": owner,
                    "public_auction_prefix": meta["prefix"],
                    "history_length": int(data["lengths"][row]),
                    "public_history_ids": data["histories"][row, : data["lengths"][row]].tolist(),
                    "legal_actions": np.flatnonzero(data["legal"][row]).tolist(),
                    "teacher_action": int(label),
                    "world_seed": 8910000 + index,
                    "bootstrap_seed": 8920000 + index,
                    "evidence_status": "heuristic_only_pending_counterfactual_audit",
                    "reconstruction_exact": True,
                    "hidden_placeholder_invariance": True,
                }
            )
            used_rows.add(row)
            used_hands.add(hand)
            if selected == count:
                break
        require(selected == count, f"Insufficient distinct candidates for {category}")
    require(len(cases) == len(used_rows) == len(used_hands) == 12, "Duplicate/missing cases")
    require(all(sha256(ROOT / p) == h for p, h in hashes.items()), "Input changed during selection")
    return {
        "version": VERSION,
        "row_numbering": "zero-based original TRAIN NPZ rows",
        "input_sha256": hashes,
        "selector_source_sha256": sha256(Path(__file__)),
        "ranking": "Ascending SHA256 of compact sorted-key JSON [version, category, row_id, observation_history_mask_sha256]; row ID breaks ties. Fixed category order; skip globally duplicate exact hands/rows.",
        "fingerprint_encoding": "Full static <f4 + full padded history <i8 + history length <i8 + legal mask bool, concatenated bytes",
        "candidate_counts": counts,
        "train_seeds": sorted(train_seeds),
        "dev_seeds": sorted(dev_seeds),
        "seed_checks_passed": True,
        "scope": "TRAIN-only selection; DEV metadata used only for seed separation. No V3 data accessed. Distinct exact hands, not independent hand families; constructed suit rotations can be related. No claim of unseen training examples.",
        "information_boundary": "Saved actor hand and public bids only; sorted and reversed complementary placeholders give identical saved observations, histories, masks and teacher labels. No hidden-card sampling or gameplay.",
        "evidence_supported_pairs": 0,
        "future_audit": "Seeds reserved only; no simulation executed or optimal labels assigned.",
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare and OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    with patch.object(TarneebEngine, "deal", side_effect=RuntimeError("Dealing forbidden")):
        manifest = build_selection()
    if args.prepare:
        write_json_new(OUTPUT, manifest)
        require(json.loads(OUTPUT.read_text(encoding="utf-8")) == manifest, "Read-back mismatch")
    print(
        json.dumps(
            {
                "status": "PASS",
                "mode": "prepare" if args.prepare else "read-only preview",
                "output": str(OUTPUT) if args.prepare else None,
                "candidate_counts": manifest["candidate_counts"],
                "cases": [{"id": c["id"], "row": c["training_row_id"]} for c in manifest["cases"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
