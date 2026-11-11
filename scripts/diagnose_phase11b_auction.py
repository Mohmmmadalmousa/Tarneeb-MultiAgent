"""Read-only model diagnosis and new regression artifacts. Never trains."""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

import torch

from tarneeb.auction_repair.artifacts import provenance, save_examples_new, sha256, write_json_new
from tarneeb.auction_repair.data import (
    frozen_features,
    generate_auction_examples,
    generate_play_examples,
)
from tarneeb.auction_repair.evaluation import evaluate_development
from tarneeb.auction_repair.learning import preservation_report
from tarneeb.auction_repair.measurement import measure_actor, training_seating_diagnostic
from tarneeb.buffers.bc import BCDataset
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("checkpoints/phase11/phase11_a3_pfsp_s0_final_u250.pt"),
    )
    parser.add_argument(
        "--latest",
        type=Path,
        default=Path("checkpoints/phase11/phase11_a3_latest_only_s0_final_u250.pt"),
    )
    parser.add_argument(
        "--earlier", type=Path, default=Path("checkpoints/phase9_gru_bc_s0_update250.pt")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=8_200_000)
    parser.add_argument("--deal-count", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--evaluation-pairs", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new diagnostic output directory")
    if not 1 <= args.deal_count <= 256 or not 1 <= args.evaluation_pairs <= 16:
        raise ValueError("Read-only diagnosis is bounded to 256 probe deals and 16 pairs/anchor")
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    examples = generate_auction_examples(
        range(args.seed_base, args.seed_base + args.deal_count), include_stress=True
    )
    fixture = generate_play_examples(range(8_210_000, 8_210_008))
    paths = {"pfsp_s0": args.candidate, "latest_s0": args.latest, "phase9_s0": args.earlier}
    loaded = {
        name: load_recurrent_checkpoint(path, load_critic=False) for name, path in paths.items()
    }
    report = {
        "seed_start": args.seed_base,
        "seed_stop": args.seed_base + args.deal_count,
        "count": len(examples),
        "bid13": "PASS-only until one active bidder remains; no legal raise above 13",
        "root_cause": "ROOT CAUSE NOT YET ISOLATED",
        "training_seating": training_seating_diagnostic(),
        "provenance": {name: provenance(paths[name], value) for name, value in loaded.items()},
        "measurements": {
            name: measure_actor(value.actor, examples) for name, value in loaded.items()
        },
    }
    swaps = {}
    for base, head in (("phase9_s0", "pfsp_s0"), ("pfsp_s0", "phase9_s0")):
        actor = deepcopy(loaded[base].actor)
        actor.policy_head.load_state_dict(loaded[head].actor.policy_head.state_dict())
        swaps[f"{base}_trunk_{head}_head"] = measure_actor(actor, examples)["overall"]
    report["component_substitution"] = swaps
    candidate = loaded["pfsp_s0"].actor
    features, hidden = frozen_features(candidate, fixture)
    with torch.no_grad():
        logits = candidate.policy_head(features).numpy()
    save_examples_new(
        args.output_dir / "play_fixture.npz",
        fixture,
        baseline_logits=logits,
        baseline_hidden=hidden.numpy(),
        source_sha256=sha256(args.candidate),
    )
    save_examples_new(args.output_dir / "auction_probe.npz", examples)
    report["preservation_baseline"] = preservation_report(candidate, candidate, fixture)
    report["fixture_sha256"] = sha256(args.output_dir / "play_fixture.npz")
    # Count original Phase 9 demonstration labels directly from saved actions.
    # Stop at the first trump/card action: no reconstruction or label inference.
    demonstrations = BCDataset.load(Path("data/bc/bc_train.npz"))
    label_counts = dict.fromkeys(range(52, 60), 0)
    opening_decisions = opening_bids = competitive_decisions = raises = 0
    for deal in demonstrations.deals:
        high = None
        for action in deal.teacher_actions:
            if not 52 <= action <= 59:
                break
            label_counts[action] += 1
            if high is None:
                opening_decisions += 1
                opening_bids += int(action != 59)
            elif high < 13:
                competitive_decisions += 1
                raises += int(action != 59)
            if action != 59:
                high = action - 45
    report["phase9_bc_training_labels"] = {
        "dataset_sha256": sha256(Path("data/bc/bc_train.npz")),
        "deals": len(demonstrations.deals),
        "label_counts": label_counts,
        "opening_decisions": opening_decisions,
        "opening_bids": opening_bids,
        "raise_opportunities": competitive_decisions,
        "raises": raises,
    }
    evaluation_start = time.perf_counter()
    report["bounded_development_play"] = {
        anchor: evaluate_development(
            candidate,
            range(8_230_000, 8_230_000 + args.evaluation_pairs),
            anchor,
            8_300_020 + index,
        )
        for index, anchor in enumerate(("random", "basic", "strong"))
    }
    report["development_play_seconds"] = time.perf_counter() - evaluation_start
    report["wall_seconds"] = time.perf_counter() - start
    write_json_new(args.output_dir / "diagnosis.json", report)
    print(
        {
            "output": str(args.output_dir),
            "count": len(examples),
            "wall_seconds": report["wall_seconds"],
            "overall": {name: result["overall"] for name, result in report["measurements"].items()},
        }
    )


if __name__ == "__main__":
    main()
