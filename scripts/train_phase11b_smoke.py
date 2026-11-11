"""Proposed bounded auction-row BC experiment. Requires explicit execution flag.

This command is prepared for review; implementing/testing it does not authorize
running it on project checkpoints. No Phase 11 PPO or league training is called.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tarneeb.auction_repair.artifacts import (
    provenance,
    save_examples_new,
    save_repair_new,
    sha256,
    write_json_new,
)
from tarneeb.auction_repair.data import (
    DATA_VERSION,
    TEACHER_VERSION,
    generate_auction_examples,
    generate_play_examples,
)
from tarneeb.auction_repair.evaluation import compare_development
from tarneeb.auction_repair.learning import AuctionBCConfig, fit_auction_rows, preservation_report
from tarneeb.auction_repair.measurement import measure_actor
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint

SOURCE_SHA256 = "317fd320fa33e98f435f1eb4b790b3ad5804635f59a6e86c73d8c35fda23ba7a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument(
        "--source", type=Path, default=Path("checkpoints/phase11/phase11_a3_pfsp_s0_final_u250.pt")
    )
    parser.add_argument("--source-sha256", default=SOURCE_SHA256)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not args.execute_training:
        parser.error("No training performed. Obtain approval, then pass --execute-training.")
    if args.checkpoint.exists() or args.output_dir.exists():
        raise FileExistsError("All Phase 11B output paths must be new")
    if "phase11b" not in args.checkpoint.parts:
        raise ValueError("Use a dedicated checkpoints/phase11b path")
    if sha256(args.source) != args.source_sha256:
        raise ValueError("Initialization checkpoint provenance mismatch")
    if args.threads < 1 or args.threads > 8:
        raise ValueError("Use 1..8 CPU threads")
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    parent = load_recurrent_checkpoint(args.source, load_critic=False)
    train = generate_auction_examples(range(8_100_000, 8_100_064), include_stress=True)
    # Natural DEV is independent. Constructed training coverage is not reused
    # as a claimed validation sample.
    dev = generate_auction_examples(range(8_200_000, 8_200_128))
    fixture = generate_play_examples(range(8_210_000, 8_210_008))
    baseline = preservation_report(parent.actor, parent.actor, fixture)
    before = measure_actor(parent.actor, dev)
    save_examples_new(args.output_dir / "train_examples.npz", train)
    save_examples_new(args.output_dir / "dev_examples.npz", dev)
    write_json_new(
        args.output_dir / "plan.json",
        {
            "source": provenance(args.source, parent),
            "teacher": TEACHER_VERSION,
            "dataset": DATA_VERSION,
            "train_count": len(train),
            "dev_count": len(dev),
            "train_seeds": [8_100_000, 8_100_064],
            "dev_seeds": [8_200_000, 8_200_128],
            "play_fixture_seeds": [8_210_000, 8_210_008],
            "evaluation_seeds": [8_220_000, 8_220_100],
            "baseline_preservation": baseline,
        },
    )
    repaired, learning = fit_auction_rows(parent.actor, train, AuctionBCConfig())
    preservation = preservation_report(parent.actor, repaired, fixture)
    if (
        preservation["deterministic_action_agreement"] != 1.0
        or preservation["max_legal_logit_drift"] != 0.0
        or preservation["max_masked_log_probability_drift"] != 0.0
        or not preservation["hidden_state_bit_equal"]
    ):
        raise ValueError("Card/trump preservation gate failed; no checkpoint exported")
    after = measure_actor(repaired, dev)

    def teacher_nll(measurement: dict[str, Any]) -> float:
        values = []
        for row in measurement["rows"]:
            probability = (
                row["pass_probability"]
                if row["teacher_action"] == 59
                else row["legal_bid_probabilities"][str(row["teacher_action"] - 45)]
            )
            values.append(-np.log(max(probability, np.finfo(np.float32).tiny)))
        return float(np.mean(values))

    report = {
        **learning,
        "teacher": TEACHER_VERSION,
        "dataset": DATA_VERSION,
        "train_count": len(train),
        "dev_count": len(dev),
        "train_sha256": sha256(args.output_dir / "train_examples.npz"),
        "dev_sha256": sha256(args.output_dir / "dev_examples.npz"),
        "preservation": preservation,
        "dev_before": before,
        "dev_after": after,
        "dev_nll_before": teacher_nll(before),
        "dev_nll_after": teacher_nll(after),
        "evaluation": compare_development(parent.actor, repaired, range(8_220_000, 8_220_100)),
        "wall_seconds": time.perf_counter() - start,
        "status": "development smoke only; no promotion or strength claim",
    }
    report["implementation_sha256"] = {
        str(path): sha256(path) for path in sorted(Path("tarneeb/auction_repair").glob("*.py"))
    }
    report["implementation_sha256"][str(Path(__file__))] = sha256(Path(__file__))
    report["auction_learning_gate_passed"] = (
        learning["epoch_nll"][-1] < learning["initial_nll"]
        and report["dev_nll_after"] < report["dev_nll_before"]
    )
    save_repair_new(args.checkpoint, args.source, parent, repaired, report)
    write_json_new(args.output_dir / "report.json", report)
    print(
        {
            "checkpoint": str(args.checkpoint),
            "report": str(args.output_dir / "report.json"),
            "wall_seconds": report["wall_seconds"],
            "preservation": preservation,
        }
    )


if __name__ == "__main__":
    main()
