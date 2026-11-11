"""Read-only Phase 11B retention objective integration preflight.

No optimizer, training, sampling, game simulation, or checkpoint export.
Only tests the objective's compatibility with the SAVED training archive and
existing frozen PFSP checkpoint. Does not use DEV/V3 for learning labels.
Run from C:\\dev\\tarneeb-rl using .venv\\Scripts\\python.exe -B.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tarneeb.auction_repair.data import PublicExamples, frozen_features
from tarneeb.auction_repair.retention_objective import loss_terms, retention_masks
from tarneeb.auction_repair.train_evidence import fingerprints_for, load_train_evidence
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "checkpoints/phase11/phase11_a3_pfsp_s0_final_u250.pt"
SOURCE_SHA256 = "317fd320fa33e98f435f1eb4b790b3ad5804635f59a6e86c73d8c35fda23ba7a"
TRAIN = ROOT / "results/phase11b/auction_bc_smoke_s8300001_v1/train_examples.npz"
DEV = ROOT / "results/phase11b/auction_bc_smoke_s8300001_v1/dev_examples.npz"
SEALED_SEED = 20250910


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def examples(path: Path) -> PublicExamples:
    with np.load(path, allow_pickle=False) as d:
        return PublicExamples(
            *(d[k].copy() for k in ("statics", "histories", "lengths", "legal", "labels")),
            tuple(json.loads(str(d["metadata_json"]))),
        )


def main() -> None:
    torch.set_num_threads(4)
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("PFSP source checkpoint hash mismatch; stopping")
    source_hash = sha256(SOURCE)
    training = examples(TRAIN)
    development = examples(DEV)
    evidence = load_train_evidence()
    fingerprints = fingerprints_for(training)
    if SEALED_SEED in {m["seed"] for m in (*training.metadata, *development.metadata)}:
        raise RuntimeError("Sealed seed found in train/development examples")
    if not {m["seed"] for m in training.metadata}.isdisjoint(
        {m["seed"] for m in development.metadata}
    ):
        raise RuntimeError("TRAIN and DEV seeds overlap")
    if not bool(training.legal[np.arange(len(training)), training.labels].all()):
        raise RuntimeError("Illegal TRAIN target")
    keep = np.flatnonzero(training.legal.sum(axis=1) > 1)
    assert len(keep) > 0

    with torch.no_grad():
        actor = (
            load_recurrent_checkpoint(SOURCE, load_critic=False)
            .actor.cpu()
            .eval()
            .requires_grad_(False)
        )
        features, _ = frozen_features(actor, training)
        baseline = actor.policy_head(features).detach()
    mask = torch.from_numpy(training.legal[keep]).bool()
    labels = torch.from_numpy(training.labels[keep]).long()
    orig = baseline[keep]
    bid_anchor, pass_anchor = retention_masks(orig, mask, labels)
    candidate = orig.clone().detach().requires_grad_(True)
    terms = loss_terms(
        candidate,
        orig,
        mask,
        labels,
        evidence=evidence,
        row_ids=torch.from_numpy(keep),
        row_fingerprints=[fingerprints[i] for i in keep],
    )
    assert terms["supported_pair_count"] == 30 and terms["supported_case_count"] == 8
    assert terms["heuristic_excluded_count"] == 8
    affected = {p.row for p in evidence.pairs}
    untouched = torch.tensor([i not in affected for i in keep])
    assert terms["bid_anchor_count"] == bid_anchor[untouched].sum()
    assert terms["pass_anchor_count"] == pass_anchor[untouched].sum()
    # Same rows in a different minibatch order must retain the same evidence.
    selected = [int(np.flatnonzero(keep == row)[0]) for row in sorted(affected, reverse=True)]
    mini = loss_terms(
        candidate[selected],
        orig[selected],
        mask[selected],
        labels[selected],
        evidence=evidence,
        row_ids=torch.from_numpy(keep[selected]),
        row_fingerprints=[fingerprints[i] for i in keep[selected]],
    )
    assert mini["supported_pair_count"] == 30 and mini["heuristic_ce"] == 0
    assert torch.allclose(mini["supported_pair"], terms["supported_pair"])
    assert mini["bid_retention"] == 0 and mini["pass_retention"] == 0
    terms["total"].backward()
    assert candidate.grad is not None and bool(torch.isfinite(candidate.grad).all())
    assert not bool(candidate.grad[~mask].any())
    assert torch.equal(candidate.detach(), orig)
    assert all(p.grad is None for p in actor.parameters())
    assert source_hash == sha256(SOURCE), "Checkpoint was modified during preflight"

    print("PASS — real-data objective compatibility (READ-ONLY)")
    print("TRAIN examples:", len(training), "nontrivial:", len(keep))
    print("DEV examples (not used in objective):", len(development))
    print(
        "TRAIN target counts (action IDs):", dict(sorted(Counter(training.labels.tolist()).items()))
    )
    print("Baseline bid anchors:", int(bid_anchor.sum().item()))
    print("Baseline PASS anchors:", int(pass_anchor.sum().item()))
    print("Unaudited-row heuristic CE:", round(float(terms["heuristic_ce"].item()), 6))
    print(
        "Authenticated TRAIN comparisons:",
        int(terms["supported_pair_count"].item()),
        "across",
        int(terms["supported_case_count"].item()),
        "rows",
    )
    print(
        "Conflicting teacher/retention targets suppressed on 8 evidence rows; unaffected anchors preserved."
    )
    print("Reordered minibatch identity and equal-per-case pair loss: PASS")
    print("Checkpoint hash unchanged:", source_hash)
    print("NO optimization, checkpoint writes, hidden worlds, or gameplay executed.")


if __name__ == "__main__":
    main()
