"""Phase 9 seed splits, and the proof that they do not overlap.

Every Phase 9 seed range lives inside the frozen ``TRAINING_SEEDS``
(``range(0, 10**8)``), which Phase 5 proved disjoint from
``HELD_OUT_EVALUATION_SEEDS`` (``range(10**9, 10**9 + 100_000)``). So no amount
of Phase 9 training or tuning can reach a deal the final evaluation will report
on -- that separation is structural, not a matter of discipline.

Within the training range the splits are further separated by purpose. The
reason each one exists:

``BC_TRAIN`` / ``BC_VALIDATION`` / ``BC_TEST``
    Cloning accuracy measured on the deals it was fitted to means nothing. The
    validation split steers nothing but is watched during training; the test
    split is touched once, at the end.

``PPO_TRAIN``
    Disjoint from the BC splits so that "the policy plays well" cannot be a
    restatement of "the policy memorised its demonstrations".

``PPO_DEVELOPMENT``
    Every mid-training evaluation, checkpoint comparison and sanity check draws
    from here. This is what protects the held-out result: a checkpoint chosen by
    repeatedly consulting the final seeds would have its interval quietly
    invalidated.

``FINAL_HELD_OUT_SEED``
    Used once, after selection is frozen. It folds into
    ``HELD_OUT_EVALUATION_SEEDS`` through the frozen arena helper.

A1's matched feedforward condition uses the **same** ranges, so the two
architectures see the same deals.
"""

from __future__ import annotations

from tarneeb.evaluation.arena import HELD_OUT_EVALUATION_SEEDS, TRAINING_SEEDS

BC_TRAIN_SEEDS = range(1_000_000, 1_040_000)
"""Demonstration deals the cloned policy is fitted to."""

BC_VALIDATION_SEEDS = range(1_040_000, 1_045_000)
"""Watched during cloning; never fitted to."""

BC_TEST_SEEDS = range(1_045_000, 1_050_000)
"""Touched once, to report final cloning accuracy."""

PPO_TRAIN_SEEDS = range(2_000_000, 3_000_000)
"""Environment root seeds for recurrent PPO. Disjoint from every BC split."""

PPO_DEVELOPMENT_SEEDS = range(4_000_000, 4_010_000)
"""Mid-training evaluation and checkpoint comparison."""

FINAL_HELD_OUT_SEED = 20250910
"""The frozen final evaluation seed, folded into ``HELD_OUT_EVALUATION_SEEDS``."""

PHASE9_SPLITS: dict[str, range] = {
    "bc_train": BC_TRAIN_SEEDS,
    "bc_validation": BC_VALIDATION_SEEDS,
    "bc_test": BC_TEST_SEEDS,
    "ppo_train": PPO_TRAIN_SEEDS,
    "ppo_development": PPO_DEVELOPMENT_SEEDS,
}


def ranges_overlap(first: range, second: range) -> bool:
    """Return True when two ranges share at least one value."""
    return first.start < second.stop and second.start < first.stop


def splits_are_disjoint() -> bool:
    """Return True when no two Phase 9 splits share a seed."""
    names = sorted(PHASE9_SPLITS)
    return all(
        not ranges_overlap(PHASE9_SPLITS[left], PHASE9_SPLITS[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    )


def splits_are_inside_training_range() -> bool:
    """Return True when every Phase 9 split lies inside the frozen training range."""
    return all(
        TRAINING_SEEDS.start <= span.start and span.stop <= TRAINING_SEEDS.stop
        for span in PHASE9_SPLITS.values()
    )


def splits_avoid_held_out_range() -> bool:
    """Return True when no Phase 9 split reaches the held-out evaluation range."""
    return all(
        not ranges_overlap(span, HELD_OUT_EVALUATION_SEEDS) for span in PHASE9_SPLITS.values()
    )


def describe_splits() -> dict[str, dict[str, int]]:
    """Return a JSON-safe description, for the Phase 9 decision record."""
    return {
        name: {"start": span.start, "stop": span.stop, "size": len(span)}
        for name, span in PHASE9_SPLITS.items()
    }


def training_environment_seed(seed_index: int) -> int:
    """Return the PPO environment root seed for one independent training seed.

    Spread widely inside ``PPO_TRAIN_SEEDS`` so two runs cannot deal overlapping
    hands early, which would make "independent seeds" a weaker claim than it
    sounds.
    """
    if seed_index < 0:
        raise ValueError(f"seed_index must be non-negative, got {seed_index}")
    stride = 100_000
    offset = seed_index * stride
    if offset >= len(PPO_TRAIN_SEEDS):
        raise ValueError(f"seed_index {seed_index} falls outside PPO_TRAIN_SEEDS")
    return PPO_TRAIN_SEEDS.start + offset
