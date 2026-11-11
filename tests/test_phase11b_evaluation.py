from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tarneeb.auction_repair.evaluation import compare_development, evaluate_development
from tarneeb.models.recurrent import RecurrentArchitectureSpec, build_recurrent_actor_and_critic


def test_development_pairing_metrics_and_identity_comparison() -> None:
    actor = build_recurrent_actor_and_critic(
        RecurrentArchitectureSpec(hidden_size=16, static_hidden_dim=24),
        seed=81,
    )[0]
    seeds = range(8_220_000, 8_220_002)
    results = compare_development(actor, actor, seeds)
    for anchor, result in results.items():
        assert result["before"] == result["after"]
        assert result["before"] == evaluate_development(
            actor, seeds, anchor, 8_300_010 + ("random", "basic", "strong").index(anchor)
        )
        assert result["paired_change_95ci"]["mean"] == 0
        assert result["paired_change_95ci"]["lower_bound"] == 0
        assert result["paired_change_95ci"]["upper_bound"] == 0
        assert len(result["before"]["paired_scores"]) == 2
        assert result["before"]["seed_start"] == 8_220_000


def test_smoke_command_refuses_training_without_explicit_flag(tmp_path: Path) -> None:
    checkpoint = tmp_path / "phase11b" / "smoke.pt"
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "scripts/train_phase11b_smoke.py",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "No training performed" in result.stderr
    assert not checkpoint.exists()
    assert not output.exists()
