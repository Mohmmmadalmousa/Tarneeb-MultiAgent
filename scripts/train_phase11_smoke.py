"""Run tiny Phase 11 PPO smoke jobs; never the principal A3 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tarneeb.league.population import LeagueConfig, LeaguePopulation
from tarneeb.league.registry import PolicyRegistry
from tarneeb.league.sampling import PFSPConfig
from tarneeb.training.phase11_training import (
    PHASE11_CONDITIONS,
    Phase11SmokeConfig,
    run_phase11_smoke,
    write_phase11_smoke_report,
)
from tarneeb.training.ppo import PPOConfig

BC_CHECKPOINT = Path("checkpoints/phase9_bc_gru.pt")
CHAMPION = (
    "phase10_centralized_s0",
    Path("checkpoints/phase10_centralized_s0_update250.pt"),
    "phase10",
    0,
    250,
)
RECENT = (
    ("phase9_gru_bc_s0", Path("checkpoints/phase9_gru_bc_s0_update250.pt"), "phase9", 0, 250),
    ("phase9_gru_bc_s1", Path("checkpoints/phase9_gru_bc_s1_update250.pt"), "phase9", 1, 250),
    ("phase9_gru_bc_s2", Path("checkpoints/phase9_gru_bc_s2_update250.pt"), "phase9", 2, 250),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=PHASE11_CONDITIONS,
        default=list(PHASE11_CONDITIONS),
    )
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--environment-count", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=8)
    parser.add_argument("--seed-index", type=int, default=0)
    return parser.parse_args()


def build_smoke_population(seed: int) -> LeaguePopulation:
    required_paths = [CHAMPION[1], *(entry[1] for entry in RECENT), BC_CHECKPOINT]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Phase 11 smoke checkpoint(s) missing: " + ", ".join(missing))

    registry = PolicyRegistry()
    champion_id = registry.register_checkpoint(
        policy_id=CHAMPION[0],
        checkpoint_path=CHAMPION[1],
        source_phase=CHAMPION[2],
        category="champion",
        seed=CHAMPION[3],
        update=CHAMPION[4],
    )
    recent_ids: list[str] = []
    for policy_id, path, phase, seed_index, update in RECENT:
        registered = registry.register_checkpoint(
            policy_id=policy_id,
            checkpoint_path=path,
            source_phase=phase,
            category="snapshot",
            seed=seed_index,
            update=update,
        )
        if registered not in recent_ids and registered != champion_id:
            recent_ids.append(registered)
    registry.register_anchor("strong", "strong")
    registry.register_anchor("random", "random")

    pfsp = PFSPConfig(
        ema_beta=0.2,
        unobserved_learner_win_rate=0.5,
        empty_category_strategy="proportional",
    )
    population = LeaguePopulation(
        registry=registry,
        config=LeagueConfig(
            pfsp=pfsp,
            # Smoke-only acceleration so Part 3 exercises dynamic snapshots.
            snapshot_updates=1,
            snapshot_environment_steps=10**9,
        ),
        rng=np.random.default_rng(seed),
    )
    for policy_id in recent_ids:
        population.add(policy_id, make_latest=False)
    population.add(champion_id, make_latest=True)
    population.add("strong", make_latest=False)
    population.add("random", make_latest=False)
    return population


def main() -> None:
    arguments = parse_arguments()
    if arguments.updates < 1:
        raise ValueError("--updates must be >= 1")
    if arguments.seed_index < 0:
        raise ValueError("--seed-index must be non-negative")
    if not BC_CHECKPOINT.is_file():
        raise FileNotFoundError(str(BC_CHECKPOINT))

    torch.set_num_threads(8)
    for condition_index, condition in enumerate(arguments.conditions):
        population = build_smoke_population(
            seed=7_650_001 + arguments.seed_index * 10 + condition_index
        )
        config = Phase11SmokeConfig(
            run_name=f"phase11_{condition}_smoke_s{arguments.seed_index}",
            condition=condition,
            environment_count=arguments.environment_count,
            rollout_length=arguments.rollout_length,
            total_updates=arguments.updates,
            bc_checkpoint=BC_CHECKPOINT,
            seed_index=arguments.seed_index,
            initialization_seed=1,
            action_sampling_seed=7_650_101 + arguments.seed_index,
            minibatch_shuffle_seed=7_650_201 + arguments.seed_index,
            opponent_seed=7_650_301 + arguments.seed_index,
            ppo=PPOConfig(),
        )
        result = run_phase11_smoke(config, population, progress=True)
        report = write_phase11_smoke_report(config, result, population)
        print(
            f"[{condition}] smoke complete: updates={result.completed_updates}, "
            f"deals={result.completed_deals}, illegal={result.illegal_action_count}, "
            f"action64={result.no_trump_action_count}, "
            f"e0={result.worst_epoch_zero_deviation:.2e} -> {report}",
            flush=True,
        )


if __name__ == "__main__":
    main()
