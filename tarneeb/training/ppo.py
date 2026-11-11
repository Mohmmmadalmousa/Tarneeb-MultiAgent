"""The clipped PPO update for the feedforward masked baseline.

One function, :func:`ppo_update`, consumes a finished rollout and returns
diagnostics as detached Python floats. It owns no logger, no writer and no UI,
and removing every caller's logging would not change a single gradient.

The objective is the standard one::

    ratio_t          = exp(log pi_new(a_t | o_t) - log pi_old(a_t | o_t))
    policy_loss      = -mean( min(ratio_t * A_t, clip(ratio_t, 1-eps, 1+eps) * A_t) )
    value_loss       = mean( (V(o_t) - R_t)^2 )                # no value clipping
    entropy_bonus    = mean( H[pi_new(. | o_t)] )              # masked distribution only
    total_loss       = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus

Advantages are normalized per minibatch. Gradients are clipped by global norm
across both networks, and the pre-clip norm is reported.

**Epoch-zero identity.** On the first minibatch of the first epoch the policy has
not been updated yet, so the recomputed log-probabilities must reproduce the
stored ones and the ratio must be exactly 1. For a feedforward policy this is an
algebraic identity, and any deviation means the replayed inputs do not match the
inputs used during collection. The maximum deviation is measured every update and
must stay below :data:`EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from tarneeb.buffers.rollout import RolloutBatch
from tarneeb.models.masking import MaskedCategorical
from tarneeb.models.nets import ActorNetwork, CriticNetwork

EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE = 1e-4
"""Maximum tolerated epoch-zero log-probability deviation before a run is suspect."""


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Optimization hyper-parameters.

    ``discount_factor`` is 1.0 on purpose. One episode is one deal, the reward
    arrives once at the end, and discounting it would systematically starve the
    auction — the earliest and most consequential decisions — of signal.
    """

    learning_rate: float = 3e-4
    anneal_learning_rate: bool = True
    discount_factor: float = 1.0
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_function_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    update_epochs: int = 4
    minibatch_count: int = 4
    target_kl_divergence: float | None = 0.02
    adam_epsilon: float = 1e-5
    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        if not self.learning_rate > 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must lie in (0, 1)")
        if self.update_epochs < 1 or self.minibatch_count < 1:
            raise ValueError("update_epochs and minibatch_count must be at least 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UpdateMetrics:
    """PPO diagnostics for one update. Every field is a detached scalar."""

    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl_divergence: float
    clip_fraction: float
    explained_variance: float
    gradient_norm: float
    learning_rate: float
    epoch_zero_log_probability_max_deviation: float
    optimizer_steps: int
    early_stopped: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def build_optimizer(
    actor: ActorNetwork, critic: CriticNetwork, config: PPOConfig
) -> torch.optim.Optimizer:
    """Return one Adam optimizer over both networks' parameters.

    A single optimizer is safe precisely because the parameter sets are disjoint:
    the two loss terms reach only their own network.
    """
    return torch.optim.Adam(
        [*actor.parameters(), *critic.parameters()],
        lr=config.learning_rate,
        eps=config.adam_epsilon,
    )


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Set the learning rate on every parameter group."""
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def explained_variance(predicted_values: Tensor, returns: Tensor) -> float:
    """Return ``1 - Var(returns - predicted) / Var(returns)``.

    Below about 0.1 the critic explains nothing and the advantages are noise.
    Returns 0.0 when the returns have no variance at all, which is the only
    honest answer: nothing was there to explain.
    """
    return_variance = float(torch.var(returns))
    if return_variance == 0.0:
        return 0.0
    return 1.0 - float(torch.var(returns - predicted_values)) / return_variance


def _minibatch_index_batches(
    decision_count: int, minibatch_count: int, shuffle_generator: np.random.Generator
) -> list[np.ndarray]:
    """Split a shuffled index permutation into roughly equal minibatches."""
    permutation = shuffle_generator.permutation(decision_count)
    return [batch for batch in np.array_split(permutation, minibatch_count) if batch.size > 0]


def ppo_update(
    actor: ActorNetwork,
    critic: CriticNetwork,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    config: PPOConfig,
    shuffle_generator: np.random.Generator,
    learning_rate: float | None = None,
) -> UpdateMetrics:
    """Run the configured PPO epochs over ``batch`` and return diagnostics."""
    if batch.decision_count == 0:
        raise ValueError("cannot update on an empty rollout")

    effective_learning_rate = (
        config.learning_rate if learning_rate is None else float(learning_rate)
    )
    set_learning_rate(optimizer, effective_learning_rate)

    actor.train()
    critic.train()

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    kl_divergences: list[float] = []
    clip_fractions: list[float] = []
    gradient_norms: list[float] = []
    epoch_zero_max_deviation = 0.0
    optimizer_steps = 0
    early_stopped = False

    for epoch_index in range(config.update_epochs):
        for minibatch_position, minibatch_indices in enumerate(
            _minibatch_index_batches(
                batch.decision_count, config.minibatch_count, shuffle_generator
            )
        ):
            indices = torch.from_numpy(minibatch_indices)
            observations = batch.observations[indices]
            history_ids = batch.history_ids[indices]
            history_masks = batch.history_masks[indices]
            legal_action_masks = batch.legal_action_masks[indices]
            selected_actions = batch.selected_actions[indices]
            stored_log_probabilities = batch.action_log_probabilities[indices]
            advantages = batch.advantages[indices]
            returns = batch.returns[indices]

            logits = actor(observations, history_ids, history_masks)
            distribution = MaskedCategorical.from_logits(logits, legal_action_masks)
            recomputed_log_probabilities = distribution.log_probability_of(selected_actions)
            entropy_bonus = distribution.entropy().mean()

            log_ratio = recomputed_log_probabilities - stored_log_probabilities
            ratio = log_ratio.exp()

            if epoch_index == 0 and minibatch_position == 0:
                epoch_zero_max_deviation = float(log_ratio.detach().abs().max())

            with torch.no_grad():
                # Schulman's low-variance approximate KL, which is non-negative.
                approximate_kl = float(((ratio - 1.0) - log_ratio).mean())
                clip_fractions.append(
                    float(((ratio - 1.0).abs() > config.clip_epsilon).float().mean())
                )

            normalized_advantages = advantages
            if config.normalize_advantages and normalized_advantages.numel() > 1:
                normalized_advantages = (normalized_advantages - normalized_advantages.mean()) / (
                    normalized_advantages.std() + 1e-8
                )

            unclipped_objective = ratio * normalized_advantages
            clipped_objective = (
                torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
                * normalized_advantages
            )
            policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()

            predicted_values = critic(observations, history_ids, history_masks)
            value_loss = torch.nn.functional.mse_loss(predicted_values, returns)

            total_loss = (
                policy_loss
                + config.value_function_coefficient * value_loss
                - config.entropy_coefficient * entropy_bonus
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()  # type: ignore[no-untyped-call]
            pre_clip_gradient_norm = nn.utils.clip_grad_norm_(
                [*actor.parameters(), *critic.parameters()], config.max_gradient_norm
            )
            optimizer.step()
            optimizer_steps += 1

            policy_losses.append(float(policy_loss.detach()))
            value_losses.append(float(value_loss.detach()))
            entropies.append(float(entropy_bonus.detach()))
            kl_divergences.append(approximate_kl)
            gradient_norms.append(float(pre_clip_gradient_norm))

        if (
            config.target_kl_divergence is not None
            and kl_divergences
            and kl_divergences[-1] > config.target_kl_divergence
        ):
            early_stopped = True
            break

    return UpdateMetrics(
        policy_loss=float(np.mean(policy_losses)),
        value_loss=float(np.mean(value_losses)),
        entropy=float(np.mean(entropies)),
        approximate_kl_divergence=float(np.mean(kl_divergences)),
        clip_fraction=float(np.mean(clip_fractions)),
        explained_variance=explained_variance(batch.state_values, batch.returns),
        gradient_norm=float(np.mean(gradient_norms)),
        learning_rate=effective_learning_rate,
        epoch_zero_log_probability_max_deviation=epoch_zero_max_deviation,
        optimizer_steps=optimizer_steps,
        early_stopped=early_stopped,
    )


def linear_learning_rate(
    initial_learning_rate: float, completed_updates: int, total_updates: int
) -> float:
    """Return the linearly annealed learning rate for the next update."""
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    remaining_fraction = max(0.0, 1.0 - completed_updates / total_updates)
    return initial_learning_rate * remaining_fraction
