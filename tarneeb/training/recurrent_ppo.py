"""Recurrent replay and the clipped PPO update over ``chunk_v2`` chunks.

The recurrent counterpart of :mod:`tarneeb.training.ppo`. The objective is the
same one, unchanged:

    ratio_t       = exp(log pi_new(a_t | o_t) - log pi_old(a_t | o_t))
    policy_loss   = -masked_mean( min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) )
    value_loss    = masked_mean( (V(o_t) - R_t)^2 )
    entropy_bonus = masked_mean( H[pi_new] )
    total_loss    = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus

Two things differ from the feedforward path, and both are about sequences rather
than about the mathematics.

**Minibatches sample whole chunks.** A chunk is a temporal unit; splitting one
across minibatches would either lose the hidden state carried between its
positions or reuse a stale one.

**Every mean is masked by ``valid``.** A short tail chunk is padded, and a padded
position must contribute exactly zero to the policy loss, the value loss, the
entropy term and every gradient. That is a property, not an aspiration, so a
test overwrites every padded field with noise and asserts nothing moves.

**Epoch-zero identity.** Replay follows rule R5: restore ``h0`` at the chunk
head, then for each position consume that position's stored segment *first* and
evaluate the decision *second*. Position 0's segment is empty by construction, so
the first forward is a strict no-op on the state -- exactly matching rollout,
where the acting policy consumed nothing. Under those conditions the recomputed
log-probabilities must reproduce the stored ones before any optimizer step. Any
deviation means the replayed inputs do not match the collected ones, which is a
silent corruption of every update, so it is measured on every update and gated
by :data:`RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from tarneeb.buffers.recurrent import ChunkBatch, masked_mean
from tarneeb.models.masking import MaskedCategorical
from tarneeb.models.recurrent import RecurrentActorNetwork, RecurrentCriticNetwork
from tarneeb.training.ppo import PPOConfig, explained_variance, set_learning_rate

RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE = 1e-4
"""Maximum tolerated epoch-zero log-probability deviation before a run is suspect.

Not exact equality, because packing may reorder a batch internally and float32
addition is not associative. A deviation above 1e-2 is a logic error, not
floating point.
"""

RECURRENT_EPOCH_ZERO_VALUE_TOLERANCE = 1e-3
"""Maximum tolerated epoch-zero value deviation."""


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """What replaying a chunk minibatch produced, all shaped ``(C, N)``."""

    log_probabilities: Tensor
    entropies: Tensor
    values: Tensor

    def max_log_probability_deviation(self, batch: ChunkBatch) -> float:
        """Return the largest ``|new - old|`` over valid positions only."""
        if batch.valid_transition_count == 0:  # pragma: no cover - empty rollouts raise earlier
            return 0.0
        deviation = (self.log_probabilities.detach() - batch.old_log_probabilities).abs()
        return float(deviation[batch.valid].max())

    def max_value_deviation(self, batch: ChunkBatch) -> float:
        """Return the largest ``|V_new - V_stored|`` over valid positions only."""
        if batch.valid_transition_count == 0:  # pragma: no cover
            return 0.0
        deviation = (self.values.detach() - batch.values).abs()
        return float(deviation[batch.valid].max())


def replay_chunks(
    actor: RecurrentActorNetwork,
    critic: RecurrentCriticNetwork,
    batch: ChunkBatch,
) -> ReplayResult:
    """Reconstruct a chunk minibatch's decisions under rule R5.

    Starts from the stored ``h0``/``hc0``, which are **detached**: gradients flow
    only within the chunk, and events before its head influence the loss through
    the value of ``h0`` but not through gradients into the parameters that
    produced it. That is the standard truncated-BPTT trade-off, chosen
    deliberately.
    """
    actor_hidden = batch.initial_actor_hidden.detach()
    critic_hidden = batch.initial_critic_hidden.detach()

    log_probabilities: list[Tensor] = []
    entropies: list[Tensor] = []
    values: list[Tensor] = []

    for position in range(batch.chunk_length):
        segment_ids = batch.previous_event_ids[position]
        segment_lengths = batch.previous_event_lengths[position]

        # R5: consume the between-decision segment FIRST (a strict no-op at
        # position 0, where the stored segment is empty), then evaluate.
        logits, actor_hidden = actor(
            batch.static[position], segment_ids, segment_lengths, actor_hidden
        )
        distribution = MaskedCategorical.from_logits(logits, batch.legal_action_masks[position])
        log_probabilities.append(distribution.log_probability_of(batch.actions[position]))
        entropies.append(distribution.entropy())

        position_values, critic_hidden = critic(
            batch.critic_static[position], segment_ids, segment_lengths, critic_hidden
        )
        values.append(position_values)

    return ReplayResult(
        log_probabilities=torch.stack(log_probabilities),
        entropies=torch.stack(entropies),
        values=torch.stack(values),
    )


@dataclass(frozen=True, slots=True)
class RecurrentUpdateMetrics:
    """PPO diagnostics for one recurrent update. Every field is a detached scalar."""

    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl_divergence: float
    clip_fraction: float
    explained_variance: float
    gradient_norm: float
    learning_rate: float
    epoch_zero_log_probability_max_deviation: float
    epoch_zero_value_max_deviation: float
    optimizer_steps: int
    chunk_count: int
    valid_transition_count: int
    early_stopped: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def build_recurrent_optimizer(
    actor: RecurrentActorNetwork, critic: RecurrentCriticNetwork, config: PPOConfig
) -> torch.optim.Optimizer:
    """Return one Adam optimizer over both networks' parameters.

    A single optimizer is safe precisely because the parameter sets are disjoint:
    each loss term reaches only its own network.
    """
    return torch.optim.Adam(
        [*actor.parameters(), *critic.parameters()],
        lr=config.learning_rate,
        eps=config.adam_epsilon,
    )


def normalize_advantages(advantages: Tensor, valid: Tensor) -> Tensor:
    """Return advantages standardized over the valid positions only.

    Including padded positions in the mean and standard deviation would let the
    amount of padding shift every real advantage, which would make the update
    depend on how the rollout happened to align with chunk boundaries.
    """
    if int(valid.sum()) <= 1:
        return advantages
    selected = advantages[valid]
    return (advantages - selected.mean()) / (selected.std() + 1e-8)


def _chunk_minibatches(
    chunk_count: int, minibatch_count: int, shuffle_generator: np.random.Generator
) -> list[np.ndarray]:
    """Split a shuffled chunk permutation into roughly equal groups of whole chunks."""
    permutation = shuffle_generator.permutation(chunk_count)
    return [group for group in np.array_split(permutation, minibatch_count) if group.size > 0]


def recurrent_ppo_update(
    actor: RecurrentActorNetwork,
    critic: RecurrentCriticNetwork,
    optimizer: torch.optim.Optimizer,
    batch: ChunkBatch,
    config: PPOConfig,
    shuffle_generator: np.random.Generator,
    learning_rate: float | None = None,
    check_epoch_zero: bool = True,
) -> RecurrentUpdateMetrics:
    """Run the configured PPO epochs over whole chunks and return diagnostics."""
    if batch.chunk_count == 0:
        raise ValueError("cannot update on an empty rollout")
    if batch.valid_transition_count == 0:
        raise ValueError("cannot update on a rollout with no valid transition")

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
    epoch_zero_log_probability_deviation = 0.0
    epoch_zero_value_deviation = 0.0
    optimizer_steps = 0
    early_stopped = False

    for epoch_index in range(config.update_epochs):
        for minibatch_position, chunk_indices in enumerate(
            _chunk_minibatches(batch.chunk_count, config.minibatch_count, shuffle_generator)
        ):
            minibatch = batch.select(torch.from_numpy(chunk_indices))
            replayed = replay_chunks(actor, critic, minibatch)

            valid = minibatch.valid
            log_ratio = replayed.log_probabilities - minibatch.old_log_probabilities
            ratio = log_ratio.exp()

            if epoch_index == 0 and minibatch_position == 0:
                epoch_zero_log_probability_deviation = replayed.max_log_probability_deviation(
                    minibatch
                )
                epoch_zero_value_deviation = replayed.max_value_deviation(minibatch)
                if (
                    check_epoch_zero
                    and epoch_zero_log_probability_deviation
                    > RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
                ):
                    raise AssertionError(
                        "epoch-zero log-probability deviation "
                        f"{epoch_zero_log_probability_deviation:.3e} exceeds "
                        f"{RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE:.0e}; the replayed "
                        "recurrent inputs do not match the collected ones (see "
                        "docs/RECURRENT_CONTRACT.md section 4)"
                    )

            with torch.no_grad():
                # Schulman's low-variance approximate KL, which is non-negative.
                approximate_kl = float(masked_mean((ratio - 1.0) - log_ratio, valid))
                clip_fractions.append(
                    float(masked_mean(((ratio - 1.0).abs() > config.clip_epsilon).float(), valid))
                )

            advantages = minibatch.advantages
            if config.normalize_advantages:
                advantages = normalize_advantages(advantages, valid)

            unclipped_objective = ratio * advantages
            clipped_objective = (
                torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
                * advantages
            )
            policy_loss = -masked_mean(torch.min(unclipped_objective, clipped_objective), valid)
            value_loss = masked_mean((replayed.values - minibatch.returns) ** 2, valid)
            entropy_bonus = masked_mean(replayed.entropies, valid)

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

    return RecurrentUpdateMetrics(
        policy_loss=float(np.mean(policy_losses)),
        value_loss=float(np.mean(value_losses)),
        entropy=float(np.mean(entropies)),
        approximate_kl_divergence=float(np.mean(kl_divergences)),
        clip_fraction=float(np.mean(clip_fractions)),
        explained_variance=explained_variance(
            batch.values[batch.valid], batch.returns[batch.valid]
        ),
        gradient_norm=float(np.mean(gradient_norms)),
        learning_rate=effective_learning_rate,
        epoch_zero_log_probability_max_deviation=epoch_zero_log_probability_deviation,
        epoch_zero_value_max_deviation=epoch_zero_value_deviation,
        optimizer_steps=optimizer_steps,
        chunk_count=batch.chunk_count,
        valid_transition_count=batch.valid_transition_count,
        early_stopped=early_stopped,
    )


def recurrent_losses_are_finite(metrics: RecurrentUpdateMetrics) -> bool:
    """Return True when no reported loss or diagnostic is a NaN or an infinity."""
    return all(
        np.isfinite(value)
        for value in (
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
            metrics.approximate_kl_divergence,
            metrics.gradient_norm,
        )
    )


def gradients_are_finite(module: nn.Module) -> bool:
    """Return True when every existing gradient is finite."""
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def recurrent_parameters_with_gradient(module: nn.Module) -> list[str]:
    """Return the names of parameters that received a non-zero gradient."""
    return [
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is not None and bool((parameter.grad != 0).any())
    ]


def gru_gradient_norm(module: nn.Module) -> float:
    """Return the total gradient norm over the GRU parameters only.

    Used by the gradient test: a recurrent update that leaves the GRU untouched
    would train a feedforward policy wearing a GRU's name, and the loss curve
    would not say so.
    """
    total = 0.0
    for name, parameter in module.named_parameters():
        if ".gru." in name and parameter.grad is not None:
            total += float((parameter.grad**2).sum())
    return float(np.sqrt(total))


def collect_update_diagnostics(
    actor: RecurrentActorNetwork, critic: RecurrentCriticNetwork
) -> dict[str, Any]:
    """Return post-backward gradient diagnostics for reporting."""
    return {
        "actor_gradients_finite": gradients_are_finite(actor),
        "critic_gradients_finite": gradients_are_finite(critic),
        "actor_gru_gradient_norm": gru_gradient_norm(actor),
        "critic_gru_gradient_norm": gru_gradient_norm(critic),
    }
