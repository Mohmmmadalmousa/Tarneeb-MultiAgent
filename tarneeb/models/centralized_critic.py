"""The centralized (MAPPO-style) recurrent critic, Phase 10 / ablation A2.

Exactly one thing changes: the critic's static input width, from the 692-dim
actor-safe observation to the 856-dim environment observation, which adds the
three hidden hands and two derived per-seat counts.

The actor is untouched -- same architecture, same 956,617 parameters, same
692-dim input, same event stream, same recurrent state. That is what makes A2 an
ablation rather than two unrelated experiments: if the actor differed too, a
measured difference could not be attributed to critic information.

Why this needs almost no new model code. The frozen Phase 8
RecurrentArchitectureSpec already carries ``critic_observation_dim`` as a field,
and its docstring says why: a later centralized-critic phase can widen it without
touching that module. Phase 10 is the phase that claim was written for. So the
centralized critic is a *configuration* of the frozen network, not a
reimplementation, and this module is a thin factory plus the assertions that keep
it honest.

Building it any other way would be worse: a hand-written 856 critic would be a
second implementation of the same architecture, free to drift from the one the
control condition uses, and any A2 difference could then be an artifact of that
drift rather than of critic information.

The privileged information is training-only. The critic exists to reduce the
variance of the advantage estimate, and it is discarded at inference:
``load_recurrent_checkpoint(..., load_critic=False)`` is the deployment path, and
the arena agents never construct a critic at all. So a policy trained with a
centralized critic is, at play time, exactly as blind as one trained without it.

That is the CTDE claim -- centralized training, decentralized execution -- and
here it is structural rather than procedural: the actor network owns no parameter
shaped to accept 856 inputs, so a wiring mistake raises a shape error instead of
quietly leaking cards into the policy.

Parameter count. The 692 critic costs 940,169. Widening the first static layer
from 692 to 856 adds (856 - 692) * 512 = 83,968 weights, giving 1,024,137 -- the
figure blueprint section 6.3 states for the recurrent MAPPO critic. That
arithmetic is asserted by a test rather than trusted.
"""

from __future__ import annotations

from typing import Any

from tarneeb.env.spaces import D_CRITIC, D_STATIC
from tarneeb.models.nets import count_parameters
from tarneeb.models.recurrent import (
    RecurrentActorNetwork,
    RecurrentArchitectureSpec,
    RecurrentCriticNetwork,
    build_recurrent_actor_and_critic,
)

CRITIC_MODE_DECENTRALIZED = "decentralized"
"""Control condition: the critic reads the 692-dim actor-safe observation."""

CRITIC_MODE_CENTRALIZED = "centralized"
"""Treatment condition: the critic reads the 856-dim environment observation."""

CRITIC_MODES = (CRITIC_MODE_DECENTRALIZED, CRITIC_MODE_CENTRALIZED)

DECENTRALIZED_CRITIC_PARAMETERS = 940_169
"""The Phase 8/9 critic, asserted so a silent architecture change is caught."""

CENTRALIZED_CRITIC_PARAMETERS = 1_024_137
"""The 856-input critic. Equals the decentralized count plus (856-692)*512."""

ACTOR_PARAMETERS = 956_617
"""Unchanged by Phase 10. The actor is the constant in ablation A2."""

CENTRALIZED_EXTRA_PARAMETERS = (D_CRITIC - D_STATIC) * 512
"""164 * 512 = 83,968 -- the width of the critic-only feature block."""


def critic_observation_dim_for(critic_mode: str) -> int:
    """Return the critic's static input width for one A2 condition."""
    if critic_mode == CRITIC_MODE_DECENTRALIZED:
        return D_STATIC
    if critic_mode == CRITIC_MODE_CENTRALIZED:
        return D_CRITIC
    raise ValueError(f"critic_mode must be one of {CRITIC_MODES}, got {critic_mode!r}")


def architecture_spec_for(critic_mode: str) -> RecurrentArchitectureSpec:
    """Return the frozen recurrent spec configured for one A2 condition.

    The actor width is always D_STATIC. Only the critic width moves.
    """
    return RecurrentArchitectureSpec(
        actor_observation_dim=D_STATIC,
        critic_observation_dim=critic_observation_dim_for(critic_mode),
    )


def build_actor_and_critic_for(
    critic_mode: str, seed: int | None = None
) -> tuple[RecurrentActorNetwork, RecurrentCriticNetwork]:
    """Build the A2 networks for one condition.

    The actor is built from the same seed in both conditions, so control and
    treatment start from an identical policy. Only the critic differs, which is
    the whole point of the ablation.
    """
    return build_recurrent_actor_and_critic(architecture_spec_for(critic_mode), seed=seed)


def expected_critic_parameters(critic_mode: str) -> int:
    """Return the parameter count this condition's critic must have."""
    if critic_mode == CRITIC_MODE_DECENTRALIZED:
        return DECENTRALIZED_CRITIC_PARAMETERS
    if critic_mode == CRITIC_MODE_CENTRALIZED:
        return CENTRALIZED_CRITIC_PARAMETERS
    raise ValueError(f"critic_mode must be one of {CRITIC_MODES}, got {critic_mode!r}")


def verify_parameter_counts(
    actor: RecurrentActorNetwork, critic: RecurrentCriticNetwork, critic_mode: str
) -> dict[str, int]:
    """Return the measured counts, raising if they disagree with the contract.

    Called at the start of every Phase 10 run, so a configuration mistake stops
    the run instead of producing an experiment that answers a different question.
    """
    actor_count = count_parameters(actor)
    critic_count = count_parameters(critic)
    expected = expected_critic_parameters(critic_mode)

    if actor_count != ACTOR_PARAMETERS:
        raise ValueError(
            f"the actor must be unchanged in Phase 10: expected {ACTOR_PARAMETERS} "
            f"parameters, built {actor_count}"
        )
    if critic_count != expected:
        raise ValueError(
            f"the {critic_mode} critic must have {expected} parameters, built {critic_count}"
        )
    if actor.spec.actor_observation_dim != D_STATIC:
        raise ValueError(
            f"the actor must read {D_STATIC} features, not {actor.spec.actor_observation_dim}"
        )
    if critic.spec.critic_observation_dim != critic_observation_dim_for(critic_mode):
        raise ValueError("the critic width does not match the requested mode")

    return {
        "actor": actor_count,
        "critic": critic_count,
        "combined": actor_count + critic_count,
        "critic_observation_dim": critic.spec.critic_observation_dim,
        "actor_observation_dim": actor.spec.actor_observation_dim,
    }


def describe_condition(critic_mode: str) -> dict[str, Any]:
    """Return a JSON-safe description of one A2 condition, for the run report."""
    return {
        "critic_mode": critic_mode,
        "actor_observation_dim": D_STATIC,
        "critic_observation_dim": critic_observation_dim_for(critic_mode),
        "actor_parameters": ACTOR_PARAMETERS,
        "critic_parameters": expected_critic_parameters(critic_mode),
        "privileged": critic_mode == CRITIC_MODE_CENTRALIZED,
        "note": (
            "Privileged critic input is training-only; inference loads the actor "
            "alone, so execution is decentralized in both conditions."
        ),
    }
