"""Legal-action masking for the policy distribution.

The hard requirement: an illegal action must carry **exactly zero** probability,
not merely a small one. Legality always comes from the frozen engine's 65-entry
mask; the network never decides what is legal and is never taught legality
through a penalty.

How the zero is achieved. Illegal logits are replaced by
:data:`MASKED_LOGIT_FILL_VALUE`, a value so far below any reachable logit that
``exp(masked_logit - max_logit)`` underflows to exactly ``0.0`` in float32. The
fill is finite rather than ``-inf`` on purpose: with ``-inf`` the entropy term
would evaluate ``0 * -inf`` and produce NaN, which is a much worse failure than
a slightly inelegant constant. With a finite fill the log-probability of an
illegal action is a large finite negative number, its probability is exactly
zero, and ``probability * log_probability`` is exactly zero.

Action 64 (reserved no-trump) is never legal under ``tarneeb-classic-v1``, so it
is masked out by the engine and therefore unreachable here. That is asserted
rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tarneeb.core.actions import NO_TRUMP_ACTION, NUM_ACTIONS

MASKED_LOGIT_FILL_VALUE = -1.0e9
"""Logit assigned to an illegal action. ``exp(-1e9)`` underflows to exactly 0.0."""


def apply_legal_action_mask(logits: Tensor, legal_action_masks: Tensor) -> Tensor:
    """Return ``logits`` with every illegal entry driven to an unreachable value.

    Raises:
        ValueError: If a row has no legal action at all. A seat that cannot act
            must never be asked to, so an all-False row is a caller bug, not a
            situation to paper over with a uniform distribution.
    """
    if logits.shape != legal_action_masks.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} and masks {tuple(legal_action_masks.shape)} "
            "must have the same shape"
        )
    if logits.shape[-1] != NUM_ACTIONS:
        raise ValueError(f"expected {NUM_ACTIONS} actions, got {logits.shape[-1]}")
    legal = legal_action_masks.bool()
    if not bool(legal.any(dim=-1).all()):
        raise ValueError("a legal-action mask has no legal action; the seat must not be asked")
    if bool(legal[..., NO_TRUMP_ACTION].any()):
        raise ValueError(f"action {NO_TRUMP_ACTION} is reserved and must never be legal")
    return logits.masked_fill(~legal, MASKED_LOGIT_FILL_VALUE)


@dataclass(frozen=True, slots=True)
class MaskedCategorical:
    """A categorical distribution restricted to the legal actions.

    Built from raw logits plus the engine's mask. Sampling, log-probabilities and
    entropy are all computed from the masked logits, so no code path can put
    probability on an illegal action.
    """

    masked_logits: Tensor
    legal_action_masks: Tensor

    @classmethod
    def from_logits(cls, logits: Tensor, legal_action_masks: Tensor) -> MaskedCategorical:
        return cls(
            masked_logits=apply_legal_action_mask(logits, legal_action_masks),
            legal_action_masks=legal_action_masks.bool(),
        )

    @property
    def log_probabilities(self) -> Tensor:
        """``(B, 65)`` log-probabilities; illegal entries are large and finite."""
        return torch.log_softmax(self.masked_logits, dim=-1)

    @property
    def probabilities(self) -> Tensor:
        """``(B, 65)`` probabilities; illegal entries are exactly ``0.0``."""
        return torch.softmax(self.masked_logits, dim=-1)

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        """Draw one action per row from an explicit generator.

        ``torch.multinomial`` never returns an index whose probability is exactly
        zero, so a sampled action is legal by construction. The caller still
        verifies it: masking is a safety property and is checked, not trusted.
        """
        return torch.multinomial(self.probabilities, num_samples=1, generator=generator).squeeze(-1)

    def most_probable_action(self) -> Tensor:
        """Return the greedy action per row, used by deterministic evaluation."""
        return torch.argmax(self.masked_logits, dim=-1)

    def log_probability_of(self, selected_actions: Tensor) -> Tensor:
        """Return ``(B,)`` log-probabilities of the given actions."""
        return self.log_probabilities.gather(-1, selected_actions.long().unsqueeze(-1)).squeeze(-1)

    def entropy(self) -> Tensor:
        """Return ``(B,)`` entropy over the legal actions only.

        Illegal entries contribute ``0 * large_negative``, which is exactly zero,
        so the sum is the entropy of the restricted distribution.
        """
        probabilities = self.probabilities
        return -(probabilities * self.log_probabilities).sum(dim=-1)


def actions_are_legal(selected_actions: Tensor, legal_action_masks: Tensor) -> bool:
    """Return True when every selected action is allowed by its own mask."""
    chosen = legal_action_masks.bool().gather(-1, selected_actions.long().unsqueeze(-1))
    return bool(chosen.all())
