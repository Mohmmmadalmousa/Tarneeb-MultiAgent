"""Phase 11B additive auction repair utilities.

The repair layer is additive. ``data`` provides the public-only unclamped teacher
and reproducible coverage cases, ``learning`` fits isolated BID/PASS output rows,
and ``measurement``/``evaluation`` diagnose behavior without changing frozen PPO.
The legacy AuctionTeacher export remains a diagnostic Strong baseline; it is
not the unclamped teacher used by the new learning path.
"""

from __future__ import annotations

from tarneeb.auction_repair.diagnostics import (
    AuctionProbeSummary,
    AuctionTeacher,
    collect_phase11b_probe,
    summarize_learner_auction_metrics,
)

__all__ = [
    "AuctionProbeSummary",
    "AuctionTeacher",
    "collect_phase11b_probe",
    "summarize_learner_auction_metrics",
]
