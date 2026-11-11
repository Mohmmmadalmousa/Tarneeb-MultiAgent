# Tarneeb Rules — `tarneeb-classic-v1`

**Status: FROZEN.** This document, not the code, is the source of truth for the
rules of the game. Tests are written from this document; the implementation is
checked against it, never the other way round.

- Rules version: `tarneeb-classic-v1`
- Implemented by: `tarneeb/core/rules.py` (`RulesConfig`), `tarneeb/core/scoring.py` (`score_deal`)
- Rules fingerprint of the frozen defaults: `sha1:c18b43e7e9ebfc6855ff5fb56a968308d222a2f5`
  (recompute with `RulesConfig().fingerprint()`; the value changes if any frozen field changes)

The rule identifiers (C01–C26) match blueprint v1.2 section 2.2 so the two
documents can be cross-read.

---

## 1. Players, partnerships, deck

| ID | Rule |
|---|---|
| C01 | Four players. Two fixed partnerships, partners sitting opposite: team 0 is seats `(0, 2)`, team 1 is seats `(1, 3)`. |
| C02 | One standard 52-card deck. Ranks within a suit, high to low: A K Q J 10 9 8 7 6 5 4 3 2. Ace is high. |
| C03 | Thirteen cards are dealt to each player. All 52 cards are dealt; there is no kitty and no stock. |
| C04 | Bidding and play both proceed **counter-clockwise** (seat `p` is followed by seat `(p + 1) mod 4`). |

## 2. Dealer and the auction

| ID | Rule |
|---|---|
| C05 | The first dealer is chosen at random. Thereafter the deal passes to the right, i.e. dealer `d` is followed by dealer `(d + 1) mod 4`. |
| C06 | The first bidder is the player to the dealer's right, seat `(d + 1) mod 4`. |
| C07 | A bid is an integer from **7 to 13** inclusive. It is a claim of how many of the 13 tricks the bidder's team will take. |
| C08 | Each new bid must be **strictly greater** than the current highest bid. Equalling the current high bid is not allowed. |
| C09 | Passing is **permanent**: a player who passes takes no further part in that auction and may not bid again, even if bidding continues. |
| C10 | If all four players pass without any bid being made, the hand is thrown in and **redealt by the same dealer**. The deal does not pass. |
| C11 | The auction ends when exactly one active bidder remains **and** at least one bid has been made. That player is the **declarer**. |

## 3. Trump selection and the opening lead

| ID | Rule |
|---|---|
| C12 | The declarer chooses the trump suit from the four suits: clubs, diamonds, hearts, spades. |
| C13 | **No-trump is not available.** It is not part of this rules version. The action space reserves an action id for it, but that action is permanently masked and can never be legal or selected. |
| C14 | The declarer leads to the first trick. |
| C15 | There is no constraint on the opening lead. The declarer may lead any card, including a trump. |

## 4. Trick play

| ID | Rule |
|---|---|
| C16 | A player **must follow the suit led** if holding at least one card of that suit. |
| C17 | A player void in the led suit may play any card, trump or otherwise. There is no obligation to trump and no obligation to overtrump. |
| C18 | The trick is won by the **highest trump** played to it; if no trump was played, by the **highest card of the suit led**. Cards of other non-trump suits cannot win. |
| C19 | The winner of a trick leads to the next trick. |

Thirteen tricks are played. Every trick is won by exactly one team, so the two
teams' trick counts always sum to 13.

## 5. Scoring one deal

Let `b` be the winning bid, `B` the bidding team, `D` the defending team,
`t_B` the tricks taken by `B`, and `t_D = 13 − t_B` the tricks taken by `D`.
Scoring produces a pair of score changes `(ΔS_B, ΔS_D)`.

| ID | Situation | Result |
|---|---|---|
| C20 | `b < 13` and `t_B ≥ b` (contract made) | `(+t_B, 0)` — the bidding team scores its actual trick count; defenders score nothing. |
| C21 | `b < 13` and `t_B < b` (contract failed) | `(−b, t_D)` — the bidding team loses the value of its bid; defenders score their tricks. |
| C22 | `b < 13` and `t_B = 13` (bidder kaboot) | `(+16, 0)` — taking all 13 tricks after bidding fewer scores 16, not 13. |
| C23 | `b = 13` and `t_B = 13` (bid 13 made) | `(+26, 0)` |
| C24 | `b = 13` and `t_B < 13` (bid 13 failed) | `(−16, 2 · t_D)` — the bidding team loses 16 and defenders score **twice** their tricks. |
| C26 | `b < 13` and `t_D = 13` (defender kaboot) | `(−b, 16)` — a defending team taking all 13 tricks scores 16. |

### 5.1 Rule precedence — bid 13 outranks defender kaboot

The bid-13 rules are evaluated **first**. Only if `b < 13` do the kaboot rules
C22 and C26 apply. This matters in exactly one case:

> **Bid of 13, failed, with the defenders taking all 13 tricks
> (`b = 13`, `t_B = 0`, `t_D = 13`) scores `(−16, +26)`, not `(−16, +16)`.**

C24 applies and C26 does not. The defenders' 26 is `2 × 13` under the bid-13
doubling rule, not the defender-kaboot award of 16. This case is the minimum of
the score-difference range below, and the ordering is therefore load-bearing:
reversing it would change the frozen bounds.

### 5.2 Evaluation order

Applied top to bottom, the first matching line wins:

```text
b == 13 and t_B == 13   ->  (+26, 0)          C23
b == 13                 ->  (-16, 2 * t_D)    C24   (takes precedence over C26)
t_B == 13               ->  (+16, 0)          C22
t_B >= b                ->  (+t_B, 0)         C20
t_D == 13               ->  (-b, 16)          C26
otherwise               ->  (-b, t_D)         C21
```

### 5.3 Score-difference bounds

With `Δ = ΔS_B − ΔS_D` measured from the bidding team's point of view over all
7 bids × 14 possible trick counts = **98** inputs:

- Maximum `Δ = +26`, uniquely at `(b = 13, t_B = 13)` — bid 13 made.
- Minimum `Δ = −42`, uniquely at `(b = 13, t_B = 0)` — bid 13 failed with the
  defenders taking all 13 tricks, i.e. `−16 − 26`.
- Therefore `Δ ∈ [−42, +26]` for the bidding team and `[−26, +42]` for the
  defenders, so `|Δ| ≤ 42` for every seat.

The defender-kaboot branch cannot reach the minimum: it requires `b ≤ 12`, giving
`Δ = −b − 16 ≥ −28`.

## 6. Match end

| ID | Rule |
|---|---|
| C25 | The match target is **31**. The first team to reach 31 or more points wins the match. |

Deals are played and scored one after another, accumulating team scores, until
the target is reached. Scores may go negative; only reaching 31 ends the match.

## 7. Options that are OFF in this version

Each of the following is a real variation played at some tables. All are
disabled in `tarneeb-classic-v1` and none is implemented. Enabling any of them
produces a different rules version.

| Option | Value here | What enabling it would mean |
|---|---|---|
| `allow_no_trump` | `False` | No-trump becomes a biddable contract (C13). |
| `dealer_forced_seven` | `False` | The dealer must bid at least 7 if all others pass. |
| `dealer_may_equal` | `False` | The dealer may match rather than exceed the high bid. |
| `single_bid_round` | `False` | The Lebanese one-bid form: one bid per player, no second round. |
| `bidder_must_lead_trump` | `False` | The declarer must lead a trump to trick one. |
| `weak_hand_throw_in` | `False` | A sufficiently weak hand may force a redeal. |
| `long_suit_rule` | `False` | A long-suit variation on trick play. |
| `egyptian_bidding` | `False` | The Egyptian (Masrawy) suit-bidding variant. Out of scope; must raise `NotImplementedError`. |
| `defender_kaboot_bonus` | `True` | Turning it off gives defender kaboot 13 instead of 16 — released separately as `tarneeb-classic-v1-alt13`. |
| `target_score` | `31` | Other tables play to 41, 51 or 61. |
| `bid13_failed_defender_multiplier` | `2` | Setting it to 1 removes the bid-13 doubling. |

## 8. Rule versioning

No rule change may be silent. **Changing any frozen rule above requires all of:**

1. A **new rules version string** — `tarneeb-classic-v1` is never edited in place.
2. A **new rules fingerprint**, recomputed from the changed configuration.
3. **Updated tests**: the frozen-defaults table in `tests/test_rules.py` and, for
   any scoring change, the hand-written 98-row truth table in
   `tests/test_scoring.py`, rewritten from this document rather than from code.
4. **Invalidation of incompatible artifacts**: every checkpoint, evaluation
   result and replay recorded under the old fingerprint is no longer comparable
   and must be regenerated or clearly labelled with the version that produced it.

Artifacts carry `rules_version` and `rules_fingerprint`. Loading an artifact
whose fingerprint differs from the running configuration must be refused, not
silently accepted.
