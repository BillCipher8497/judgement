# Judgement engine (Phase 0)

A pure-Python (numpy only) rules engine for **Judgement / Oh Hell**, built for
research on bidding and play.  Deterministic, seed-reproducible, cross-validated
against OpenSpiel's `oh_hell` and against an independent naive reference.

```
judgement/        core package (cards, rules, state, engine, oracle, sampling, trace, policies/)
tests/            pytest + hypothesis (82 tests)
bench/bench.py    performance benchmarks          bench/baseline.py  greedy-baseline reproduction
crossval/         openspiel_check.py, naive_ref.py, REPORT.md
RESULTS.md        test / benchmark / cross-validation / baseline results
```

## Install & run

```bash
pip install -r requirements.txt          # numpy, pytest, hypothesis, open_spiel (optional)
python -m pytest                          # all tests (OpenSpiel tests skip if pyspiel is absent)
python bench/bench.py                     # benchmarks (--profile for a cProfile of the hot loop)
python bench/baseline.py                  # greedy baseline table
python -m crossval.openspiel_check --deals 10000 --players 4 --naive
```

## Rules (v1, all in the frozen `Rules` dataclass)

| Rule | Default | `Rules` field |
|---|---|---|
| Players | 4 (3–7 supported) | `n_players` |
| Max cards M | floor(52/N) | `max_cards` (None = floor) |
| Round sizes | M, M−1, …, 1, …, M (2M−1 rounds; 25 for N=4) | derived: `ladder()` |
| Trump | S, H, D, C, NT cycling by round index (round 0 = S) | `trump_cycle` |
| Dealer | seat 0 in round 0, then rotates left (+1) | `first_dealer` |
| Bidding | starts left of dealer, dealer last, bids 0..n | — |
| Hook | last bidder may not make Σbids == n | `hook` |
| Lead | left of dealer leads trick 1; trick winner leads next | — |
| Play | must follow suit; void → any card (no obligation to trump) | — |
| Trick | highest trump, else highest of led suit | — |
| Score | exact: `hit_base + bid` (10 + bid); miss: `miss_penalty` | `hit_base`, `miss_penalty` |
| Miss modes | `"zero"` (default), `"neg_bid"` (−bid), `"neg_10"`, `"neg_diff"` (−10·\|b−t\|), or `callable(bid, tricks)` | `miss_penalty` |
| Game | sum of round scores; k tied leaders each get 1/k win | — |

"Left" means seat + 1 (mod N).

## Encoding

`card = suit*13 + rank`, suits `S H D C` = 0..3, ranks `2..A` = 0..12; `"AS"`,
`"TD"` via `parse_card` / `card_str`.  Hands and legal-move sets are 52-bit int
bitmasks; `SUIT_MASK[s]` selects a suit, `cards_of(mask)` lists cards.
No-trump is `NT = -1`.

## API

```python
import numpy as np
from judgement import Rules, RoundState, play_game, play_round, parse_hand, hand_str
from judgement.policies import RandomPolicy, GreedyPolicy
from judgement.oracle import mc_trick_distribution, optimal_bid, mean_tricks_bid
from judgement.trace import replay, dumps

rules = Rules(n_players=4, miss_penalty="zero")

# A full 25-round game; policies are per seat.  Same seed -> byte-identical trace.
g = play_game(rules, [GreedyPolicy(), RandomPolicy(), RandomPolicy(), RandomPolicy()], seed=7)
print(g.totals, g.wins)
assert replay(g.trace) == g.trace
open("game.json", "w").write(dumps(g.trace))

# One round from an explicit position
hands = [parse_hand(h) for h in ["AS KS QS", "2H 3H 4H", "5D 6D 7D", "8C 9C TC"]]
st = RoundState(rules, n_cards=3, trump=0, dealer=3, hands=hands)
st.legal_bids()                 # [0, 1, 2, 3]
info = st.info_state(0)         # what seat 0 can see

# Oracle: P(tricks = k) under given play policies, and the score-optimal bid
dist = mc_trick_distribution(info, [GreedyPolicy()] * 4, n_samples=1000, rng=np.random.default_rng(0))
bid, ev = optimal_bid(info, dist)       # argmax_b E[score | b] over hook-legal b
mbid, mean = mean_tricks_bid(dist)      # round(E[tricks]) for comparison
```

### RNG and common random numbers

`play_game(rules, policies, seed)` builds `SeedSequence(seed).spawn(1 + N)`: stream
0 deals, stream 1+p belongs to seat p's policy.  Deals depend only on the seed,
so swapping a policy changes neither the deals nor any other seat's random stream —
paired comparisons across experiments see identical cards.

### State, info, and solver hooks

* `RoundState` — full information.  `apply_bid`, `apply_card` (checked),
  `_apply_card` (unchecked, for search), `undo_card` (O(1)), `copy()`.
  This is the surface a later DDS-style solver will search over.
* `InfoState` — a read-only per-seat **view** (seat, dealer, n_cards, trump,
  own hand, bids in order, tricks won, current trick and the card winning it,
  full play history, trick winners, known voids, cards left per seat).  Its API
  never yields another player's cards.  It is a live view for speed; call
  `info.snapshot()` for a detached immutable copy.
* `sampling.HandSampler(info)` — exact uniform sampling of hidden hands
  consistent with the observation (played cards, card counts, voids; the undealt
  stock when N·n < 52).

## Adding a policy

Any object with `bid` and `play` works (see `judgement/policies/base.py`):

```python
from judgement.cards import SUIT_MASK, nth_card

class LowballPolicy:
    name = "lowball"

    def bid(self, info, legal, rng):          # legal: list[int], hook already applied
        return legal[0]

    def play(self, info, legal_mask, rng):    # legal_mask: card bitmask; return a card int
        return (legal_mask & -legal_mask).bit_length() - 1   # lowest card index

play_game(rules, [LowballPolicy(), GreedyPolicy(), RandomPolicy(), RandomPolicy()], seed=1)
```

Rules for policies: use only the passed `rng` for randomness (reproducibility);
do not keep the `info` object between calls (it is a live view); return a legal
action (the engine raises `IllegalAction` otherwise).

## Baselines

* `RandomPolicy` — uniform over legal bids and cards.
* `GreedyPolicy` — naive "try to win every trick": lead the strongest card
  (highest trump, else highest rank); when following, play the highest card that
  beats the current winner (incl. ruffing/over-ruffing when void), otherwise
  discard the lowest card, preferring non-trumps.  Its own bid is a crude
  honour count; the research baseline bids with the oracle instead.

## Out of scope (Phase 0)

Target-aware play, endgame solver, CFR.
