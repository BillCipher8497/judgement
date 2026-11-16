# Phase 0 Spec — Judgement Engine

## Locked rules (v1)
- 52-card deck, ranks 2..A, suits S,H,D,C. N players (default 4, engine parametric 3–7).
- Max cards M = floor(52/N). Ladder: M, M-1, ..., 2, 1, 2, ..., M  → 2M-1 rounds (25 for N=4).
- Trump cycles S, H, D, C, NT continuously by round index (round 0 = S).
- Dealer rotates left each round (round 0 dealer = seat 0). Bidding starts left of dealer; dealer bids last.
- Bids are integers 0..n. Hook rule: the last bidder may not bid b such that sum(bids) == n.
- First lead: left of dealer. Trick winner leads next.
- Must follow suit if able; if void, any card (no obligation to trump). Highest trump wins, else highest card of led suit. NT rounds: highest of led suit.
- Score: hit exactly → hit_base + bid (hit_base=10). Miss → miss_penalty(bid, tricks), default 0.
  Parametric modes: "zero", "neg_bid" (−bid), "neg_10" (−10), "neg_diff" (−10·|b−t|), or a callable.
- Game: sum of round scores. Ties share the win (report fractional win = 1/k).
- All rules live in a frozen `Rules` dataclass; nothing hard-coded elsewhere.

## Architecture (Python 3.11+, no heavy deps in core; numpy allowed)
```
judgement/
  cards.py      # card = int 0..51 (suit*13 + rank), hands as 52-bit int bitmasks, suit masks, pretty-print/parse ("AS","TD")
  rules.py      # Rules dataclass, ladder(), trump_for_round(), score()
  state.py      # RoundState (full), InfoState (per-player observation), legal_bids(), legal_cards()
  engine.py     # play_round(), play_game(); deterministic given seed
  policies/     # Policy protocol + baselines
    base.py     # bid(info, legal, rng) -> int ; play(info, legal_mask, rng) -> card
    random_p.py
    greedy.py   # reproduces the naive baseline: always try to win every trick
  oracle.py     # mc_trick_distribution(info, play_policies, n_samples, rng) -> P(tricks=k);
                # optimal_bid = argmax_b Σ_t P(t)·score(b,t) under Rules (= (10+b)·P(b) only when miss=0) (respecting hook legality)
  sampling.py   # sample opponent hands consistent with InfoState (known cards, observed voids)
  trace.py      # JSON-serializable round/game traces; replay(trace) reproduces exactly
tests/
bench/bench.py
crossval/openspiel_check.py
README.md
```
- InfoState must expose: seat, dealer, n_cards, trump, own hand, bids so far (in order), tricks won per player, current trick, full play history, known voids per player. Never leak hidden hands.
- Policies are pluggable per seat: `play_game(rules, policies=[p0,p1,p2,p3], seed)`.
- RNG: one `numpy.random.Generator` seeded per game; deal RNG and policy RNGs derived via SeedSequence spawn so changing a policy doesn't change deals (common random numbers across experiments — required for paired comparisons later).

## Correctness requirements
1. Unit tests: follow-suit legality, trump beats led suit, NT resolution, off-suit discard never wins, hook rule (incl. n=1 and edge sums), ladder/trump schedule for N=3..7, scoring modes.
2. Invariants (property-based, hypothesis): every card played once; Σtricks == n; winner leads; legal_cards never empty; dealer always has ≥1 legal bid.
3. Determinism: same seed → byte-identical trace; replay(trace) == trace.
4. Cross-validation vs OpenSpiel `oh_hell` (pip install open_spiel): on ≥10,000 seeded deals, force identical deal/trump/bids/actions and assert identical legal-move sets and trick winners. Document any rule differences (trump selection, scoring, hook) rather than forcing agreement on them.
5. Reproduce naive baseline: greedy policy, 4p, report trick distributions + oracle optimal bids for 5 fixed sample hands (incl. one bimodal: long trump + void).

## Performance targets (single core)
- ≥ 5,000 full 13-card 4p rounds/sec with random policies; report greedy too.
- oracle: 1,000-sample trick distribution for a 13-card hand in < 2 s.
- Profile; if targets missed, optimize hot paths (bitmask ops, avoid object churn). Numba optional, only behind a flag — core must run pure Python.

## Out of scope for Phase 0
Target-aware play, endgame solver, CFR. But design so a perfect-info DDS-style solver can consume RoundState later (cheap copy/undo).

## Deliverables
Code + tests passing + `bench/bench.py` output + `crossval` report + README (rules, API example, how to add a policy) + RESULTS.md with baseline reproduction and benchmark numbers.
