"""Single-core benchmarks.  Run:  python bench/bench.py [--quick] [--profile]

Targets (SPEC): >= 5,000 full 13-card 4p rounds/sec with random policies;
oracle 1,000-sample trick distribution for a 13-card hand in < 2 s.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from judgement.engine import deal, make_rngs, play_game, play_round, run_round  # noqa: E402
from judgement.oracle import mc_trick_distribution  # noqa: E402
from judgement.policies import GreedyPolicy, RandomPolicy  # noqa: E402
from judgement.rules import Rules  # noqa: E402
from judgement.state import RoundState  # noqa: E402


def best_of(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def bench_rounds(policy, n_rounds, api=False):
    rules = Rules()
    pols = [policy] * 4

    if api:
        def go():
            for i in range(n_rounds):
                play_round(rules, pols, seed=i, n_cards=13, trump=0, dealer=i % 4)
    else:
        def go():
            d, prs = make_rngs(0, 4)
            for i in range(n_rounds):
                st = RoundState(rules, 13, i % 4, i % 4, deal(rules, 13, d))
                run_round(st, pols, prs)
    return n_rounds / best_of(go)


def bench_games(policy, n_games, trace):
    rules = Rules()
    pols = [policy] * 4

    def go():
        for i in range(n_games):
            play_game(rules, pols, seed=i, record_trace=trace)
    return n_games / best_of(go)


def bench_oracle(policy, n_hands=5, n_samples=1000):
    rules = Rules()
    times = []
    for h in range(n_hands):
        rng = np.random.default_rng(100 + h)
        st = RoundState(rules, 13, h % 4, 0, deal(rules, 13, rng))
        info = st.info_state(1)
        t = time.perf_counter()
        mc_trick_distribution(info, [policy] * 4, n_samples, np.random.default_rng(h))
        times.append(time.perf_counter() - t)
    return float(np.mean(times)), float(np.max(times))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args(argv)
    R = 2000 if a.quick else 10000
    G = 20 if a.quick else 100

    if a.profile:
        import cProfile
        import pstats
        cProfile.runctx("bench_rounds(RandomPolicy(), 3000)", globals(), locals(), "/tmp/judgement.prof")
        pstats.Stats("/tmp/judgement.prof").sort_stats("tottime").print_stats(12)
        return

    print(f"python {platform.python_version()} | numpy {np.__version__} | {platform.machine()} | single core")
    rr = bench_rounds(RandomPolicy(), R)
    rg = bench_rounds(GreedyPolicy(), R)
    rra = bench_rounds(RandomPolicy(), R // 2, api=True)
    print(f"13-card 4p rounds/sec, random  (run_round loop, incl. deal+bids): {rr:9,.0f}   target >= 5,000  "
          f"[{'PASS' if rr >= 5000 else 'FAIL'}]")
    print(f"13-card 4p rounds/sec, greedy  (run_round loop, incl. deal+bids): {rg:9,.0f}")
    print(f"13-card 4p rounds/sec, random  (play_round API, per-round seeding): {rra:9,.0f}")
    for name, pol in (("random", RandomPolicy()), ("greedy", GreedyPolicy())):
        g0 = bench_games(pol, G, trace=False)
        g1 = bench_games(pol, G, trace=True)
        print(f"full 25-round 4p games/sec, {name:6s}: {g0:7,.1f} (no trace)  {g1:7,.1f} (with trace)")
    for name, pol in (("greedy", GreedyPolicy()), ("random", RandomPolicy())):
        mean, worst = bench_oracle(pol)
        print(f"oracle 1,000-sample trick dist, 13-card hand, {name} rollouts: mean {mean:.3f}s  max {worst:.3f}s"
              f"   target < 2s  [{'PASS' if worst < 2 else 'FAIL'}]")


if __name__ == "__main__":
    main()
