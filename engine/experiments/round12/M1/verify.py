"""Fresh manual forward recursion; never imports a tree/cache/value optimizer."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
import sys
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; ENGINE = HERE.parents[2]
sys.path[:0] = [str(ENGINE), str(ENGINE/'experiments/round9/D3'), str(ROUND)]
from judgement.cfr.toy import ToyGame
from judgement.cfr.heuristics import myopic_bid
from bounded_ta import ta_card, clear_mid
from common import save, sha, stamp, log


def deals(deck, n, hands=()):
    if len(hands) == 3:
        yield hands+(tuple(deck),); return
    for hand in combinations(deck, n):
        yield from deals(tuple(c for c in deck if c not in hand), n, hands+(hand,))


def legal(hand, trick):
    following = [c for c in hand if trick and c//4 == trick[0][1]//4]
    return following or list(hand)


def advance(g, state, p, c):
    left, trick, leader, won, hist = state
    left = [list(h) for h in left]; left[p].remove(c); trick = list(trick)+[(p, c)]; won = list(won)
    if len(trick) == 4:
        led = trick[0][1]//4
        def strength(pc):
            suit = pc[1]//4
            return (2 if g.trump != -1 and suit == g.trump else int(suit == led), pc[1]%4)
        leader = max(trick, key=strength)[0]; won[leader] += 1; trick = []
    return left, trick, leader, won, hist+bytes([c])


def collapse(g, hands, bids, state):
    while len(state[4]) < 4+4*g.n:
        left, trick, leader, won, hist = state
        # Same legitimate early impossibility condition, recomputed from cards.
        remaining = len(left[0])+int(any(p == 0 for p, _ in trick))
        if bids[0] < won[0] or bids[0] > won[0]+remaining: return 0, None
        p = (leader+len(trick)) % 4; choices = legal(left[p], trick)
        if p and len(choices) > 1: return None, (p, choices, state)
        c = choices[0] if len(choices) == 1 else ta_card(g, p, hands[p], hist)
        state = advance(g, state, p, c)
    return (10+bids[0] if state[3][0] == bids[0] else 0), None


def minimum(g, hands, bids, state):
    terminal, free = collapse(g, hands, bids, state)
    if free is None: return terminal
    p, choices, state = free
    return min(minimum(g, hands, bids, advance(g, state, p, c)) for c in choices)


def main():
    reports = {}
    for S, n in ((2, 2), (3, 3)):
        for trump in (0, -1):
            tag = f'S{S}R4n{n}_t{trump}'; folder = HERE/'results'/tag
            m = json.loads((folder/'manifest.json').read_text()); data = dict(np.load(folder/'root_data.npz'))
            assert sha(folder/'root_data.npz') == m['data_sha256']
            rows = [json.loads(s)['observation'] for s in (folder/'private_contexts.jsonl').read_text().splitlines()]
            selected = set(range(2520)) if n == 2 else set(np.linspace(0, 369599, 64, dtype=int).tolist())
            saved = {int(d): i for i, d in enumerate(data['root_deal_index'])}; checked = 0; roots = 0; total_pi = 0
            g = ToyGame(N=4, S=S, R=4, n=n, trump=trump)
            for d, hands in enumerate(deals(tuple(range(S*4)), n)):
                if d not in selected: continue
                bids = []
                for p in range(4): bids.append(myopic_bid(g, p, hands[p], tuple(bids)))
                initial = ([list(h) for h in hands], [], 0, [0]*4, bytes(bids))
                terminal, root = collapse(g, hands, bids, initial)
                if root is None:
                    assert d not in saved; total_pi += terminal
                else:
                    assert d in saved; i = saved[d]; group = int(data['root_group'][i]); obs = rows[group]
                    p, choices, state = root
                    assert obs['seat'] == p and obs['hand'] == list(hands[p]) and obs['history'] == list(state[4])
                    fresh = {hands[p].index(c): minimum(g, hands, bids, advance(g, state, p, c)) for c in choices}
                    assert set(fresh) == set(obs['legal'])
                    for a, q in fresh.items(): assert q == int(data['q'][i, a]), (tag, d, a, q)
                    total_pi += min(fresh.values()); roots += 1
                checked += 1
            assert checked == len(selected)
            if n == 2: assert total_pi == m['PI_sum']
            reports[tag] = {'checked_deals': checked, 'checked_free_roots': roots,
                'selection': 'all2520 small;64fixed evenly spaced lexicographic large indices',
                'exact_PI_if_full_population': str(Fraction(total_pi, checked)) if n == 2 else None,
                'data_sha256': m['data_sha256'], 'status': 'pass'}
            clear_mid(); save(HERE/'results/fresh_recursion_verification.json', {'conditions': reports, 'verifier_sha256': sha(__file__), 'utc': stamp()})
            log(HERE, f'Fresh recursion passes {tag}: {checked}deals/{roots}free-root Q vectors. No cache/tree transitions used.')


if __name__ == '__main__': main()
