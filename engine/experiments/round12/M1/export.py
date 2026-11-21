"""Portable first-decision supervision with full-population integer certificates."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[key] = '1'
import gc
import hashlib
import json
import pickle
import sys
import threading
import time
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; ENGINE = HERE.parents[2]; ROOT = ENGINE.parent
D3 = ENGINE/'experiments/round9/D3'; OUT = HERE/'results'; sys.path[:0] = [str(ENGINE), str(D3)]
from judgement.cfr.toy import ToyGame
from compact_tree import CompactTree
from bounded_ta import clear_mid

STOP = datetime(2026, 11, 17, 6, 0, tzinfo=timezone.utc).timestamp() #01:00ET
EXPECTED = {(2, 2, 0): ('1257/140', '1257/140'), (2, 2, -1): ('946/105', '946/105'),
            (3, 3, 0): ('367921/61600', '446669/73920'), (3, 3, -1): ('106419/15400', '213259/30800')}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()


def save(path, row):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(row, indent=2)+'\n', encoding='utf-8'); replace(tmp, path)


def replace(tmp, path):
    # OneDrive may briefly hold the previous checkpoint open on Windows.
    for attempt in range(40):
        try:
            os.replace(tmp, path); return
        except PermissionError:
            if attempt == 39: raise
            time.sleep(.1)


def log(s):
    with (HERE/'STATUS.txt').open('a', encoding='utf-8') as f: f.write(datetime.now(timezone.utc).isoformat()+' '+s+'\n')
    print(s, flush=True)


def split_hands(g):
    hands = list(combinations(g.deck, g.n))
    def key(hand): return hashlib.sha256(json.dumps([2000000000, g.S, g.R, g.n, list(hand)], separators=(',', ':')).encode('ascii')).digest()
    hands.sort(key=key); H = len(hands); cuts = [int(.6*H), int(.8*H), int(.9*H)]
    table = {h: int(i >= cuts[0])+int(i >= cuts[1])+int(i >= cuts[2]) for i, h in enumerate(hands)}
    return table, {'seed_base': 2000000000, 'ordered_hands': [list(h) for h in hands], 'cutpoints': cuts,
                   'names': ['train', 'validation', 'discovery', 'confirmation']}


def export(g):
    tag = f'S{g.S}R{g.R}n{g.n}_t{g.trump}'; directory = OUT/tag; directory.mkdir(exist_ok=True)
    sources = [Path(__file__), D3/'compact_tree.py', D3/'bounded_ta.py', ENGINE/'judgement/cfr/toy.py', ENGINE/'judgement/cfr/heuristics.py', ROUND/'M3/PROTOCOL.md']
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sources}
    if (directory/'manifest.json').exists():
        old = json.loads((directory/'manifest.json').read_text()); assert old['source_sha256'] == hashes
        assert sha(directory/'root_data.npz') == old['data_sha256']; return
    split_table, split_manifest = split_hands(g); save(directory/'split_frozen.json', split_manifest)
    if g.n == 3:
        meta = json.loads((D3/'results'/f'tree_cache_{g.trump}.json').read_text())
        fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in sources[1:5])).hexdigest()
        assert fingerprint == meta['source_sha256']; cache = Path(meta['derived_cache'])
        with cache.open('rb') as f: t = pickle.load(f)
        cache_hash = sha(cache)
    else:
        def checkpoint(d, nodes, infos):
            save(directory/'build_progress.json', {'deals': d, 'nodes': nodes, 'infosets': infos})
            if time.time() >= STOP: raise TimeoutError('M1 build hard stop')
        t = CompactTree(g, checkpoint=checkpoint); cache_hash = None
    assert t.g == g and len(t.roots) == (2520 if g.n == 2 else 369600)
    v = t.U[:, 0].astype(np.int16); np.testing.assert_array_equal(v, t.U[:, 0])
    for j, (a, b, seg, parents) in enumerate(reversed(t.chunks)):
        if time.time() >= STOP: raise TimeoutError('M1 DP hard stop')
        v[parents] = np.minimum.reduceat(v[t.e_ch[a:b]], seg)
        save(directory/'dp_progress.json', {'layers_complete': j+1, 'n_layers': len(t.chunks)})
    choice = t.node_pl[t.roots] >= 0; nodes = t.roots[choice]; deal_indices = np.flatnonzero(choice)
    assert len(set(nodes.tolist())) == len(nodes)
    infos, group = np.unique(t.node_is[nodes], return_inverse=True); G = len(infos); N = len(nodes)
    marker = np.full(t.n_nodes, -1, np.int32); marker[nodes] = np.arange(N, dtype=np.int32)
    selected = marker[t.e_par] >= 0
    q = np.full((N, 3), 127, np.int8)
    q[marker[t.e_par[selected]], t.e_a[selected]] = v[t.e_ch[selected]]
    mask = t.is_valid[infos, :3]; assert np.all(t.is_valid[infos, 3:] == 0)
    np.testing.assert_array_equal(q != 127, mask[group]); assert np.all(mask.sum(1) >= 2)
    keys = []; splits = []; payoff = []
    for i in infos:
        p, hand, hist = t.is_key[int(i)]; assert p in [1, 2, 3]
        keys.append({'seat': int(p), 'hand': list(hand), 'history': list(hist), 'n': g.n, 'S': g.S, 'R': g.R, 'trump': g.trump,
                     'legal': np.flatnonzero(t.is_valid[i, :3]).tolist()})
        splits.append(split_table[tuple(hand)]); payoff.append(10+int(hist[0]))
    payoff = np.array(payoff, np.int16); splits = np.array(splits, np.uint8)
    assert np.all((q == 127) | (q == 0) | (q == payoff[group, None])), 'Binary public-payoff hypothesis refuted'
    count = np.bincount(group, minlength=G).astype(np.int64)
    q_sum = np.zeros((G, 3), np.int64); hard = np.zeros_like(q_sum); optimal = np.zeros_like(q_sum); soft6 = np.zeros_like(q_sum)
    best = q.min(1); good = (q == best[:, None]) & mask[group]
    chosen = q.argmin(1); ties = good.sum(1)
    for a in range(3):
        np.add.at(q_sum[:, a], group, np.where(q[:, a] == 127, 0, q[:, a]))
        np.add.at(hard[:, a], group, chosen == a)
        np.add.at(optimal[:, a], group, good[:, a])
        np.add.at(soft6[:, a], group, good[:, a]*(6//ties))
    pi_by_group = np.zeros(G, np.int64); np.add.at(pi_by_group, group, best)
    terminal_sum = int(v[t.roots[~choice]].sum()); pi_sum = int(v[t.roots].sum())
    j1_sum = terminal_sum+int(np.where(mask, q_sum, 2**60).min(1).sum())
    pi, j1 = str(Fraction(pi_sum, t.n_deals)), str(Fraction(j1_sum, t.n_deals))
    assert (pi, j1) == EXPECTED[(g.S, g.n, g.trump)], ('External rational mismatch', tag, pi, j1)
    assert np.all(hard.sum(1) == count) and np.all(soft6.sum(1) == 6*count)
    data_path = directory/'root_data.npz'; tmp = data_path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, q=q, root_group=group.astype(np.int32), root_deal_index=deal_indices.astype(np.int32),
                            mask=mask, count=count, q_sum=q_sum, hard_counts=hard, optimal_counts=optimal,
                            uniform_optimal_units6=soft6, pi_by_group=pi_by_group, payoff=payoff, split=splits)
    replace(tmp, data_path)
    key_path = directory/'private_contexts.jsonl'
    with key_path.open('w', encoding='utf-8') as f:
        for i, key in enumerate(keys): f.write(json.dumps({'group': i, 'observation': key}, separators=(',', ':'))+'\n')
    manifest = {'condition': tag, 'N': 4, 'S': g.S, 'R': g.R, 'n': g.n, 'trump': g.trump,
                'n_exact_deals': t.n_deals, 'n_root_decision_worlds': N, 'n_private_groups': G,
                'terminal_root_score_sum': terminal_sum, 'PI_sum': pi_sum, 'J1_sum': j1_sum,
                'PI_fraction': pi, 'J1_fraction': j1, 'source_sha256': hashes,
                'cache_sha256': cache_hash, 'data_sha256': sha(data_path), 'contexts_sha256': sha(key_path),
                'split_group_counts': np.bincount(splits, minlength=4).tolist(),
                'split_world_counts': [int(count[splits == x].sum()) for x in range(4)],
                'scope': 'Exact first-free-team-decision relaxation, PI continuation afterward; not feasible full hidden teamwork; no sampling CI',
                'created_utc': datetime.now(timezone.utc).isoformat()}
    save(directory/'manifest.json', manifest); log(json.dumps(manifest))
    del t, v, marker, q, selected, keys; gc.collect(); clear_mid()


def main():
    assert time.time() < STOP; OUT.mkdir(exist_ok=True)
    timer = threading.Timer(STOP-time.time(), lambda: os._exit(124)); timer.daemon = True; timer.start()
    try:
        for S, n in [(2, 2), (3, 3)]:
            for trump in [0, -1]: export(ToyGame(N=4, S=S, R=4, n=n, trump=trump))
    finally: timer.cancel()


if __name__ == '__main__': main()
