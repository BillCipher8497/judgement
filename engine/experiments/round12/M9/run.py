"""Frozen root students as fully executable private team policies, exact values."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import gc
import gzip
import importlib.util
import json
import pickle
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; ENGINE = HERE.parents[2]; ROOT = ENGINE.parent
D3 = ENGINE/'experiments/round9/D3'; sys.path[:0] = [str(ROUND), str(ENGINE), str(D3)]
from common import save, sha, replace, stamp, log
from learning import decisions
from root_features import encode
from judgement.cfr.toy import ToyGame
from compact_tree import CompactTree
from bounded_ta import ta_card, clear_mid
from judgement.cfr.heuristics import myopic_bid
spec = importlib.util.spec_from_file_location('round12_manual_forward', ROUND/'M1/verify.py')
manual = importlib.util.module_from_spec(spec); spec.loader.exec_module(manual)
OUT = HERE/'results'; CACHE = Path(tempfile.gettempdir())/'judgement_round12_M9'
STOP = datetime(2026, 11, 17, 11, 45, tzinfo=timezone.utc).timestamp() #06:45ET
OLD_UPPERS = {(2, 0): Fraction(1257, 140), (2, -1): Fraction(946, 105),
              (3, 0): Fraction(574247, 92400), (3, -1): Fraction(109709, 15400)}


def checkpoint(path, **data):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **data)
    replace(tmp, path)


def observation(g, p, hand, hist, legal):
    return dict(seat=p, hand=list(hand), history=list(hist), n=g.n, S=g.S, R=g.R,
                trump=g.trump, legal=[hand.index(c) for c in sorted(legal)])


def load_tree(g):
    if g.n == 2: return CompactTree(g), None
    meta = json.loads((D3/'results'/f'tree_cache_{g.trump}.json').read_text())
    source = [D3/'compact_tree.py', D3/'bounded_ta.py', ENGINE/'judgement/cfr/toy.py', ENGINE/'judgement/cfr/heuristics.py']
    import hashlib
    assert hashlib.sha256(b''.join(p.read_bytes() for p in source)).hexdigest() == meta['source_sha256']
    with Path(meta['derived_cache']).open('rb') as f: tree = pickle.load(f)
    return tree, sha(meta['derived_cache'])


def feature_cache(tree, folder, tag):
    path = CACHE/(tag+'_features.npy'); receipt = CACHE/(tag+'_features.json')
    expected = {'encoder_sha256': sha(ROUND/'root_features.py'), 'n_private_keys': tree.n_is, 'width': 241}
    if receipt.exists():
        old = json.loads(receipt.read_text()); assert all(old[k] == v for k, v in expected.items()) and sha(path) == old['data_sha256']
    else:
        x = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=(tree.n_is, 241))
        for start in range(0, tree.n_is, 10000):
            if time.time() >= STOP: raise TimeoutError('M9 encoder stop')
            for i in range(start, min(start+10000, tree.n_is)):
                p, hand, hist = tree.is_key[i]; legal = np.flatnonzero(tree.is_valid[i, :3]).tolist()
                x[i] = encode(dict(seat=int(p), hand=list(hand), history=list(hist), n=tree.g.n, S=tree.g.S, R=4,
                    trump=tree.g.trump, legal=legal))
            x.flush(); save(folder/'feature_progress.json', {'complete_private_keys': min(start+10000, tree.n_is), 'target': tree.n_is})
        del x; save(receipt, dict(expected, data_sha256=sha(path)))
    return np.load(path, mmap_mode='r'), json.loads(receipt.read_text())


def private_scores(tree, action):
    assert np.all(tree.is_valid[np.arange(tree.n_is), action])
    value = tree.U[:, 0].astype(np.int16)
    for lo, hi, seg, parents in reversed(tree.chunks):
        chosen = tree.e_a[lo:hi] == action[tree.e_is[lo:hi]]
        selected_parents = tree.e_par[lo:hi][chosen]
        assert np.array_equal(selected_parents, parents)
        value[selected_parents] = value[tree.e_ch[lo:hi][chosen]]
    return value[tree.roots].astype(np.uint8)


def fresh_forward(tree, params, objective, actions, indices, expected_scores, folder, label):
    g = tree.g; lookup = {key: i for i, key in enumerate(tree.is_key.packed)}; extra = {}; count = 0; total = 0
    chosen_indices = None if indices is None else set(indices)
    for d, hands in enumerate(manual.deals(tuple(range(g.S*4)), g.n)):
        if chosen_indices is not None and d not in chosen_indices: continue
        if time.time() >= STOP: raise TimeoutError('M9 forward stop')
        bids = []
        for p in range(4): bids.append(myopic_bid(g, p, hands[p], tuple(bids)))
        state = ([list(h) for h in hands], [], 0, [0]*4, bytes(bids))
        while len(state[4]) < 4+4*g.n:
            left, trick, leader, won, hist = state; p = (leader+len(trick)) % 4; legal = manual.legal(left[p], trick)
            if len(legal) == 1: card = legal[0]
            elif p == 0: card = ta_card(g, p, hands[p], hist)
            else:
                key = bytes((p, *hands[p]))+hist
                if key not in extra:
                    obs = observation(g, p, hands[p], hist, legal); mask = np.zeros((1, 3), bool); mask[0, obs['legal']] = True
                    extra[key] = int(decisions(encode(obs)[None], mask, params, objective)[0])
                    if key in lookup: assert extra[key] == int(actions[lookup[key]])
                a = extra[key]
                card = hands[p][a]; assert card in legal
            state = manual.advance(g, state, p, card)
        score = 10+bids[0] if state[3][0] == bids[0] else 0
        assert score == int(expected_scores[d]), (label, d, score, expected_scores[d])
        total += score; count += 1
        if count % 10000 == 0: save(folder/f'{label}_forward_progress.json', {'complete_deals': count, 'hero_score_sum': total})
    assert count == (tree.n_deals if chosen_indices is None else len(chosen_indices))
    return {'n_deals': count, 'full_population': chosen_indices is None, 'all_scores_agree': True,
            'fresh_encoded_private_keys': len(extra), 'queried_keys_outside_pruned_tree': sum(k not in lookup for k in extra), 'hero_score_sum': total,
            'path': 'Fresh manual deals/legal/winners/transitions, saved private-key policy; network completion outside zero-pruned tree.'}


def condition(g):
    tag = f'S{g.S}R4n{g.n}_t{g.trump}'; folder = OUT/tag; folder.mkdir(exist_ok=True)
    tree, cache_sha = load_tree(g); assert tree.g == g
    target_manifest = json.loads((ROUND/'M1/results'/tag/'manifest.json').read_text())
    assert cache_sha == target_manifest['cache_sha256']
    if g.n == 3:
        baseline_path = D3/'results'/f'review_pure_trump_{g.trump}.npz'
        baseline = np.load(baseline_path)['actions']
        check_scores = private_scores(tree, baseline)
        assert Fraction(int(check_scores.sum()), tree.n_deals) == OLD_UPPERS[(g.n, g.trump)]
        save(folder/'private_DP_baseline_gate.json', {'status': 'pass', 'exact_mean': str(OLD_UPPERS[(g.n, g.trump)]),
            'baseline_policy_sha256': sha(baseline_path), 'n_exact_deals': tree.n_deals})
    roots = tree.roots[tree.node_pl[tree.roots] >= 0]; root_keys = np.unique(tree.node_is[roots])
    nonroot = np.ones(tree.n_nodes, bool); nonroot[tree.roots] = False
    later_keys = np.unique(tree.node_is[nonroot & (tree.node_pl >= 0)])
    overlap = np.intersect1d(root_keys, later_keys)
    save(folder/'reach_semantics.json', {'n_root_keys': len(root_keys), 'n_later_keys': len(later_keys),
        'n_root_keys_also_later': len(overlap), 'interpretation': 'Root selection is a relaxation, no hidden root-stage flag allowed for deployed policy.'})
    x, feature_receipt = feature_cache(tree, folder, tag)
    keys_file = folder/'private_keys.bin.gz'
    if not keys_file.exists():
        temp = keys_file.with_suffix('.tmp')
        with gzip.open(temp, 'wb') as f:
            for key in tree.is_key.packed: f.write(bytes([len(key)])+key)
        replace(temp, keys_file)
    reports = []
    for objective in ('hard_ce', 'set_bce'):
        for seed in range(5):
            if time.time() >= STOP: raise TimeoutError('M9 policy stop')
            name = f'{objective}_s{seed}'; receipt_path = folder/f'{name}.json'
            model_path = ROUND/'M3/results'/f'{tag}_mlp_s{seed}_{objective}'/'selected.npz'
            model_meta = json.loads((model_path.parent/'frozen.json').read_text()); assert sha(model_path) == model_meta['model_sha256']
            if receipt_path.exists():
                saved = json.loads(receipt_path.read_text()); assert saved['model_sha256'] == sha(model_path) and saved['policy_sha256'] == sha(folder/f'{name}.npz')
                reports.append(saved); continue
            a = dict(np.load(model_path)); params = [a[f'p{i}'] for i in range(6)]; action = np.empty(tree.n_is, np.uint8)
            for start in range(0, tree.n_is, 4096):
                if time.time() >= STOP: raise TimeoutError('M9 action stop')
                end = min(start+4096, tree.n_is)
                action[start:end] = decisions(x[start:end], tree.is_valid[start:end, :3], params, objective)
            scores = private_scores(tree, action); total = int(scores.sum()); mean = Fraction(total, tree.n_deals)
            assert mean >= Fraction(target_manifest['J1_fraction'])
            checkpoint(folder/f'{name}.npz', actions=action, root_scores=scores)
            selected = np.linspace(0, tree.n_deals-1, 128, dtype=int).tolist()
            verification = fresh_forward(tree, params, objective, action, selected, scores, folder, name)
            if mean < OLD_UPPERS[(g.n, g.trump)]:
                verification = fresh_forward(tree, params, objective, action, None, scores, folder, name+'_full')
            report = {'condition': tag, 'objective': objective, 'seed_index': seed, 'n_exact_deals': tree.n_deals,
                'n_private_keys': tree.n_is, 'hero_score_sum': total, 'exact_mean': str(mean),
                'hero_mean': float(mean), 'prior_feasible_upper': str(OLD_UPPERS[(g.n, g.trump)]),
                'beats_prior_upper': mean < OLD_UPPERS[(g.n, g.trump)], 'fresh_forward_verification': verification,
                'model_sha256': sha(model_path), 'policy_sha256': sha(folder/f'{name}.npz'),
                'private_keys_sha256': sha(keys_file), 'tree_cache_sha256': cache_sha, 'feature_cache': feature_receipt,
                'scope': 'Fully private feasible pure profile with fixed hero/bids; all-deal evaluation mixes trained/held teammate hands, not held-out team generalization.', 'utc': stamp()}
            save(receipt_path, report); reports.append(report)
            save(OUT/'progress.json', {'condition': tag, 'completed_policy': name, 'exact_mean': str(mean), 'utc': stamp()})
            log(HERE, f'{tag}/{name}: exact{mean}, n{tree.n_deals}deals; fresh forward{verification["n_deals"]}; no global optimality claim.')
    del tree, x; gc.collect(); clear_mid()
    return reports


def main():
    assert time.time() < STOP; OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    source_paths = [Path(__file__), HERE/'PROTOCOL.md', ROUND/'root_features.py', ROUND/'learning.py', ROUND/'common.py',
                    ROUND/'M1/verify.py', D3/'compact_tree.py', D3/'bounded_ta.py', ENGINE/'judgement/cfr/toy.py', ENGINE/'judgement/cfr/heuristics.py']
    source = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in source_paths}
    if (OUT/'source_freeze.json').exists(): assert json.loads((OUT/'source_freeze.json').read_text()) == source
    else: save(OUT/'source_freeze.json', source)
    timer = threading.Timer(STOP-time.time(), lambda: os._exit(124)); timer.daemon = True; timer.start()
    try:
        rows = []
        for S, n in ((2, 2), (3, 3)):
            for trump in (0, -1): rows.extend(condition(ToyGame(N=4, S=S, R=4, n=n, trump=trump)))
        contrasts = {}
        for tag in sorted(set(r['condition'] for r in rows)):
            delta = [next(r['hero_mean'] for r in rows if r['condition'] == tag and r['seed_index'] == s and r['objective'] == 'set_bce')-
                     next(r['hero_mean'] for r in rows if r['condition'] == tag and r['seed_index'] == s and r['objective'] == 'hard_ce') for s in range(5)]
            mean = float(np.mean(delta)); half = float(student_t.ppf(.975, 4)*np.std(delta, ddof=1)/np.sqrt(5))
            contrasts[tag] = {'mean': mean, 'ci95': [mean-half, mean+half], 'n_training_seeds': 5, 'paired_differences': delta,
                'unit': 'Optimization seeds, all chance deals exact perfit, not an IID deal interval'}
        save(OUT/'summary.json', {'rows': rows, 'paired_set_minus_hard': contrasts, 'source_sha256': source, 'utc': stamp(), 'status': 'Independent review pending'})
    finally: timer.cancel()


if __name__ == '__main__': main()
