"""Independent arithmetic/forward/selection replay of frozen root experiments."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import hashlib
import json
import math
import statistics
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROUND))
from common import save, sha, stamp, log


def independent_encode(o):
    assert set(o) == {'seat', 'hand', 'history', 'n', 'S', 'R', 'trump', 'legal'}
    def onehot(k, size): return [int(i == k) for i in range(size)]
    own = o['hand']; hist = o['history']; bits = [int(i in own) for i in range(12)]
    bits += sum([onehot(own[i] if i < len(own) else 12, 13) for i in range(3)], [])
    bits += sum([onehot(b, 5) for b in hist[:4]], [])
    bits += sum([onehot(hist[i+4] if i+4 < len(hist) else 12, 13) for i in range(12)], [])
    bits += onehot(o['seat'], 4)+onehot(3 if o['trump'] == -1 else o['trump'], 4)
    bits += [o['n']/3, o['S']/3, o['R']/4]+[int(i in o['legal']) for i in range(3)]
    assert len(bits) == 241
    return bits


def independent_actions(x, mask, params, objective):
    actions = np.empty(len(x), np.int64)
    for start in range(0, len(x), 2048):
        z = x[start:start+2048]
        for k in range(0, len(params), 2):
            z = np.matmul(z, params[k])+params[k+1]
            if k+2 < len(params): z = np.tanh(z)
        if objective == 'q_reg': actions[start:start+len(z)] = np.argmin(np.where(mask[start:start+len(z)], z, np.inf), axis=1)
        else: actions[start:start+len(z)] = np.argmax(np.where(mask[start:start+len(z)], z, -np.inf), axis=1)
    return actions


def ci(values):
    n = len(values); mean = statistics.mean(values); half = 2.7764451051977987*statistics.stdev(values)/math.sqrt(n)
    return {'n_training_seeds': n, 'mean': mean, 'ci95': [mean-half, mean+half]}


def main():
    recorded = json.loads((ROUND/'M3/results/held_out_results.json').read_text()); freeze = json.loads((ROUND/'M3/results/source_freeze.json').read_text())
    for rel, h in freeze.items(): assert sha(ROOT/rel) == h
    matched_rows = 0; matched_models = 0; primary = {'discovery': {}, 'confirmation': {}}
    for directory in sorted((ROUND/'M1/results').iterdir()):
        if not directory.is_dir() or not (directory/'manifest.json').exists(): continue
        meta = json.loads((directory/'manifest.json').read_text()); d = dict(np.load(directory/'root_data.npz'))
        assert sha(directory/'root_data.npz') == meta['data_sha256'] and sha(directory/'private_contexts.jsonl') == meta['contexts_sha256']
        obs = [json.loads(line)['observation'] for line in (directory/'private_contexts.jsonl').read_text().splitlines()]
        x = np.array([independent_encode(o) for o in obs]); group = d['root_group']; q = d['q']; G = len(obs)
        count = np.bincount(group, minlength=G); np.testing.assert_array_equal(count, d['count'])
        costs = np.column_stack([np.bincount(group, weights=np.where(q[:, a] == 127, 0, q[:, a]), minlength=G).astype(np.int64) for a in range(3)])
        np.testing.assert_array_equal(costs, d['q_sum']); np.testing.assert_array_equal(np.array([[a in o['legal'] for a in range(3)] for o in obs]), d['mask'])
        best = q.min(1); membership = np.column_stack([np.bincount(group, weights=(q[:, a] == best), minlength=G).astype(np.int64) for a in range(3)])
        membership *= d['mask']; np.testing.assert_array_equal(membership, d['optimal_counts'])
        aligned = membership*d['payoff'][:, None]+costs
        assert all(len(set(r[mask].tolist())) == 1 for r, mask in zip(aligned, d['mask']))
        hands = list(combinations(range(meta['S']*4), meta['n']))
        hands.sort(key=lambda h: hashlib.sha256(json.dumps([2000000000, meta['S'], 4, meta['n'], list(h)], separators=(',', ':')).encode()).digest())
        cuts = [int(.6*len(hands)), int(.8*len(hands)), int(.9*len(hands))]
        partition = {h: sum(i >= c for c in cuts) for i, h in enumerate(hands)}
        np.testing.assert_array_equal([partition[tuple(o['hand'])] for o in obs], d['split'])
        for folder in sorted((ROUND/'M3/results').glob(directory.name+'_*')):
            if not folder.is_dir(): continue
            r = json.loads((folder/'frozen.json').read_text()); saved = dict(np.load(folder/'selected.npz'))
            assert sha(folder/'selected.npz') == r['model_sha256'] and r['source_sha256'] == freeze
            params = [saved[f'p{i}'] for i in range(len(saved))]; actions = independent_actions(x, d['mask'], params, r['objective'])
            np.testing.assert_array_equal(actions, np.load(folder/'evaluation_actions.npz')['actions'])
            traces = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
            assert [s['epoch'] for s in traces] == list(range(1, 101))
            va = np.flatnonzero(d['split'] == 1); candidates = []
            for epoch in (10, 25, 50, 100):
                a = dict(np.load(folder/f'candidate_{epoch}.npz')); p = [a[f'p{i}'] for i in range(len(a))]
                act = independent_actions(x[va], d['mask'][va], p, r['objective']); total = int(costs[va, act].sum())
                assert next(t['validation_score_sum'] for t in traces if t['epoch'] == epoch) == total
                candidates.append((total, epoch))
                if epoch == r['selected_epoch']:
                    for left, right in zip(params, p): np.testing.assert_array_equal(left, right)
            assert min(candidates) == (r['validation_score_sum'], r['selected_epoch'])
            for split, label in ((2, 'discovery'), (3, 'confirmation')):
                worlds = np.flatnonzero(d['split'][group] == split); score = int(q[worlds, actions[group[worlds]]].sum())
                ids = np.flatnonzero(d['split'] == split); j1 = int(np.where(d['mask'][ids], costs[ids], 2**60).min(1).sum())
                matched = next(row for row in recorded['rows'] if row['condition'] == directory.name and row['architecture'] == r['architecture'] and row['seed_index'] == r['seed_index'] and row['objective'] == r['objective'] and row['split'] == label)
                assert score == matched['hero_score_sum'] and j1 == matched['J1_score_sum'] and len(worlds) == matched['n_root_worlds']
                weighted_membership_regret = int((d['payoff'][ids]*(membership[ids].max(1)-membership[ids, actions[ids]])).sum())
                assert weighted_membership_regret == score-j1
                if directory.name == 'S3R4n3_t-1' and r['architecture'] == 'mlp': primary[label][(r['seed_index'], r['objective'])] = (score-j1)/len(worlds)
                matched_rows += 1
            matched_models += 1
        save(HERE/'results/root_verification_progress.json', {'condition': directory.name, 'models_verified': matched_models, 'rows_verified': matched_rows, 'utc': stamp()})
    contrasts = {}
    for label, arms in primary.items():
        contrasts[label] = ci([arms[(s, 'set_bce')]-arms[(s, 'hard_ce')] for s in range(5)])
        for k in ('mean', 'ci95'): np.testing.assert_allclose(contrasts[label][k], recorded['primary'][label][k], rtol=0, atol=2e-13)
    assert matched_models == 200 and matched_rows == 400
    save(HERE/'results/root_verification.json', {'status': 'pass', 'n_models': matched_models, 'n_held_out_rows': matched_rows,
        'primary_recomputed_from_world_costs': contrasts, 'checks': ['independent241encoder', 'raw-world group sums/membership identity',
        'own-hand SHA partitions', 'independent forward/decoder', 'all four candidate validation selections', 'all100epochs', 'raw-world held-out integer scores', 'seed-t95CIs'],
        'source_sha256': sha(__file__), 'utc': stamp(), 'external_audit': False})
    log(HERE, 'Independent root arithmetic/forward/selection replay passed200models/400held-out rows; M15 external gate remains.')
    import runpy
    runpy.run_path(str(HERE/'tie_sensitivity.py'), run_name='__main__')


if __name__ == '__main__': main()
