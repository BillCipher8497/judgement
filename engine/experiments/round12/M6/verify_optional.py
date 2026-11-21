"""Separate raw-score/CI audit of optional fixed learning/deployment studies."""
import json
import math
import statistics
import sys
from fractions import Fraction
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROUND)]
from common import save, sha, stamp
from verify_root import independent_encode, independent_actions, ci


def main():
    base = ROUND/'M1/results/S3R4n3_t-1'; d = dict(np.load(base/'root_data.npz'))
    obs = [json.loads(s)['observation'] for s in (base/'private_contexts.jsonl').read_text().splitlines()]; x = np.array([independent_encode(o) for o in obs])
    freeze = json.loads((ROUND/'M7/results/freeze.json').read_text()); recorded = json.loads((ROUND/'M7/results/summary.json').read_text())
    for rel, h in freeze['source_sha256'].items(): assert sha(ROOT/rel) == h
    va = np.flatnonzero(d['split'] == 1); values = {}; models = 0
    for folder in sorted((ROUND/'M7/results').iterdir()):
        if not folder.is_dir() or not (folder/'frozen.json').exists(): continue
        r = json.loads((folder/'frozen.json').read_text()); assert sha(folder/'selected.npz') == r['model_sha256']
        assert r['source_sha256'] == freeze['source_sha256']; selected_ids = freeze['group_order'][:int(r['fraction']*len(freeze['group_order']))]
        assert r['n_training_groups'] == len(selected_ids) and r['n_training_worlds'] == int(d['count'][selected_ids].sum())
        assert np.all(d['split'][selected_ids] == 0)
        params = dict(np.load(folder/'selected.npz')); params = [params[f'p{i}'] for i in range(6)]
        actions = independent_actions(x, d['mask'], params, r['objective']); np.testing.assert_array_equal(actions, np.load(folder/'evaluation_actions.npz')['actions'])
        choices = []
        for epoch in (10, 25, 50, 100):
            saved = dict(np.load(folder/f'candidate_{epoch}.npz')); p = [saved[f'p{i}'] for i in range(6)]
            a = independent_actions(x[va], d['mask'][va], p, r['objective']); total = int(d['q_sum'][va, a].sum()); choices.append((total, epoch))
        assert min(choices) == (r['validation_score_sum'], r['selected_epoch'])
        traces = [json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()]; assert [t['epoch'] for t in traces] == list(range(1, 101))
        for split, label in ((2, 'discovery'), (3, 'confirmation')):
            worlds = np.flatnonzero(d['split'][d['root_group']] == split); score = int(d['q'][worlds, actions[d['root_group'][worlds]]].sum())
            row = next(z for z in recorded['rows'] if z['fraction'] == r['fraction'] and z['seed_index'] == r['seed_index'] and z['objective'] == r['objective'] and z['split'] == label)
            assert score == row['score_sum'] and len(worlds) == row['n_root_worlds']
            values[(r['fraction'], label, r['objective'], r['seed_index'])] = (score-row['J1_sum'])/len(worlds)
        models += 1
    for fraction in (.25, .5, 1.):
        for label in ('discovery', 'confirmation'):
            result = ci([values[(fraction, label, 'set_bce', s)]-values[(fraction, label, 'hard_ce', s)] for s in range(5)])
            expected = recorded['summary'][f'{fraction}/{label}']['paired_set_minus_hard']
            np.testing.assert_allclose(result['ci95'], expected['ci95'], rtol=0, atol=2e-13)
    assert models == 30
    source = json.loads((ROUND/'M9/results/source_freeze.json').read_text()); reports = json.loads((ROUND/'M9/results/summary.json').read_text()); raw_values = {}
    for rel, h in source.items(): assert sha(ROOT/rel) == h
    for row in reports['rows']:
        folder = ROUND/'M9/results'/row['condition']; path = folder/f"{row['objective']}_s{row['seed_index']}.npz"
        assert sha(path) == row['policy_sha256']; data = dict(np.load(path)); scores = data['root_scores']; actions = data['actions']
        assert len(scores) == row['n_exact_deals'] and len(actions) == row['n_private_keys']
        assert set(np.unique(scores)).issubset({0, 10, 11, 12, 13}) and actions.max() <= 2
        total = sum(map(int, scores)); assert total == row['hero_score_sum'] and str(Fraction(total, len(scores))) == row['exact_mean']
        assert sha(folder/'private_keys.bin.gz') == row['private_keys_sha256']
        model = ROUND/'M3/results'/f"{row['condition']}_mlp_s{row['seed_index']}_{row['objective']}"/'selected.npz'
        assert sha(model) == row['model_sha256']; check = row['fresh_forward_verification']
        assert check['all_scores_agree'] and check['n_deals'] >= 128
        if row['beats_prior_upper']: assert check['full_population'] and check['n_deals'] == len(scores)
        raw_values[(row['condition'], row['objective'], row['seed_index'])] = total/len(scores)
    for tag, expected in reports['paired_set_minus_hard'].items():
        result = ci([raw_values[(tag, 'set_bce', s)]-raw_values[(tag, 'hard_ce', s)] for s in range(5)])
        np.testing.assert_allclose(result['ci95'], expected['ci95'], rtol=0, atol=2e-13)
    assert len(reports['rows']) == 40
    save(HERE/'results/optional_verification.json', {'status': 'pass', 'M7_models': models, 'M9_profiles': len(reports['rows']),
        'checks': ['independentM7forward/raw-world totals/validation choices', 'training subset split/n', 'allM7pairedseedCIs',
                   'M9allrawscorevectors/exactfractions/modelpolicykeyhashes', 'M9requiredfreshforwardreceipts', 'allM9pairedseedCIs'],
        'limitation': 'M9fresh forward receipt checked, not repeated; original fresh driver reconstructs cards/winners and network encoding on128presetdeals/profile.',
        'source_sha256': sha(__file__), 'utc': stamp(), 'external_audit': False})


if __name__ == '__main__': main()
