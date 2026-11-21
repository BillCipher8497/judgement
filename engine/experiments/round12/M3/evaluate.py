"""Evaluate only after all200 fits/validation selections have frozen receipts."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import t
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; sys.path.insert(0, str(ROUND))
from common import save, sha, stamp, log
from learning import decisions
from root_features import encode


def interval(values):
    x = np.asarray(values, float); n = len(x); mean = float(x.mean())
    half = float(t.ppf(.975, n-1)*x.std(ddof=1)/np.sqrt(n))
    return {'n_training_seeds': n, 'mean': mean, 'ci95': [mean-half, mean+half], 'half95': half,
            'unit': 'independent optimization seeds, paired for objective differences; fixed held-out population exact per fit'}


def main():
    out = HERE/'results'; frozen = json.loads((out/'training_complete.json').read_text())
    assert frozen['n_frozen_models'] == 200
    results = []; summaries = {}; primary = {}
    for tag in ('S2R4n2_t0', 'S2R4n2_t-1', 'S3R4n3_t0', 'S3R4n3_t-1'):
        directory = ROUND/'M1/results'/tag; manifest = json.loads((directory/'manifest.json').read_text())
        assert sha(directory/'root_data.npz') == manifest['data_sha256']
        data = dict(np.load(directory/'root_data.npz'))
        obs = [json.loads(s)['observation'] for s in (directory/'private_contexts.jsonl').read_text().splitlines()]
        x = np.array([encode(o) for o in obs])
        for arch in ('linear', 'mlp'):
            arm_values = {}
            for si in range(5):
                for objective in ('hard_ce', 'uniform_ce', 'set_bce', 'q_reg', 'direct'):
                    name = f'{tag}_{arch}_s{si}_{objective}'; folder = out/name
                    assert sha(folder/'frozen.json') == frozen['receipts_sha256'][name]
                    receipt = json.loads((folder/'frozen.json').read_text())
                    assert sha(folder/'selected.npz') == receipt['model_sha256']
                    assert receipt['data_sha256'] == manifest['data_sha256']
                    saved = dict(np.load(folder/'selected.npz')); params = [saved[f'p{i}'] for i in range(len(saved))]
                    chosen = decisions(x, data['mask'], params, objective)
                    assert np.all(data['mask'][np.arange(len(chosen)), chosen])
                    with (folder/'evaluation_actions.npz').open('wb') as f: np.savez_compressed(f, actions=chosen.astype(np.uint8))
                    for split, label in ((2, 'discovery'), (3, 'confirmation')):
                        ids = np.flatnonzero(data['split'] == split); counts = int(data['count'][ids].sum())
                        score = int(data['q_sum'][ids, chosen[ids]].sum())
                        j1 = int(np.where(data['mask'][ids], data['q_sum'][ids], 2**60).min(1).sum())
                        pi = int(data['pi_by_group'][ids].sum()); regret = (score-j1)/counts
                        assert score >= j1 >= pi
                        row = {'condition': tag, 'architecture': arch, 'seed_index': si, 'objective': objective,
                            'split': label, 'n_root_worlds': counts, 'n_private_groups': len(ids),
                            'hero_score_sum': score, 'J1_score_sum': j1, 'PI_score_sum': pi,
                            'hero_score_exact': score/counts, 'avoidable_cost_exact': regret, 'information_floor_exact': (j1-pi)/counts,
                            'hard_teacher_agreement_exact': int(data['hard_counts'][ids, chosen[ids]].sum())/counts,
                            'optimal_set_agreement_exact': int(data['optimal_counts'][ids, chosen[ids]].sum())/counts,
                            'selected_epoch': receipt['selected_epoch'], 'model_sha256': receipt['model_sha256'],
                            'population': 'All first-free-decision worlds whose acting own hand is in this held-out partition; terminal roots excluded, no IID world CI.'}
                        results.append(row); arm_values[(label, objective, si)] = regret
            for label in ('discovery', 'confirmation'):
                summaries[f'{tag}/{arch}/{label}'] = {
                    'objectives': {o: interval([arm_values[(label, o, s)] for s in range(5)]) for o in ('hard_ce', 'uniform_ce', 'set_bce', 'q_reg', 'direct')},
                    'paired_minus_hard_ce': {o: interval([arm_values[(label, o, s)]-arm_values[(label, 'hard_ce', s)] for s in range(5)]) for o in ('uniform_ce', 'set_bce', 'q_reg', 'direct')}}
                if tag == 'S3R4n3_t-1' and arch == 'mlp': primary[label] = summaries[f'{tag}/{arch}/{label}']['paired_minus_hard_ce']['set_bce']
    assert len(results) == 400
    save(out/'held_out_results.json', {'rows': results, 'summaries': summaries, 'primary': primary,
        'primary_passes_both': all(v['ci95'][1] < 0 for v in primary.values()), 'n_fits': 200,
        'source_sha256': sha(__file__), 'training_freeze_sha256': sha(out/'training_complete.json'), 'utc': stamp(),
        'scope': 'Seed uncertainty only; no claim about unseen hand populations, full decentralized teams,52-card performance or general distillation.'})
    log(HERE, 'Separate held-out evaluation complete: '+json.dumps(primary)+'; no fitting or selection changed.')


if __name__ == '__main__': main()
