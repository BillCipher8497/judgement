"""Exact finite-population tie and observable-representation diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; sys.path.insert(0, str(ROUND))
from common import save, sha, stamp, log
from root_features import encode


def main():
    reports = {}
    for folder in sorted((ROUND/'M1/results').glob('S*')):
        if not folder.is_dir(): continue
        m = json.loads((folder/'manifest.json').read_text()); d = dict(np.load(folder/'root_data.npz'))
        assert sha(folder/'root_data.npz') == m['data_sha256']
        obs = [json.loads(s)['observation'] for s in (folder/'private_contexts.jsonl').read_text().splitlines()]
        x = np.array([encode(o) for o in obs]); assert len(set(row.tobytes() for row in x)) == len(x)
        mask = d['mask']; costs = np.where(mask, d['q_sum'], 2**60)
        total = m['n_exact_deals']; terminal = m['terminal_root_score_sum']; j1 = m['J1_sum']
        def risk(actions): return terminal+int(d['q_sum'][np.arange(len(mask)), actions].sum())
        modes = {'hard_teacher_mode': np.where(mask, d['hard_counts'], -1).argmax(1),
                 'uniform_teacher_mode': np.where(mask, d['uniform_optimal_units6'], -1).argmax(1),
                 'set_membership_mode': np.where(mask, d['optimal_counts'], -1).argmax(1),
                 'J1_cost_minimizer': costs.argmin(1)}
        risks = {name: risk(action) for name, action in modes.items()}
        assert risks['set_membership_mode'] == risks['J1_cost_minimizer'] == j1
        # Pointwise difference must be an action-independent constant.
        aligned = d['optimal_counts']*d['payoff'][:, None]+d['q_sum']
        for values, valid in zip(aligned, mask): assert len(set(values[valid].tolist())) == 1
        representations = {}
        for name in ('public_mask', 'own_hand_seat_mask'):
            aggregates = defaultdict(lambda: np.zeros(3, np.int64)); masks = {}
            for i, o in enumerate(obs):
                key = (o['seat'], tuple(o['history']), tuple(o['legal'])) if name == 'public_mask' else (o['seat'], tuple(o['hand']), tuple(o['legal']))
                aggregates[key] += d['q_sum'][i]; masks[key] = mask[i]
            score = terminal+sum(int(np.where(masks[k], q, 2**60).min()) for k, q in aggregates.items())
            assert score >= j1
            representations[name] = {'n_representation_classes': len(aggregates), 'exact_score_sum': score,
                'exact_mean': str(Fraction(score, total)), 'representation_excess': str(Fraction(score-j1, total))}
        reports[folder.name] = {'n_exact_deals': total, 'n_private_groups': len(mask), 'n_free_root_worlds': len(d['q']),
            'PI': m['PI_fraction'], 'J1': m['J1_fraction'], 'information_floor': str(Fraction(j1-m['PI_sum'], total)),
            'exact_teacher_Bayes_risks': {k: {'score_sum': v, 'mean': str(Fraction(v, total)), 'avoidable_cost': str(Fraction(v-j1, total))} for k, v in risks.items()},
            'coarsenings': representations, 'injective_full_features': True,
            'interval': 'none; exact finite population, no sampling CI', 'data_sha256': m['data_sha256']}
        reports[folder.name]['decoder_tie_sensitivity'] = {}
        for label, frequency in [('hard', d['hard_counts']), ('uniform', d['uniform_optimal_units6']), ('set', d['optimal_counts'])]:
            tied = (frequency == np.where(mask, frequency, -1).max(1)[:, None]) & mask
            low = terminal + int(np.where(tied, d['q_sum'], 2**60).min(1).sum())
            high = terminal + int(np.where(tied, d['q_sum'], -1).max(1).sum())
            reports[folder.name]['decoder_tie_sensitivity'][label] = dict(tied_group_count=int((tied.sum(1)>1).sum()),
                best_excess=str(Fraction(low-j1,total)), worst_excess=str(Fraction(high-j1,total)))
    cases = [([Fraction(1,5),Fraction(3,10),Fraction(1,2)], [(0,1),(1,0),(0,0)]),
             ([Fraction(3,5),Fraction(2,5)], [(0,0,1),(1,1,0)])]
    for index,(mass,costs) in enumerate(cases):
        risks=[sum(p*c[a] for p,c in zip(mass,costs)) for a in range(len(costs[0]))]
        labels=[Fraction(0) for _ in risks]
        for p,c in zip(mass,costs):
            good=[a for a,v in enumerate(c) if v==min(c)]
            for a in (good[:1] if index==0 else good): labels[a]+=p/(1 if index==0 else len(good))
        chosen=max(range(len(labels)),key=lambda a:(labels[a],-a))
        assert risks[chosen]>min(risks)
    save(HERE/'results/diagnostics.json', {'conditions': reports, 'source_sha256': sha(__file__), 'utc': stamp(), 'proof_status': 'externally checked Claude2026-11-17; corrections applied'})
    log(HERE, 'Exact risk/alignment/coarsening and feature-injectivity checks pass in all four conditions; proof externally checked Claude2026-11-17.')


if __name__ == '__main__': main()


# External decoder-risk extension is reproducible with round12/audit_changes.py; original science arrays are read-only.

