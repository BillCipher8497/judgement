"""Exploratory exact sensitivity to every global teacher tie priority."""
import json
import sys
from fractions import Fraction
from itertools import permutations
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; sys.path.insert(0, str(ROUND))
from common import save, sha, stamp


def main():
    reports = {}
    for folder in sorted((ROUND/'M1/results').iterdir()):
        if not folder.is_dir() or not (folder/'manifest.json').exists(): continue
        m = json.loads((folder/'manifest.json').read_text()); d = dict(np.load(folder/'root_data.npz'))
        q, group = d['q'], d['root_group']; G = len(d['count']); good = q == q.min(1, keepdims=True)
        totals = np.zeros((G, 3), np.int64); rows = []
        for order in permutations(range(m['n'])):
            selected = np.array(order)[np.argmax(good[:, order], axis=1)]
            counts = np.column_stack([np.bincount(group, weights=(selected == a), minlength=G).astype(np.int64) for a in range(3)])
            totals += counts
            mode = np.where(d['mask'], counts, -1).argmax(1)
            score = m['terminal_root_score_sum']+int(d['q_sum'][np.arange(G), mode].sum())
            rows.append({'priority': order, 'score_sum': score, 'exact_mean': str(Fraction(score, m['n_exact_deals'])),
                         'Bayes_excess_above_J1': str(Fraction(score-m['J1_sum'], m['n_exact_deals']))})
        np.testing.assert_array_equal(totals*6, d['uniform_optimal_units6']*len(rows))
        excess = [Fraction(r['Bayes_excess_above_J1']) for r in rows]
        reports[folder.name] = {'n_exact_deals': m['n_exact_deals'], 'n_global_priorities': len(rows), 'priorities': rows,
            'min_exact_Bayes_excess': str(min(excess)), 'max_exact_Bayes_excess': str(max(excess)),
            'averaged_priority_labels_equal_uniform_optimal': True,
            'scope': 'Exploratory complete priority enumeration afterM3outcomes; no new fitting or retuning, no sampling interval.'}
    save(HERE/'results/tie_priority_sensitivity.json', {'conditions': reports, 'source_sha256': sha(__file__), 'utc': stamp()})


if __name__ == '__main__': main()
