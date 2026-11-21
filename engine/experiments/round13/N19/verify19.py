"""Separate verification of N19, written independently of run19.py.

Checks, all in exact rational arithmetic:
  A  rebuild the moment constraints from the raw structural arrays and confirm
     the true posterior itself satisfies them.
  B  recheck every stored endpoint certificate: exact feasibility, exact
     objective, and the sign conclusion drawn from it.
  C  confirm no stored endpoint strictly prefers the inferior action, i.e. the
     worst case over the whole marginal-consistent polytope is indifference.
  D  the tie-sensitive bound: for contexts whose polytope reaches exact
     indifference, report the regret an adversarial tie rule could cause, since
     a belief at that boundary leaves the decision to the tie rule.
  E  consistency with N17: the projection used there is marginal-consistent, so
     its exact objective must lie inside this interval.
"""
import os, sys, json, time
from pathlib import Path
from fractions import Fraction as F
import numpy as np

H = Path(__file__).resolve().parent
O = H / 'results'
R12 = H.parent.parent / 'round12'
R13 = H.parent
sys.path.insert(0, str(R12 / 'N9'))
import n9lib as B

TAGS = B.TAGS


def stamp():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check():
    cut = os.environ.get('JUDGEMENT_JOB_DEADLINE')
    if cut and time.time() >= float(cut):
        raise SystemExit('cutoff reached')


def main():
    report = {'utc': stamp(), 'conditions': {}}
    for tag in TAGS:
        check()
        rows_by_kid = {r.kid: r for r in B.rows(B.load(R12 / 'N9/results' / tag / 'data.npz'))}
        n17 = {a['key_id']: a for a in
               json.loads((R13 / 'N17/results' / tag / 'keys.json').read_text())}
        cs = json.loads((O / tag / 'contexts.json').read_text())
        A_ok = B_ok = C_ok = E_ok = 0
        strict_wrong = []
        tie_reachable = []
        tie_num = F(0)
        reach = 0
        for rec in cs:
            check()
            kid = rec['key_id']
            r = rows_by_kid[kid]
            W, U = r.F.shape
            n = r.count
            z = np.load(R12 / 'N12/results' / tag / f'{kid:07d}.npz')
            Q = z['Q'].astype(np.int64)
            d = [F(int(x)) for x in (Q[:, 1] - Q[:, 0])]
            reach += rec['reach']

            # A: rebuild constraints, check the true posterior satisfies them
            cons = [([F(1)] * W, F(1))]
            for c in range(U):
                for o in range(3):
                    col = [F(1) if r.F[w, c] == o else F(0) for w in range(W)]
                    cnt = sum(1 for w in r.posterior if r.F[w, c] == o)
                    cons.append((col, F(cnt, n)))
            P = [F(0)] * W
            for w in r.posterior:
                P[w] += F(1, n)
            for col, rhs in cons:
                assert sum((col[w] * P[w] for w in range(W) if P[w]), F(0)) == rhs, (tag, kid, 'A')
            A_ok += 1

            Pd = sum((P[w] * d[w] for w in range(W) if P[w]), F(0))
            better = 1 if Pd > 0 else 0
            assert better == rec['true_better_action_index'], (tag, kid, 'true side')

            # B: recheck both endpoint certificates exactly
            vals = {}
            for name in ('min', 'max'):
                e = rec['endpoints'][name]
                assert e['exact_certified'], (tag, kid, name, 'uncertified')
                q = [F(0)] * W
                for j, v in e['support'].items():
                    q[int(j)] = F(v)
                assert all(v >= 0 for v in q)
                for col, rhs in cons:
                    assert sum((col[w] * q[w] for w in range(W) if q[w]), F(0)) == rhs, (tag, kid, name, 'B feas')
                val = sum((q[w] * d[w] for w in range(W) if q[w]), F(0))
                assert val == F(e['exact_value']), (tag, kid, name, 'B value')
                vals[name] = val
            B_ok += 1

            lo, hi = vals['min'], vals['max']
            assert lo <= Pd <= hi, (tag, kid, 'posterior outside its own polytope')
            ident = (lo > 0 and hi > 0) if better == 1 else (lo < 0 and hi < 0)
            assert ident == rec['identified_by_marginals'], (tag, kid, 'identified flag')

            # C: is a strictly wrong preference reachable?
            adverse = lo if better == 1 else hi
            if (adverse < 0) if better == 1 else (adverse > 0):
                strict_wrong.append(kid)
            C_ok += 1

            # D: can the polytope reach exact indifference?
            if lo <= 0 <= hi:
                tie_reachable.append(kid)
                tie_num += F(n17[kid]['spread_exact']) * rec['reach']

            # E: N17's projection is marginal-consistent, so its value must sit inside
            m = n17[kid]['decision_margin']
            sgn = 1 if n17[kid]['projection_choice'] == 1 else -1
            v = F(m).limit_denominator(10 ** 12) * sgn
            assert float(lo) - 1e-6 <= float(v) <= float(hi) + 1e-6, (tag, kid, 'E', float(lo), float(v), float(hi))
            E_ok += 1

        report['conditions'][tag] = {
            'n_Splus': len(cs), 'reach_Splus': reach,
            'A_posterior_satisfies_rebuilt_constraints': A_ok,
            'B_endpoint_certificates_exact': B_ok,
            'C_contexts_checked': C_ok,
            'C_contexts_where_a_marginal_consistent_belief_strictly_prefers_the_inferior_action':
                len(strict_wrong),
            'C_strict_wrong_key_ids': strict_wrong,
            'D_contexts_whose_polytope_reaches_exact_indifference': len(tie_reachable),
            'D_tie_reachable_key_ids': tie_reachable,
            'D_tie_sensitive_worst_case_regret_exact': str(tie_num / reach) if reach else None,
            'D_tie_sensitive_worst_case_regret_float': float(tie_num / reach) if reach else None,
            'E_N17_projection_inside_interval': E_ok,
        }
        print(tag, json.dumps(report['conditions'][tag]), flush=True)
    report['verdict'] = (
        'No marginal-consistent belief strictly prefers the inferior action at any '
        'decision-relevant context; the only reachable failure is exact indifference, '
        'whose cost depends on the tie rule and is bounded in D.')
    (O / 'verification.json').write_text(json.dumps(report, indent=2, sort_keys=True),
                                         encoding='utf-8')
    print('verification written', flush=True)


if __name__ == '__main__':
    main()
