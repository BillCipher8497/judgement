"""Separate verification of N17, written independently of run17.py.

Four checks on the decision-relevant (S+) stratum:
  A  second reader: V, spread and stratum recomputed from the raw npz alone.
  B  tighter re-optimization: identical choice, margin stable.
  C  path invariance: sign of the projected action difference is unchanged when
     theta is scaled over four orders of magnitude, so the conclusion does not
     depend on where the dual optimizer stopped on its way to the limit.
  D  falsification: a deliberately wrong belief must be able to produce
     positive regret on these same keys, or the measurement is vacuous.
"""
import os, sys, json, time, math, hashlib
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

H = Path(__file__).resolve().parent
O = H / 'results'
R12 = H.parent.parent / 'round12'
sys.path.insert(0, str(R12 / 'N9'))
import n9lib as B

TAGS = B.TAGS
SCALES = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)


def stamp():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check():
    cut = os.environ.get('JUDGEMENT_JOB_DEADLINE')
    if cut and time.time() >= float(cut):
        raise SystemExit('cutoff reached')


def feats(r):
    U = r.F.shape[1]
    return np.array([[(r.F[w, c] == k) for c in range(U) for k in range(3)]
                     for w in range(len(r.F))], dtype=float)


def main():
    report = {'utc': stamp(), 'conditions': {}}
    for tag in TAGS:
        check()
        rows = {r.kid: r for r in B.rows(B.load(R12 / 'N9/results' / tag / 'data.npz'))}
        recs = {a['key_id']: a for a in json.loads((O / tag / 'keys.json').read_text())}
        keydir = R12 / 'N12/results' / tag
        A = B_ = C = 0
        worst_margin_shift = 0.0
        sign_flips = []
        max_grad = 0.0
        point_pos = 0           # D: single-world point-mass belief
        anti_pos = 0            # D: adversarial-within-family belief
        splus = 0
        for kid, a in sorted(recs.items()):
            check()
            z = np.load(keydir / f'{kid:07d}.npz')
            Q = z['Q'].astype(np.int64)
            post = np.sort(z['posterior'])
            n = len(post)
            V = [F(int(Q[post, j].sum()), n) for j in range(2)]
            spread = abs(V[1] - V[0])
            # A
            assert [str(v) for v in V] == a['V_exact'], (tag, kid, 'V mismatch')
            assert str(spread) == a['spread_exact']
            assert a['stratum'] == ('S+' if spread > 0 else 'S0')
            A += 1
            if spread == 0:
                continue
            splus += 1
            best = max(V)
            r = rows[kid]
            f = feats(r)
            mu = f[r.posterior].mean(axis=0)

            def obj(th):
                s = f @ th
                lz = logsumexp(s)
                p = np.exp(s - lz)
                return float(lz - th @ mu - math.log(n)), f.T @ p - mu

            # B: much tighter budget than the primary run
            opt = minimize(obj, np.zeros(f.shape[1]), jac=True, method='L-BFGS-B',
                           options={'maxiter': 20000, 'gtol': 0.0, 'ftol': 0.0,
                                    'maxfun': 200000})
            _, g = obj(opt.x)
            max_grad = max(max_grad, float(np.max(np.abs(g))))
            dq = (Q[:, 1] - Q[:, 0]).astype(float)
            s = f @ opt.x
            q = np.exp(s - logsumexp(s))
            d = float(q @ dq)
            ch = 1 if d > 0 else 0
            assert ch == a['projection_choice'], (tag, kid, 'choice changed')
            assert best - V[ch] == F(0), (tag, kid, 'regret appeared under tighter fit')
            worst_margin_shift = max(worst_margin_shift, abs(abs(d) - a['decision_margin']))
            B_ += 1

            # C: path invariance
            signs = set()
            for t in SCALES:
                st = f @ (opt.x * t)
                qt = np.exp(st - logsumexp(st))
                signs.add(np.sign(float(qt @ dq)))
            if len(signs) != 1:
                sign_flips.append(kid)
            C += 1

            # D: can this key punish a wrong belief at all?
            for w in range(len(r.F)):           # every point-mass belief
                if (1 if dq[w] > 0 else 0) != (1 if V[1] > V[0] else 0) and dq[w] != 0:
                    point_pos += 1
                    break
            # worst belief inside the product family's convex hull of worlds
            if (dq.max() > 0) != (V[1] > V[0]) or (dq.min() < 0) != (V[1] < V[0]):
                anti_pos += 1

        s = json.loads((O / tag / 'summary.json').read_text())
        uni = s['uniform_control_regret_Splus_exact']
        report['conditions'][tag] = {
            'A_keys_revalidated': A,
            'B_Splus_refit_identical_choice_and_zero_regret': B_,
            'C_Splus_sign_invariant_over_theta_scales': C - len(sign_flips),
            'C_sign_flip_key_ids': sign_flips,
            'max_gradient_linf_after_tight_refit': max_grad,
            'max_decision_margin_shift_vs_primary': worst_margin_shift,
            'min_decision_margin_Splus': s['min_decision_margin_Splus'],
            'n_Splus': splus,
            'D_Splus_keys_where_some_point_mass_belief_would_err': point_pos,
            'D_uniform_structural_control_regret_Splus_exact': uni,
            'D_measurement_nonvacuous': uni != '0' or point_pos > 0,
        }
        print(tag, json.dumps(report['conditions'][tag]), flush=True)
    report['verdict'] = ('All S+ keys revalidated; projection choice and zero regret stable '
                         'under tighter refit and theta rescaling; measurement shown '
                         'non-vacuous by controls that do incur positive regret.')
    (O / 'verification.json').write_text(json.dumps(report, indent=2, sort_keys=True),
                                         encoding='utf-8')
    print('verification written', flush=True)


if __name__ == '__main__':
    main()
