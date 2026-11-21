"""N17 - decision-relevant misspecification.

Exact, deterministic. No RNG, no seeds, no fits, no games. Reads the frozen
round12 N9 supports and N12 action tables and writes only under N17/results.
See PROTOCOL.md, predeclared before any primary number was computed.
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
MARGIN = 1e-6
GRAD_TOL = 1e-8
BANDS = (('small', F(0), F(5)), ('medium', F(5), F(10)), ('large', F(10), None))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stamp():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check():
    cut = os.environ.get('JUDGEMENT_JOB_DEADLINE')
    if cut and time.time() >= float(cut):
        raise SystemExit('Absolute job cutoff reached; partial results preserved')


def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')


def projection(r):
    """I-projection of the uniform reaching posterior onto the conditional-product
    family, via the exponential-family dual on the exact structural support."""
    U = r.F.shape[1]
    feat = np.array([[(r.F[w, c] == k) for c in range(U) for k in range(3)]
                     for w in range(len(r.F))], dtype=float)
    mu = feat[r.posterior].mean(axis=0)

    def objective(theta):
        s = feat @ theta
        lz = logsumexp(s)
        p = np.exp(s - lz)
        return float(lz - theta @ mu - math.log(r.count)), feat.T @ p - mu

    opt = minimize(objective, np.zeros(feat.shape[1]), jac=True, method='L-BFGS-B',
                   options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-16})
    value, grad = objective(opt.x)
    s = feat @ opt.x
    q = np.exp(s - logsumexp(s))
    return q, float(value), float(np.max(np.abs(grad))), bool(opt.success)


def band(s):
    for name, lo, hi in BANDS:
        if s > lo and (hi is None or s <= hi):
            return name
    return None


def main():
    O.mkdir(parents=True, exist_ok=True)
    sources = {str(p): sha(p) for p in
               [H / 'run17.py', H / 'PROTOCOL.md',
                R12 / 'N9/n9lib.py', R12 / 'N12/results/evaluation.json'] +
               [R12 / 'N9/results' / t / 'data.npz' for t in TAGS]}
    p = O / 'sources.json'
    if p.exists():
        assert json.loads(p.read_text()) == sources, 'source freeze changed'
    else:
        save(p, sources)

    n12eval = json.loads((R12 / 'N12/results/evaluation.json').read_text())
    recipes = n12eval['rows']

    summaries = {}
    for tag in TAGS:
        folder = O / tag
        folder.mkdir(exist_ok=True)
        out = folder / 'keys.json'
        if out.exists():
            summaries[tag] = json.loads((folder / 'summary.json').read_text())
            continue
        rows = {r.kid: r for r in B.rows(B.load(R12 / 'N9/results' / tag / 'data.npz'))}
        keydir = R12 / 'N12/results' / tag
        kids = sorted(int(f.stem) for f in keydir.glob('*.npz'))
        records = []
        for i, kid in enumerate(kids):
            check()
            z = np.load(keydir / f'{kid:07d}.npz')
            Q = z['Q'].astype(np.int64)
            actions = z['actions'].tolist()
            post = z['posterior']
            r = rows[kid]
            # structural index alignment between the two frozen packets
            assert len(r.F) == Q.shape[0], (kid, len(r.F), Q.shape)
            assert r.count == len(post) and np.array_equal(np.sort(r.posterior), np.sort(post))

            n = len(post)
            V = [F(int(Q[post, a].sum()), n) for a in range(Q.shape[1])]
            spread = abs(V[1] - V[0])
            best = max(V)

            q, kl, gl, ok = projection(r)
            delta = float(q @ (Q[:, 1] - Q[:, 0]))
            margin = abs(delta)
            choice = 1 if delta > 0 else 0          # lower card on tie/negative
            amb = margin < MARGIN
            reg = best - V[choice]
            reg_other = best - V[1 - choice]
            # uniform-over-structural-support control
            du = float((Q[:, 1] - Q[:, 0]).mean())
            ureg = best - V[1 if du > 0 else 0]

            records.append({
                'key_id': kid, 'reach': n, 'n_structural': int(len(r.F)),
                'actions': actions,
                'V_exact': [str(v) for v in V], 'spread_exact': str(spread),
                'stratum': 'S+' if spread > 0 else 'S0', 'band': band(spread),
                'projection_KL_nats': kl, 'projection_grad_linf': gl,
                'projection_converged': ok and gl < GRAD_TOL,
                'decision_margin': margin, 'numerically_ambiguous': amb,
                'projection_choice': choice,
                'projection_regret_exact': str(reg),
                'projection_regret_if_other_choice_exact': str(reg_other),
                'uniform_control_regret_exact': str(ureg),
            })
            if (i + 1) % 100 == 0:
                save(O / 'progress.json', {'tag': tag, 'complete': i + 1,
                                           'target': len(kids), 'utc': stamp()})
        save(out, records)

        sp = [a for a in records if a['stratum'] == 'S+']
        s0 = [a for a in records if a['stratum'] == 'S0']
        reach_sp = sum(a['reach'] for a in sp)
        num = sum((F(a['projection_regret_exact']) * a['reach'] for a in sp), F(0))
        hi = sum((max(F(a['projection_regret_exact']),
                      F(a['projection_regret_if_other_choice_exact'])) * a['reach']
                  for a in sp if a['numerically_ambiguous']), F(0))
        unum = sum((F(a['uniform_control_regret_exact']) * a['reach'] for a in sp), F(0))
        bands = {}
        for name, _, _ in BANDS:
            g = [a for a in sp if a['band'] == name]
            rr = sum(a['reach'] for a in g)
            bands[name] = {
                'n_keys': len(g), 'reach': rr,
                'projection_regret_exact': str(sum((F(a['projection_regret_exact']) * a['reach'] for a in g), F(0)) / rr) if rr else None,
                'n_positive_regret_keys': sum(F(a['projection_regret_exact']) > 0 for a in g)}
        summary = {
            'tag': tag, 'n_keys': len(records), 'n_S0': len(s0), 'n_Splus': len(sp),
            'reach_total': sum(a['reach'] for a in records), 'reach_Splus': reach_sp,
            'projection_regret_Splus_exact': str(num / reach_sp) if reach_sp else None,
            'projection_regret_Splus_float': float(num / reach_sp) if reach_sp else None,
            'n_positive_regret_keys_Splus': sum(F(a['projection_regret_exact']) > 0 for a in sp),
            'positive_regret_key_ids': [a['key_id'] for a in sp if F(a['projection_regret_exact']) > 0],
            'n_numerically_ambiguous_Splus': sum(a['numerically_ambiguous'] for a in sp),
            'ambiguous_worst_case_added_regret_exact': str(hi / reach_sp) if reach_sp else None,
            'uniform_control_regret_Splus_exact': str(unum / reach_sp) if reach_sp else None,
            'n_projection_nonconverged': sum(not a['projection_converged'] for a in records),
            'min_decision_margin_Splus': min((a['decision_margin'] for a in sp), default=None),
            'bands': bands,
            'sampling_CI': None,
            'scope': 'Exact enumeration over the complete frozen N12 key set; no sampling interval applies.',
        }
        save(folder / 'summary.json', summary)
        summaries[tag] = summary
        print(tag, 'S+', summary['n_Splus'], 'regret', summary['projection_regret_Splus_exact'],
              'positive keys', summary['n_positive_regret_keys_Splus'], flush=True)

    # secondary: the 240 frozen N12 recipes restricted to S+
    sec = {}
    for tag in TAGS:
        recs = {a['key_id']: a for a in json.loads((O / tag / 'keys.json').read_text())}
        splus = {k for k, a in recs.items() if a['stratum'] == 'S+'}
        reach_sp = sum(recs[k]['reach'] for k in splus)
        cells = {}
        for row in [a for a in recipes if a['tag'] == tag]:
            dec = json.loads((R12 / 'N12/results' / tag /
                              f"{row['config']}_{row['loss']}_s{row['seed']}_b{row['budget']}.json").read_text())
            num = sum(d for kid, ch, d in dec['decisions_key_choice_deficit'] if kid in splus)
            tot = sum(d for kid, ch, d in dec['decisions_key_choice_deficit'])
            cells.setdefault(f"{row['config']}/{row['loss']}/{row['budget']}", {})[row['seed']] = {
                'Splus_numerator': num, 'all_numerator': tot}
            assert num == tot, (tag, row, num, tot)  # S0 keys cannot contribute
        sec[tag] = {'reach_Splus': reach_sp, 'n_Splus': len(splus),
                    'cells': {k: {'Splus_regret_exact': str(F(sum(v['Splus_numerator'] for v in d.values()), 5 * reach_sp)),
                                  'per_seed_numerators': [d[s]['Splus_numerator'] for s in sorted(d)]}
                              for k, d in cells.items()}}
    save(O / 'n12_recipes_Splus.json', sec)

    assert all(sha(Path(k)) == v for k, v in sources.items()), 'source changed during run'
    save(O / 'complete.json', {'utc': stamp(), 'summaries': summaries,
                               'sources_sha256': sha(O / 'sources.json'),
                               'scope': 'Exact deterministic enumeration, complete frozen key set, no RNG/seeds/games.'})
    print('N17 complete', flush=True)


if __name__ == '__main__':
    main()
