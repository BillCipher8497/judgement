"""N19 - do the card-holder marginals already pin the one-step decision?

For every decision-relevant context, bound q.d over the whole polytope of
beliefs matching the true posterior's card-holder marginals, with no family
restriction, and certify the sign exactly in rational arithmetic.

Exact and deterministic: no RNG, no seeds, no fits, no games. See PROTOCOL.md.
"""
import os, sys, json, time, hashlib
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog

H = Path(__file__).resolve().parent
O = H / 'results'
R12 = H.parent.parent / 'round12'
R13 = H.parent
sys.path.insert(0, str(R12 / 'N9'))
import n9lib as B

TAGS = B.TAGS


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


def constraints(r):
    """Exact moment constraints A q = b for the card-holder marginals of the
    uniform reaching posterior, plus normalization. Returns A (list of rows of
    0/1 ints) and b (list of Fractions)."""
    W, U = r.F.shape
    n = r.count
    rows, rhs = [[1] * W], [F(1)]
    for c in range(U):
        for o in range(3):
            col = [1 if r.F[w, c] == o else 0 for w in range(W)]
            cnt = sum(1 for w in r.posterior if r.F[w, c] == o)
            if cnt == 0 and not any(col[w] for w in r.posterior):
                # zero-mass moment: forces q to vanish on those worlds
                rows.append(col); rhs.append(F(0))
            else:
                rows.append(col); rhs.append(F(cnt, n))
    return rows, rhs


def exact_vertex(rows, rhs, W, support, d):
    """Reconstruct an exact rational point from the LP's positive support by
    solving the active system restricted to that support. Returns (q, value) or
    (None, None) if it cannot be certified."""
    idx = sorted(support)
    if not idx:
        return None, None
    # exact Gaussian elimination on the restricted system
    A = [[F(r[j]) for j in idx] + [rhs[i]] for i, r in enumerate(rows)]
    m, k = len(A), len(idx)
    piv = []
    row = 0
    for col in range(k):
        sel = next((i for i in range(row, m) if A[i][col] != 0), None)
        if sel is None:
            continue
        A[row], A[sel] = A[sel], A[row]
        pv = A[row][col]
        A[row] = [x / pv for x in A[row]]
        for i in range(m):
            if i != row and A[i][col] != 0:
                f = A[i][col]
                A[i] = [a - f * b for a, b in zip(A[i], A[row])]
        piv.append(col)
        row += 1
        if row == m:
            break
    # inconsistent?
    for i in range(row, m):
        if all(x == 0 for x in A[i][:k]) and A[i][k] != 0:
            return None, None
    if len(piv) != k:
        return None, None          # not a unique vertex on this support
    vals = [F(0)] * k
    for i, col in enumerate(piv):
        vals[col] = A[i][k]
    q = [F(0)] * W
    for j, v in zip(idx, vals):
        q[j] = v
    if any(v < 0 for v in q):
        return None, None
    # exact feasibility check against the full system
    for r, bb in zip(rows, rhs):
        if sum((F(r[w]) * q[w] for w in range(W) if q[w]), F(0)) != bb:
            return None, None
    return q, sum((q[w] * F(int(d[w])) for w in range(W) if q[w]), F(0))


def main():
    O.mkdir(parents=True, exist_ok=True)
    sources = {str(p): sha(p) for p in
               [H / 'run19.py', H / 'PROTOCOL.md', R12 / 'N9/n9lib.py'] +
               [R12 / 'N9/results' / t / 'data.npz' for t in TAGS] +
               [R13 / 'N17/results' / t / 'keys.json' for t in TAGS]}
    p = O / 'sources.json'
    if p.exists():
        assert json.loads(p.read_text()) == sources, 'source freeze changed'
    else:
        save(p, sources)

    summaries = {}
    for tag in TAGS:
        check()
        out = O / tag / 'contexts.json'
        if out.exists():
            summaries[tag] = json.loads((O / tag / 'summary.json').read_text())
            continue
        n17 = {a['key_id']: a for a in
               json.loads((R13 / 'N17/results' / tag / 'keys.json').read_text())}
        rows_by_kid = {r.kid: r for r in B.rows(B.load(R12 / 'N9/results' / tag / 'data.npz'))}
        recs = []
        for kid, a in sorted(n17.items()):
            if a['stratum'] != 'S+':
                continue
            check()
            z = np.load(R12 / 'N12/results' / tag / f'{kid:07d}.npz')
            Q = z['Q'].astype(np.int64)
            d = (Q[:, 1] - Q[:, 0])
            r = rows_by_kid[kid]
            W = len(r.F)
            V = [F(a['V_exact'][0]), F(a['V_exact'][1])]
            true_sign = 1 if V[1] > V[0] else -1
            A, b = constraints(r)
            Af = np.array(A, dtype=float)
            bf = np.array([float(x) for x in b])
            rec = {'key_id': kid, 'reach': a['reach'], 'n_structural': W,
                   'true_better_action_index': 1 if true_sign > 0 else 0,
                   'spread_exact': a['spread_exact']}
            ends = {}
            for name, sense in (('min', 1.0), ('max', -1.0)):
                lp = linprog(sense * d.astype(float), A_eq=Af, b_eq=bf,
                             bounds=(0, None), method='highs',
                             options={'threads': 1})
                if not lp.success:
                    ends[name] = {'lp_success': False}
                    continue
                sup = {j for j, v in enumerate(lp.x) if v > 1e-12}
                q, val = exact_vertex(A, b, W, sup, d)
                ends[name] = {'lp_success': True,
                              'lp_value': float(sense * lp.fun),
                              'exact_certified': q is not None,
                              'exact_value': str(val) if q is not None else None}
                if q is not None:
                    ends[name]['support'] = {str(j): str(q[j]) for j in range(W) if q[j]}
            rec['endpoints'] = ends
            lo = ends['min'].get('exact_value')
            hi = ends['max'].get('exact_value')
            if lo is not None and hi is not None:
                lo, hi = F(lo), F(hi)
                identified = (lo > 0 and hi > 0) if true_sign > 0 else (lo < 0 and hi < 0)
                rec['certified'] = True
                rec['identified_by_marginals'] = bool(identified)
                # worst-case regret over the polytope: the adversarial endpoint
                adverse = hi if true_sign < 0 else lo
                flips = (adverse > 0) if true_sign < 0 else (adverse < 0)
                rec['worst_case_regret_exact'] = str(abs(V[1] - V[0])) if flips else '0'
            else:
                rec['certified'] = False
                rec['identified_by_marginals'] = None
                rec['worst_case_regret_exact'] = None
            recs.append(rec)
        save(out, recs)
        reach = sum(a['reach'] for a in recs)
        ident = [a for a in recs if a['identified_by_marginals'] is True]
        notid = [a for a in recs if a['identified_by_marginals'] is False]
        unc = [a for a in recs if not a['certified']]
        wc = sum((F(a['worst_case_regret_exact']) * a['reach']
                  for a in recs if a['worst_case_regret_exact'] is not None), F(0))
        summary = {
            'tag': tag, 'n_Splus': len(recs), 'reach_Splus': reach,
            'n_certified': len(recs) - len(unc),
            'n_identified_by_marginals': len(ident),
            'n_not_identified': len(notid),
            'not_identified_key_ids': [a['key_id'] for a in notid],
            'n_uncertified': len(unc), 'uncertified_key_ids': [a['key_id'] for a in unc],
            'worst_case_polytope_regret_exact': str(wc / reach) if reach else None,
            'worst_case_polytope_regret_float': float(wc / reach) if reach else None,
            'sampling_CI': None,
            'scope': 'Exact enumeration + exact rational vertex certificates over the complete S+ set; no sampling interval.',
        }
        save(O / tag / 'summary.json', summary)
        summaries[tag] = summary
        print(tag, 'S+', summary['n_Splus'], 'identified', summary['n_identified_by_marginals'],
              'not identified', summary['n_not_identified'], 'uncertified', summary['n_uncertified'],
              'worst-case polytope regret', summary['worst_case_polytope_regret_exact'], flush=True)

    assert all(sha(Path(k)) == v for k, v in sources.items()), 'source changed during run'
    save(O / 'complete.json', {'utc': stamp(), 'summaries': summaries,
                               'sources_sha256': sha(O / 'sources.json'),
                               'scope': 'Exact deterministic LP bounds with exact rational vertex certificates.'})
    print('N19 complete', flush=True)


if __name__ == '__main__':
    main()
