"""Predeclared fresh-seed learning curves; no model/feature tuning from M3."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import hashlib
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.stats import t
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROUND))
from common import save, sha, replace, stamp, log
from learning import initialize, Adam, loss_gradient, decisions
from root_features import encode
OUT = HERE/'results'; TAG = 'S3R4n3_t-1'
STOP = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc).timestamp()


def checkpoint(path, **data):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **data)
    replace(tmp, path)


def interval(values):
    x = np.array(values); mean = float(x.mean()); half = float(t.ppf(.975, 4)*x.std(ddof=1)/np.sqrt(5))
    return {'mean': mean, 'half95': half, 'ci95': [mean-half, mean+half], 'n_training_seeds': 5}


def main():
    assert time.time() < STOP; OUT.mkdir(exist_ok=True)
    folder = ROUND/'M1/results'/TAG; m = json.loads((folder/'manifest.json').read_text())
    assert sha(folder/'root_data.npz') == m['data_sha256']
    d = dict(np.load(folder/'root_data.npz')); obs = [json.loads(s)['observation'] for s in (folder/'private_contexts.jsonl').read_text().splitlines()]
    x = np.array([encode(o) for o in obs]); available = np.flatnonzero(d['split'] == 0); va = np.flatnonzero(d['split'] == 1)
    def ordering(i): return hashlib.sha256(json.dumps([2020000000, TAG, obs[i]], sort_keys=True, separators=(',', ':')).encode('ascii')).digest()
    ordered = sorted(available.tolist(), key=ordering)
    source = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in (Path(__file__), HERE/'PROTOCOL.md', ROUND/'common.py', ROUND/'learning.py', ROUND/'root_features.py')}
    freeze = {'source_sha256': source, 'data_sha256': m['data_sha256'], 'group_order': ordered}
    if (OUT/'freeze.json').exists(): assert json.loads((OUT/'freeze.json').read_text()) == freeze
    else: save(OUT/'freeze.json', freeze)
    timer = threading.Timer(STOP-time.time(), lambda: os._exit(124)); timer.daemon = True; timer.start()
    try:
        for fi, fraction in enumerate((.25, .5, 1.)):
            tr = np.array(ordered[:int(fraction*len(ordered))]); counts = d['count'][tr, None]
            targets = {'hard_ce': d['hard_counts'][tr]/counts, 'set_bce': d['optimal_counts'][tr]/counts}
            weights = counts[:, 0]*d['payoff'][tr]/13; weights /= weights.mean()
            for si in range(5):
                for objective in ('hard_ce', 'set_bce'):
                    name = f'f{int(fraction*100)}_s{si}_{objective}'; target = OUT/name; target.mkdir(exist_ok=True)
                    if (target/'frozen.json').exists():
                        old = json.loads((target/'frozen.json').read_text()); assert old['source_sha256'] == source and old['model_sha256'] == sha(target/'selected.npz'); continue
                    p = initialize([241, 64, 32, 3], np.random.SeedSequence([2020000000, fi, si, 0])); opt = Adam(p)
                    first = 1; best = None; best_epoch = None; active = target/'checkpoint.npz'
                    if active.exists():
                        a = dict(np.load(active)); first = int(a['epoch'])+1; p = [a[f'p{i}'] for i in range(6)]; opt = Adam(p)
                        opt.m = [a[f'm{i}'] for i in range(6)]; opt.v = [a[f'v{i}'] for i in range(6)]; opt.t = int(a['step'])
                        best = None if int(a['best']) < 0 else int(a['best']); best_epoch = int(a['best_epoch'])
                    for epoch in range(first, 101):
                        if time.time() >= STOP: raise TimeoutError('M7 hard stop')
                        order = np.random.default_rng(np.random.SeedSequence([2020000000, fi, si, 1, epoch])).permutation(len(tr))
                        for start in range(0, len(tr), 256):
                            ii = order[start:start+256]; ids = tr[ii]
                            _, grad = loss_gradient(x[ids], d['mask'][ids], targets[objective][ii], weights[ii], p, objective)
                            opt.step(p, grad)
                        trace = {'epoch': epoch, 'utc': stamp()}
                        if epoch in (10, 25, 50, 100):
                            actions = decisions(x[va], d['mask'][va], p, objective); score = int(d['q_sum'][va, actions].sum())
                            trace['validation_score_sum'] = score
                            checkpoint(target/f'candidate_{epoch}.npz', **{f'p{i}': v for i, v in enumerate(p)})
                            if best is None or score < best:
                                best, best_epoch = score, epoch; checkpoint(target/'selected.npz', **{f'p{i}': v for i, v in enumerate(p)})
                        checkpoint(active, epoch=epoch, step=opt.t, best=-1 if best is None else best, best_epoch=-1 if best_epoch is None else best_epoch,
                            **{f'p{i}': v for i, v in enumerate(p)}, **{f'm{i}': v for i, v in enumerate(opt.m)}, **{f'v{i}': v for i, v in enumerate(opt.v)})
                        with (target/'epochs.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(trace)+'\n')
                        save(OUT/'progress.json', {'fit': name, 'epoch': epoch, 'utc': stamp()})
                    save(target/'frozen.json', {'name': name, 'fraction': fraction, 'seed_index': si, 'seed_base': 2020000000,
                        'objective': objective, 'n_training_groups': len(tr), 'n_training_worlds': int(d['count'][tr].sum()),
                        'selected_epoch': best_epoch, 'validation_score_sum': best, 'source_sha256': source, 'data_sha256': m['data_sha256'],
                        'model_sha256': sha(target/'selected.npz'), 'frozen_utc': stamp()})
                    log(HERE, f'Frozen{name}, epoch{best_epoch}; held-out evaluation after all30fits freeze.')
        receipts = sorted(OUT.glob('*/frozen.json')); assert len(receipts) == 30
        save(OUT/'training_complete.json', {'n_models': 30, 'receipts_sha256': {p.parent.name: sha(p) for p in receipts}, 'utc': stamp()})
        results = []; summary = {}
        for path in receipts:
            r = json.loads(path.read_text()); a = dict(np.load(path.parent/'selected.npz')); p = [a[f'p{i}'] for i in range(6)]
            actions = decisions(x, d['mask'], p, r['objective'])
            checkpoint(path.parent/'evaluation_actions.npz', actions=actions.astype(np.uint8))
            for split, label in ((2, 'discovery'), (3, 'confirmation')):
                ids = np.flatnonzero(d['split'] == split); n = int(d['count'][ids].sum())
                score = int(d['q_sum'][ids, actions[ids]].sum()); j1 = int(np.where(d['mask'][ids], d['q_sum'][ids], 2**60).min(1).sum())
                results.append({'fraction': r['fraction'], 'seed_index': r['seed_index'], 'objective': r['objective'], 'split': label,
                    'n_root_worlds': n, 'n_private_groups': len(ids), 'score_sum': score, 'J1_sum': j1, 'avoidable_cost_exact': (score-j1)/n})
        for fraction in (.25, .5, 1.):
            for label in ('discovery', 'confirmation'):
                def values(o): return [next(r['avoidable_cost_exact'] for r in results if r['fraction'] == fraction and r['split'] == label and r['objective'] == o and r['seed_index'] == s) for s in range(5)]
                hard, bce = values('hard_ce'), values('set_bce')
                summary[f'{fraction}/{label}'] = {'hard_ce': interval(hard), 'set_bce': interval(bce), 'paired_set_minus_hard': interval(np.array(bce)-hard)}
        save(OUT/'summary.json', {'rows': results, 'summary': summary, 'scope': 'Predeclared descriptive learning curve/fresh fit seeds, same held-out populations; not a replacement primary.', 'utc': stamp()})
        log(HERE, 'All30fixed learning-curve fits/evaluations complete; independent review pending.')
    finally: timer.cancel()


if __name__ == '__main__': main()
