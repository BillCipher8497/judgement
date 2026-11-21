"""Frozen paired optimization experiment. Held-out evaluation is a separate script."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROUND))
from common import sha, save, replace, stamp, log
from learning import initialize, loss_gradient, Adam, decisions, OBJECTIVES
from root_features import encode

STOP = datetime(2026, 11, 17, 8, 30, tzinfo=timezone.utc).timestamp() #03:30ET
OUT = HERE/'results'
CONDITIONS = ('S2R4n2_t0', 'S2R4n2_t-1', 'S3R4n3_t0', 'S3R4n3_t-1')
SOURCES = [Path(__file__), ROUND/'learning.py', ROUND/'root_features.py', ROUND/'common.py', HERE/'PROTOCOL.md']


def checkpoint(path, **data):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f: np.savez_compressed(f, **data)
    replace(temp, path)


def arrays(tag):
    directory = ROUND/'M1/results'/tag
    manifest = json.loads((directory/'manifest.json').read_text())
    assert sha(directory/'root_data.npz') == manifest['data_sha256']
    assert sha(directory/'private_contexts.jsonl') == manifest['contexts_sha256']
    data = dict(np.load(directory/'root_data.npz'))
    rows = [json.loads(line) for line in (directory/'private_contexts.jsonl').read_text().splitlines()]
    assert all(row['group'] == i for i, row in enumerate(rows))
    x = np.array([encode(r['observation']) for r in rows])
    # An exhaustive injectivity and own-hand split audit, with no target values.
    assert len(set(row.tobytes() for row in x)) == len(x)
    hand_splits = {}
    for r, split in zip(rows, data['split']):
        h = tuple(r['observation']['hand'])
        assert h not in hand_splits or hand_splits[h] == split
        hand_splits[h] = int(split)
    assert np.array_equal(x[:, -3:].astype(bool), data['mask'])
    return x, data, manifest


def train_one(tag, ci, arch, ai, seed_index, objective, x, data, manifest, frozen_sources):
    name = f'{tag}_{arch}_s{seed_index}_{objective}'; folder = OUT/name; folder.mkdir(exist_ok=True)
    receipt = folder/'frozen.json'
    if receipt.exists():
        old = json.loads(receipt.read_text()); assert old['source_sha256'] == frozen_sources
        assert old['data_sha256'] == manifest['data_sha256'] and sha(folder/'selected.npz') == old['model_sha256']; return
    tr = np.flatnonzero(data['split'] == 0); va = np.flatnonzero(data['split'] == 1)
    assert len(tr) and len(va) and not set(tr) & set(va)
    count = data['count'][tr, None]; payoff = data['payoff'][tr, None]
    targets = {'hard_ce': data['hard_counts'][tr]/count,
               'uniform_ce': data['uniform_optimal_units6'][tr]/(6*count),
               'set_bce': data['optimal_counts'][tr]/count,
               'q_reg': data['q_sum'][tr]/(count*payoff),
               'direct': data['q_sum'][tr]/(count*payoff)}
    weights = (count[:, 0]*payoff[:, 0]/13); weights /= weights.mean()
    widths = [241, 3] if arch == 'linear' else [241, 64, 32, 3]
    params = initialize(widths, np.random.SeedSequence([2010000000, ci, ai, seed_index, 0]))
    opt = Adam(params); first_epoch = 1; best_total = None; best_epoch = None
    active = folder/'checkpoint.npz'
    if active.exists():
        saved = dict(np.load(active)); first_epoch = int(saved['epoch'])+1
        params = [saved[f'p{i}'] for i in range(len(params))]; opt = Adam(params)
        opt.m = [saved[f'm{i}'] for i in range(len(params))]; opt.v = [saved[f'v{i}'] for i in range(len(params))]; opt.t = int(saved['step'])
        history = [json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()]
        candidates = [r for r in history if r['epoch'] in (10, 25, 50, 100) and r['epoch'] < first_epoch]
        if candidates:
            chosen = min(candidates, key=lambda r: (r['validation_score_sum'], r['epoch']))
            best_total, best_epoch = chosen['validation_score_sum'], chosen['epoch']
    for epoch in range(first_epoch, 101):
        if time.time() >= STOP: raise TimeoutError('M3 absolute03:30ET hard stop')
        # Independent epoch address: objectives share exact minibatches/init.
        order = np.random.default_rng(np.random.SeedSequence([2010000000, ci, ai, seed_index, 1, epoch])).permutation(len(tr))
        cumulative = 0.
        for start in range(0, len(order), 256):
            ix = order[start:start+256]; ids = tr[ix]
            value, grads = loss_gradient(x[ids], data['mask'][ids], targets[objective][ix], weights[ix], params, objective)
            opt.step(params, grads); cumulative += value*len(ix)
        row = {'epoch': epoch, 'training_loss': cumulative/len(tr), 'optimizer_steps': opt.t, 'utc': stamp()}
        if epoch in (10, 25, 50, 100):
            chosen = decisions(x[va], data['mask'][va], params, objective)
            total = int(data['q_sum'][va, chosen].sum()); row['validation_score_sum'] = total
            if best_total is None or total < best_total:
                best_total, best_epoch = total, epoch
                checkpoint(folder/'selected.npz', **{f'p{i}': p for i, p in enumerate(params)})
            checkpoint(folder/f'candidate_{epoch}.npz', **{f'p{i}': p for i, p in enumerate(params)})
        checkpoint(active, epoch=epoch, step=opt.t, **{f'p{i}': p for i, p in enumerate(params)},
                   **{f'm{i}': p for i, p in enumerate(opt.m)}, **{f'v{i}': p for i, p in enumerate(opt.v)})
        with (folder/'epochs.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        save(OUT/'progress.json', {'model': name, 'epoch': epoch, 'utc': stamp(), 'deadline_utc': '2026-11-17T08:30:00Z'})
    save(receipt, {'name': name, 'condition': tag, 'architecture': arch, 'seed_index': seed_index,
         'seed_base': 2010000000, 'objective': objective, 'selected_epoch': best_epoch,
         'validation_score_sum': best_total, 'validation_worlds': int(data['count'][va].sum()),
         'training_groups': len(tr), 'training_worlds': int(data['count'][tr].sum()),
         'model_sha256': sha(folder/'selected.npz'), 'source_sha256': frozen_sources,
         'data_sha256': manifest['data_sha256'], 'frozen_utc': stamp(),
         'held_out_evaluation': 'Not accessed by training/selection; separate evaluation only after all model freezes.'})
    log(HERE, f'Frozen {name}: selected epoch{best_epoch}; exact validation sum{best_total}; no held-out evaluation yet.')


def main():
    assert time.time() < STOP; OUT.mkdir(exist_ok=True)
    frozen_sources = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in SOURCES}
    source_path = OUT/'source_freeze.json'
    if source_path.exists(): assert json.loads(source_path.read_text()) == frozen_sources
    else: save(source_path, frozen_sources)
    timer = threading.Timer(STOP-time.time(), lambda: os._exit(124)); timer.daemon = True; timer.start()
    try:
        for ci, tag in enumerate(CONDITIONS):
            x, data, manifest = arrays(tag)
            for ai, arch in enumerate(('linear', 'mlp')):
                for si in range(5):
                    for objective in OBJECTIVES: train_one(tag, ci, arch, ai, si, objective, x, data, manifest, frozen_sources)
        receipts = list(OUT.glob('*/frozen.json')); assert len(receipts) == 200
        save(OUT/'training_complete.json', {'n_frozen_models': len(receipts), 'finished_utc': stamp(),
            'receipts_sha256': {p.parent.name: sha(p) for p in receipts}, 'source_sha256': frozen_sources})
        log(HERE, 'All200 model fits and validation selections frozen. Held-out evaluation may now begin.')
    finally: timer.cancel()


if __name__ == '__main__': main()
