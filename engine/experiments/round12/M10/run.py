"""Fresh matched residual ablations; reuses frozen labels, never old writers."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; ENGINE = HERE.parents[2]; ROOT = ENGINE.parent
sys.path[:0] = [str(ROUND/'M4'), str(ROUND), str(ENGINE)]
import pipeline as original
from residual import (WIDTH, design, initialize, probability, loss_gradient, NTResidualBidder)
from learning import forward, Adam
from common import save, sha, replace, stamp, log
from judgement import Rules, play_game
from judgement.bidding.model import ModelBidder, Agent
from judgement.policies.ta_plus import ta_plus_v2

OUT = HERE/'results'; OLD = ROUND/'M4/results'; BASE = original.BASE; RULES = Rules()
STOP = float(os.environ.get('JUDGEMENT_M10_DEADLINE',
    datetime(2026, 11, 17, 8, 30, tzinfo=timezone.utc).timestamp()))
VARIANTS = {'L': {'architecture': 'linear', 'offset': True}, 'R': {'architecture': 'mlp', 'offset': True}, 'N': {'architecture': 'mlp', 'offset': False}}
SOURCES = sorted(set(original.SOURCE_PATHS) | {Path(__file__), HERE/'PROTOCOL.md', HERE/'RESUMPTION_2026-11-17.md'})


def sources(): return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in SOURCES}


def checkpoint(path, **data):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f: np.savez_compressed(f, **data)
    replace(temp, path)


def parameters(path):
    a = dict(np.load(path)); return [a[f'p{i}'] for i in range(len(a))]


def fit_one(variant, vi, si, train, validation, normalization, freeze):
    folder = OUT/f'{variant}_s{si}'; folder.mkdir(exist_ok=True); receipt = folder/'frozen.json'
    if receipt.exists():
        r = json.loads(receipt.read_text()); assert r['freeze_sha256'] == freeze and r['model_sha256'] == sha(folder/'selected.npz'); return
    spec = VARIANTS[variant]; x, original_offset, _, mask, score = train; vx, original_voff, _, vmask, vscore = validation
    off = original_offset if spec['offset'] else np.zeros_like(original_offset)
    voff = original_voff if spec['offset'] else np.zeros_like(original_voff)
    seed = np.random.SeedSequence([2060000000, si, 0])
    p = [np.zeros((WIDTH, 1)), np.zeros(1)] if spec['architecture'] == 'linear' else initialize(seed)
    opt = Adam(p); first = 1; best = None; best_epoch = None; active = folder/'checkpoint.npz'
    if active.exists():
        a = dict(np.load(active)); first = int(a['epoch'])+1; p = [a[f'p{i}'] for i in range(len(p))]; opt = Adam(p); opt.t = int(a['step'])
        opt.m = [a[f'm{i}'] for i in range(len(p))]; opt.v = [a[f'v{i}'] for i in range(len(p))]
        best = None if int(a['best']) < 0 else int(a['best']); best_epoch = int(a['best_epoch'])
    for epoch in range(first, 51):
        if time.time() >= STOP: raise TimeoutError('M10 fit hard stop')
        order = np.random.default_rng(np.random.SeedSequence([2060000000, si, 1, epoch])).permutation(len(x)); loss_sum = 0.
        for start in range(0, len(order), 128):
            ids = order[start:start+128]; loss, grad = loss_gradient(x[ids], mask[ids], off[ids], score[ids], p, 'bce', normalization)
            opt.step(p, grad); loss_sum += loss*len(ids)
        row = {'epoch': epoch, 'training_loss': loss_sum/len(x), 'utc': stamp()}
        if epoch in (10, 25, 50):
            probabilities = original.probabilities(vx, voff, p); total = int(original.chosen_scores(probabilities, vmask, vscore).sum())
            row['selection_score_sum'] = total; checkpoint(folder/f'candidate_{epoch}.npz', **{f'p{i}': v for i, v in enumerate(p)})
            if best is None or total > best:
                best, best_epoch = total, epoch; checkpoint(folder/'selected.npz', **{f'p{i}': v for i, v in enumerate(p)})
        checkpoint(active, epoch=epoch, step=opt.t, best=-1 if best is None else best, best_epoch=-1 if best_epoch is None else best_epoch,
            **{f'p{i}': v for i, v in enumerate(p)}, **{f'm{i}': v for i, v in enumerate(opt.m)}, **{f'v{i}': v for i, v in enumerate(opt.v)})
        with (folder/'epochs.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        save(OUT/'progress.json', {'stage': 'fit', 'variant': variant, 'seed_index': si, 'epoch': epoch, 'utc': stamp()})
    save(receipt, {'variant': variant, 'specification': spec, 'optimizer_seed_base': 2060000000, 'seed_index': si,
        'selected_epoch': best_epoch, 'selection_score_sum': best, 'n_selection_contexts': len(vx),
        'model_sha256': sha(folder/'selected.npz'), 'freeze_sha256': freeze, 'frozen_utc': stamp()})
    log(HERE, f'Frozen{variant}_s{si}; selectionepoch{best_epoch}; held-out ablationgames untouched.')


def fit_models(freeze):
    path = OUT/'frozen_ensembles.json'
    if path.exists():
        r = json.loads(path.read_text()); assert r['freeze_sha256'] == freeze
        for rel, h in r['model_sha256'].items(): assert sha(OUT/rel) == h
        return r
    training = [json.loads(s) for s in (OLD/'train_contexts.jsonl').read_text().splitlines()]
    selection = [json.loads(s) for s in (OLD/'selection_contexts.jsonl').read_text().splitlines()]
    assert len(training) == 40000 and len(selection) == 12000
    train = original.pack(training); validation = original.pack(selection); del training, selection
    norm = json.loads((OLD/'training_normalization.json').read_text()); mu, sd = np.array(norm['mu']), np.array(norm['sd'])
    train[0][:] = (train[0]-mu)/sd; validation[0][:] = (validation[0]-mu)/sd
    models = {}; hashes = {}; vx, voff, baseline, mask, scores = validation
    for vi, variant in enumerate(VARIANTS):
        for si in range(5): fit_one(variant, vi, si, train, validation, norm['bce_fixed_norm'], freeze)
        probabilities = []
        for si in range(5):
            rel = f'{variant}_s{si}/selected.npz'; hashes[rel] = sha(OUT/rel)
            probs = original.probabilities(vx, voff if VARIANTS[variant]['offset'] else np.zeros_like(voff), parameters(OUT/rel))
            probabilities.append(probs)
        ensemble = np.mean(probabilities, axis=0); choices = []; trials = []
        for alpha in (0., .25, .5, 1.):
            values = original.chosen_scores((1-alpha)*baseline+alpha*ensemble, mask, scores)
            total = int(values.sum()); mean = float(values.mean()); half = float(1.96*values.std(ddof=1)/np.sqrt(len(values)))
            choices.append((-total, alpha)); trials.append({'alpha': alpha, 'score_sum': total, 'mean': mean,
                'n_contexts': len(values), 'ci95': [mean-half, mean+half], 'limitation': 'Reused selection, optimistic; not independent effect evidence.'})
        best = min(choices)
        models[variant] = {'specification': VARIANTS[variant], 'mu': mu.tolist(), 'sd': sd.tolist(), 'alpha': best[1],
            'selected_models': [f'{variant}_s{s}/selected.npz' for s in range(5)], 'selection_trials': trials}
    r = {'models': models, 'model_sha256': hashes, 'freeze_sha256': freeze, 'source_sha256': sources(), 'frozen_before_screen_utc': stamp()}
    save(path, r); return r


class Ensemble:
    def __init__(self, packet):
        self.alpha = packet['alpha']; self.mu = np.array(packet['mu']); self.sd = np.array(packet['sd']); self.offset = packet['specification']['offset']
        self.models = [parameters(OUT/name) for name in packet['selected_models']]

    def choose(self, ctx):
        x, offset, baseline = design(ctx, BASE); z = (x-self.mu)/self.sd
        if not self.offset: offset = np.zeros_like(offset)
        predictions = np.mean([probability(offset+forward(z, p)[0][:, 0]) for p in self.models], axis=0)
        p = (1-self.alpha)*baseline+self.alpha*predictions
        return ctx['legal'][int(np.argmax(p*(10+np.array(ctx['legal']))))]


def validate(row):
    assert row['hero'] == row['g'] % 4 and set(row['arms']) == {'L', 'R', 'N', 'canonical'}
    for arm in row['arms'].values():
        assert len(arm['rounds']) == 25
        for i, r in enumerate(arm['rounds']):
            assert r['n'] == RULES.ladder()[i] and r['trump'] == RULES.trump_for_round(i)
            assert sum(r['tricks']) == r['n'] and sum(r['bids']) != r['n']
            assert r['scores'] == [10+b if b == t else 0 for b, t in zip(r['bids'], r['tricks'])]
        assert arm['totals'] == [sum(r['scores'][p] for r in arm['rounds']) for p in range(4)]
    for i in range(25):
        rounds = [row['arms'][a]['rounds'][i] for a in ('L', 'R', 'N', 'canonical')]
        if rounds[0]['trump'] != -1: assert all(r == rounds[0] for r in rounds)
        else:
            start = (RULES.dealer_for_round(i)+1) % 4
            for p in [(start+k) % 4 for k in range((row['hero']-start) % 4)]: assert len({r['bids'][p] for r in rounds}) == 1


def games(stage, base, models, digest, frozen_sources):
    folder = OUT/stage; folder.mkdir(exist_ok=True)
    for g in range(2000):
        if g % 50 == 0: assert sources() == frozen_sources
        path = folder/f'{g:04d}.json'
        if path.exists():
            row = json.loads(path.read_text()); validate(row); assert row['g'] == g and row['base'] == base and row['models_freeze_sha256'] == digest; continue
        if time.time() >= STOP: raise TimeoutError('M10 game hard stop')
        hero = g % 4; order = ['L', 'R', 'N', 'canonical']; order = order[g % 4:]+order[:g % 4]; arms = {}
        for arm in order:
            agents = [Agent(ModelBidder(BASE), ta_plus_v2()) for _ in range(4)]
            if arm != 'canonical': agents[hero] = Agent(NTResidualBidder(models[arm], ModelBidder(BASE)), ta_plus_v2())
            result = play_game(RULES, agents, base+g, record_trace=False)
            arms[arm] = {'totals': result.totals, 'rounds': [{'n': r.n_cards, 'trump': r.trump, 'bids': r.bids,
                'tricks': r.tricks, 'scores': r.scores} for r in result.rounds]}
        row = {'g': g, 'hero': hero, 'base': base, 'order': order, 'arms': arms, 'models_freeze_sha256': digest}
        validate(row); save(path, row)
        if (g+1) % 50 == 0:
            save(OUT/'progress.json', {'stage': stage, 'complete_quadruplets': g+1, 'target': 2000, 'utc': stamp()})
            log(HERE, f'{stage}: saved{g+1}/2000complete quadruplets')


def main():
    assert time.time() < STOP; OUT.mkdir(exist_ok=True)
    original_freeze = json.loads((OLD/'source_freeze.json').read_text())
    for rel, h in original_freeze['sources'].items(): assert sha(ROOT/rel) == h
    old_ensemble = json.loads((OLD/'frozen_ensembles.json').read_text())
    for rel, h in old_ensemble['raw_sha256'].items(): assert sha(OLD/rel) == h
    packet = {'source_sha256': sources(), 'originalM4_source_freeze_sha256': sha(OLD/'source_freeze.json'),
        'training_raw_sha256': old_ensemble['raw_sha256'], 'normalization_sha256': sha(OLD/'training_normalization.json'),
        'variants': VARIANTS, 'seed_base_optimizer': 2060000000}
    path = OUT/'source_freeze.json'
    if path.exists(): assert json.loads(path.read_text()) == packet
    else: save(path, packet)
    timer = threading.Timer(STOP-time.time(), lambda: os._exit(124)); timer.daemon = True; timer.start()
    try:
        ensemble_packet = fit_models(sha(path)); models = {k: Ensemble(v) for k, v in ensemble_packet['models'].items()}
        digest = sha(OUT/'frozen_ensembles.json')
        games('screen', 2040000000, models, digest, packet['source_sha256'])
        games('confirmation', 2050000000, models, digest, packet['source_sha256'])
        log(HERE, 'All15fits/4000freshquadruplets complete; separateanalysis/review/fullsuite pending.')
    finally: timer.cancel()


if __name__ == '__main__': main()
