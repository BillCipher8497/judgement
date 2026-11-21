"""Serial fresh-data residual bidder experiment; fixed protocol, atomic triplets."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import copy
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; ENGINE = HERE.parents[2]; ROOT = ENGINE.parent
sys.path[:0] = [str(HERE), str(ROUND), str(ENGINE)]
from common import save, sha, replace, stamp, log
from learning import forward, Adam
from residual import (features, design, initialize, probability, loss_gradient,
                      Ensemble, NTResidualBidder, WIDTH, MODEL_PATH)
from judgement import Rules, play_game
from judgement.engine import make_rngs, deal, run_round
from judgement.state import RoundState
from judgement.bidding.model import BidModel, ModelBidder, Agent
from judgement.policies.ta_plus import ta_plus_v2
from judgement.policies.human_guide import HumanBidder
from judgement.policies.greedy import GreedyPolicy

OUT = HERE/'results'; BASE = BidModel.load(MODEL_PATH); RULES = Rules()
STOP = datetime(2026, 11, 17, 11, 30, tzinfo=timezone.utc).timestamp() #06:30ET
FINAL_STOP = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc).timestamp()
SOURCE_PATHS = sorted({Path(__file__), HERE/'residual.py', HERE/'PROTOCOL.md', ROUND/'M5/PROTOCOL.md',
    ROUND/'learning.py', ROUND/'common.py', MODEL_PATH, ENGINE/'experiments/round11/E3/predictor.py'} |
    {Path(m.__file__).resolve() for name, m in list(sys.modules.items()) if name.startswith('judgement') and getattr(m, '__file__', None) and str(m.__file__).endswith('.py')})


def sources(): return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in SOURCE_PATHS}


def checkpoint(path, **data):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **data)
    replace(tmp, path)


def freeze():
    path = OUT/'source_freeze.json'; packet = {'sources': sources(), 'schema': ['hand', 'n', 'position', 'prior', 'legal'],
        'feature_width': WIDTH, 'numpy': np.__version__, 'canonical_model': sha(MODEL_PATH),
        'feature_names': features.names('B')[1:]+['canonical_hit_logit', 'canonical_ev_over23']}
    if path.exists(): assert json.loads(path.read_text()) == packet, 'Frozen dependency changed'
    else:
        assert not any(OUT.glob('*contexts.jsonl')); save(path, packet)
    return sha(path)


def label(base, index, digest):
    cell, g = index % 20, index//20; n, pos = [9, 4, 3, 8, 13][cell//4], cell % 4
    seed = [base, n, pos, g]; d, rngs = make_rngs(np.random.SeedSequence(seed), 4)
    hands = deal(RULES, n, d); dealer = (-pos-1) % 4; st = RoundState(RULES, n, -1, dealer, hands)
    policies = [Agent(ModelBidder(BASE), ta_plus_v2()) for _ in range(4)]
    while st.to_act != 0:
        p = st.to_act; st.apply_bid(policies[p].bid(st.info_state(p), st.legal_bids(), rngs[p]))
    ctx = features.context(st.info_state(0), st.legal_bids()); tapes = [copy.deepcopy(r.bit_generator.state) for r in rngs]
    outcomes = []
    for b in ctx['legal']:
        state = st.copy(); state.apply_bid(b); rs = [np.random.default_rng(0) for _ in range(4)]
        for r, tape in zip(rs, tapes): r.bit_generator.state = copy.deepcopy(tape)
        ps = [Agent(ModelBidder(BASE), ta_plus_v2()) for _ in range(4)]
        run_round(state, ps, rs, check=True); scores = state.scores()
        assert sum(state.tricks_won) == n and sum(state.bids) != n
        assert scores == [10+bid if bid == t else 0 for bid, t in zip(state.bids, state.tricks_won)]
        outcomes.append({'bid': b, 'hero_hit': int(state.tricks_won[0] == b), 'hero_score': scores[0],
                         'bids': state.bids[:], 'tricks': state.tricks_won[:], 'scores': scores})
    return {'base': base, 'index': index, 'seed': seed, 'hero': 0, 'dealer': dealer,
        'context': ctx, 'initial_hands': hands, 'outcomes': outcomes, 'source_freeze_sha256': digest}


def labels(stage, base, count, digest):
    path = OUT/f'{stage}_contexts.jsonl'; rows = []
    if path.exists():
        data = path.read_bytes(); lines = data.splitlines(keepends=True)
        for i, line in enumerate(lines):
            try: row = json.loads(line)
            except (ValueError, UnicodeError):
                assert i == len(lines)-1; tail = path.with_suffix('.incomplete_tail')
                assert not tail.exists(); tail.write_bytes(line); path.write_bytes(b''.join(lines[:i])); break
            assert row['base'] == base and row['index'] == i and row['source_freeze_sha256'] == digest
            rows.append(row)
    assert len(rows) <= count
    for i in range(len(rows), count):
        if time.time() >= STOP: raise TimeoutError('M4 context hard stop')
        tick = time.perf_counter(); row = label(base, i, digest); row['context_seconds'] = time.perf_counter()-tick
        with path.open('a', encoding='utf-8') as f: f.write(json.dumps(row, separators=(',', ':'))+'\n')
        rows.append(row)
        if (i+1) % 200 == 0:
            save(OUT/'progress.json', {'stage': stage, 'n_complete': i+1, 'target': count, 'utc': stamp()})
            log(HERE, f'{stage}: saved{i+1}/{count}complete contexts')
    return rows


def pack(rows):
    G = len(rows); x = np.zeros((G, 14, WIDTH)); offset = np.zeros((G, 14)); base = np.zeros((G, 14))
    mask = np.zeros((G, 14), bool); scores = np.zeros((G, 14))
    for i, row in enumerate(rows):
        ctx = row['context']; bids = ctx['legal']; features_x, off, prob = design(ctx, BASE)
        x[i, bids] = features_x; offset[i, bids] = off; base[i, bids] = prob; mask[i, bids] = True
        assert [o['bid'] for o in row['outcomes']] == bids
        scores[i, bids] = [o['hero_score'] for o in row['outcomes']]
    return x, offset, base, mask, scores


def probabilities(x, off, params):
    result = np.empty(off.shape)
    for start in range(0, len(x), 512):
        residual = forward(x[start:start+512].reshape(-1, WIDTH), params)[0].reshape(-1, 14)
        result[start:start+512] = probability(off[start:start+512]+residual)
    return result


def chosen_scores(probs, mask, scores):
    actions = np.where(mask, probs*(10+np.arange(14)), -np.inf).argmax(1)
    return scores[np.arange(len(scores)), actions]


def train_one(objective, seed, train, validation, normalization, source_digest, raw_hashes):
    folder = OUT/f'{objective}_s{seed}'; folder.mkdir(exist_ok=True); receipt = folder/'frozen.json'
    if receipt.exists():
        old = json.loads(receipt.read_text()); assert old['source_freeze_sha256'] == source_digest and old['raw_sha256'] == raw_hashes
        assert sha(folder/'selected.npz') == old['model_sha256']; return
    x, off, _, mask, scores = train; vx, voff, _, vmask, vscores = validation
    params = initialize(np.random.SeedSequence([2030000000, seed, 0])); opt = Adam(params); start_epoch = 1; best = None; best_epoch = None
    active = folder/'checkpoint.npz'
    if active.exists():
        saved = dict(np.load(active)); start_epoch = int(saved['epoch'])+1
        params = [saved[f'p{i}'] for i in range(6)]; opt = Adam(params); opt.t = int(saved['step'])
        opt.m = [saved[f'm{i}'] for i in range(6)]; opt.v = [saved[f'v{i}'] for i in range(6)]
        best = None if int(saved['best']) < 0 else int(saved['best']); best_epoch = int(saved['best_epoch'])
    for epoch in range(start_epoch, 51):
        if time.time() >= STOP: raise TimeoutError('M4 fit hard stop')
        order = np.random.default_rng(np.random.SeedSequence([2030000000, seed, 1, epoch])).permutation(len(x)); total_loss = 0.
        for start in range(0, len(order), 128):
            ids = order[start:start+128]
            value, grads = loss_gradient(x[ids], mask[ids], off[ids], scores[ids], params, objective, normalization)
            opt.step(params, grads); total_loss += value*len(ids)
        row = {'epoch': epoch, 'training_loss': total_loss/len(x), 'utc': stamp()}
        if epoch in (10, 25, 50):
            probs = probabilities(vx, voff, params); score = int(chosen_scores(probs, vmask, vscores).sum())
            row['selection_score_sum'] = score
            checkpoint(folder/f'candidate_{epoch}.npz', **{f'p{i}': p for i, p in enumerate(params)})
            if best is None or score > best:
                best, best_epoch = score, epoch
                checkpoint(folder/'selected.npz', **{f'p{i}': p for i, p in enumerate(params)})
        checkpoint(active, epoch=epoch, step=opt.t, best=-1 if best is None else best, best_epoch=-1 if best_epoch is None else best_epoch,
            **{f'p{i}': p for i, p in enumerate(params)}, **{f'm{i}': p for i, p in enumerate(opt.m)}, **{f'v{i}': p for i, p in enumerate(opt.v)})
        with (folder/'epochs.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        save(OUT/'progress.json', {'stage': 'fit', 'objective': objective, 'seed': seed, 'epoch': epoch, 'utc': stamp()})
    save(receipt, {'objective': objective, 'seed_base': 2030000000, 'seed_index': seed, 'selected_epoch': best_epoch,
        'selection_score_sum': best, 'n_selection': len(vx), 'source_freeze_sha256': source_digest,
        'raw_sha256': raw_hashes, 'model_sha256': sha(folder/'selected.npz'), 'utc': stamp(),
        'limitation': 'Epoch selection here; not held-out evidence of gain.'})
    log(HERE, f'Frozen {objective}_s{seed}; epoch{best_epoch}; selection only, no screen yet.')


def fitting(training, selection, digest):
    path = OUT/'frozen_ensembles.json'
    if path.exists():
        packet = json.loads(path.read_text()); assert packet['source_freeze_sha256'] == digest
        for rel, h in packet['model_sha256'].items(): assert sha(OUT/rel) == h
        for rel, h in packet['raw_sha256'].items(): assert sha(OUT/rel) == h
        return packet
    raw_hashes = {f'{stage}_contexts.jsonl': sha(OUT/f'{stage}_contexts.jsonl') for stage in ('train', 'selection')}
    train = pack(training); validation = pack(selection); del training, selection
    x, _, _, mask, _ = train; legal_n = mask.sum(1, keepdims=True)
    mu = (x.sum(1)/legal_n).mean(0); variance = ((x*x).sum(1)/legal_n).mean(0)-mu*mu
    sd = np.sqrt(np.maximum(variance, 0)); sd[sd < 1e-10] = 1.
    x -= mu; x /= sd; validation[0][:] = (validation[0]-mu)/sd
    normalization = float((mask*(10+np.arange(14))/(23*legal_n)).sum(1).mean())
    save(OUT/'training_normalization.json', {'mu': mu.tolist(), 'sd': sd.tolist(), 'bce_fixed_norm': normalization,
        'raw_sha256': raw_hashes, 'source_freeze_sha256': digest})
    models = {}; model_hashes = {}; vx, voff, base, vmask, vscores = validation
    for objective in ('bce', 'direct'):
        for seed in range(5): train_one(objective, seed, train, validation, normalization, digest, raw_hashes)
        probs = []
        for seed in range(5):
            rel = f'{objective}_s{seed}/selected.npz'; model_hashes[rel] = sha(OUT/rel)
            saved = dict(np.load(OUT/rel)); params = [saved[f'p{i}'] for i in range(6)]
            probs.append(probabilities(vx, voff, params))
        ensemble = np.mean(probs, axis=0); best = None; best_alpha = None; trials = []
        for alpha in (0., .25, .5, 1.):
            values = chosen_scores((1-alpha)*base+alpha*ensemble, vmask, vscores)
            score = int(values.sum()); mean = float(values.mean()); half = float(1.96*values.std(ddof=1)/np.sqrt(len(values)))
            trials.append({'alpha': alpha, 'score_sum': score, 'n_contexts': len(values), 'mean': mean, 'ci95': [mean-half, mean+half],
                'scope': 'Selection descriptive, epoch/alpha chosen here; balanced cells, normal unstratified context interval.'})
            if best is None or score > best: best, best_alpha = score, alpha
        models[objective] = {'mu': mu.tolist(), 'sd': sd.tolist(), 'alpha': best_alpha,
            'selected_models': [f'{objective}_s{s}/selected.npz' for s in range(5)], 'selection_trials': trials}
    packet = {'models': models, 'model_sha256': model_hashes, 'raw_sha256': raw_hashes,
        'source_freeze_sha256': digest, 'source_sha256': sources(), 'frozen_before_screen_utc': stamp()}
    save(path, packet); log(HERE, 'Both ensembles frozen before fullgame screen: '+json.dumps({k: v['alpha'] for k, v in models.items()}))
    return packet


def validate_game(row):
    hero = row['g'] % 4; assert row['hero'] == hero and set(row['arms']) == {'bce', 'direct', 'canonical'}
    for arm in row['arms'].values():
        assert len(arm['rounds']) == 25
        for i, r in enumerate(arm['rounds']):
            assert r['n'] == RULES.ladder()[i] and r['trump'] == RULES.trump_for_round(i)
            assert sum(r['tricks']) == r['n'] and sum(r['bids']) != r['n']
            assert r['scores'] == [10+b if b == t else 0 for b, t in zip(r['bids'], r['tricks'])]
        assert arm['totals'] == [sum(r['scores'][p] for r in arm['rounds']) for p in range(4)]
    for i in range(25):
        values = [row['arms'][a]['rounds'][i] for a in ('bce', 'direct', 'canonical')]
        if values[0]['trump'] != -1: assert values[0] == values[1] == values[2]
        else:
            leader = (RULES.dealer_for_round(i)+1) % 4; earlier = [(leader+k) % 4 for k in range((hero-leader) % 4)]
            assert all(values[0]['bids'][p] == v['bids'][p] for p in earlier for v in values)


def games(folder, stage, base, count, pool, models, frozen_digest, deadline):
    directory = folder/'results'/stage; directory.mkdir(parents=True, exist_ok=True)
    for g in range(count):
        if g % 50 == 0: assert sources() == json.loads((OUT/'source_freeze.json').read_text())['sources']
        path = directory/f'{g:04d}.json'
        if path.exists():
            row = json.loads(path.read_text()); validate_game(row)
            assert row['base'] == base and row['g'] == g and row['pool'] == pool and row['frozen_ensemble_sha256'] == frozen_digest; continue
        if time.time() >= deadline: raise TimeoutError('Fullgame phase hard stop')
        hero = g % 4; arms = {}; order = ['bce', 'direct', 'canonical']; order = order[g % 3:]+order[:g % 3]
        for arm in order:
            def opponent(): return ModelBidder(BASE) if pool == 'model' else HumanBidder() if pool == 'human' else GreedyPolicy()
            agents = [Agent(opponent(), ta_plus_v2()) for _ in range(4)]
            hero_bid = ModelBidder(BASE) if arm == 'canonical' else NTResidualBidder(models[arm], ModelBidder(BASE))
            agents[hero] = Agent(hero_bid, ta_plus_v2())
            game = play_game(RULES, agents, base+g, record_trace=False)
            arms[arm] = {'totals': game.totals, 'rounds': [{'n': r.n_cards, 'trump': r.trump, 'bids': r.bids,
                'tricks': r.tricks, 'scores': r.scores} for r in game.rounds]}
        row = {'base': base, 'g': g, 'hero': hero, 'pool': pool, 'order': order, 'arms': arms, 'frozen_ensemble_sha256': frozen_digest}
        validate_game(row); save(path, row)
        if (g+1) % 50 == 0:
            save(folder/'results/progress.json', {'stage': stage, 'complete_triplets': g+1, 'target': count, 'utc': stamp()})
            log(folder, f'{stage}: saved{g+1}/{count}complete triplets')


def main():
    OUT.mkdir(exist_ok=True); assert time.time() < STOP
    timer = threading.Timer(STOP-time.time(), lambda: os._exit(124)); timer.daemon = True; timer.start()
    try:
        digest = freeze(); pilot = labels('pilot', 1940000000, 200, digest)
        seconds = sum(r['context_seconds'] for r in pilot); remaining = STOP-time.time()
        timing = {'n_contexts': 200, 'seconds': seconds, 'projected_context_seconds': seconds*52000/200,
                  'reserve_seconds': 7200, 'remaining_seconds': remaining, 'performance_inspected': False}
        timing['accepted'] = seconds <= 120 and seconds*52000/200+7200 <= remaining
        if not (OUT/'pilot_resource.json').exists(): save(OUT/'pilot_resource.json', timing)
        else: timing = json.loads((OUT/'pilot_resource.json').read_text())
        if not timing['accepted']:
            save(OUT/'scope_failure.json', {'reason': 'Predeclared resource-only gate', 'timing': timing}); return
        training = labels('train', 1900000000, 40000, digest); selection = labels('selection', 1910000000, 12000, digest)
        packet = fitting(training, selection, digest); del training, selection, pilot
        models = {k: Ensemble(OUT, p, BASE) for k, p in packet['models'].items()}; frozen_digest = sha(OUT/'frozen_ensembles.json')
        games(HERE, 'screen', 1920000000, 2000, 'model', models, frozen_digest, STOP)
        games(HERE, 'confirmation', 1930000000, 2000, 'model', models, frozen_digest, STOP)
        log(HERE, 'M4 all fixed targets complete; independent audits/CIs required before validation.')
    finally: timer.cancel()
    assert time.time() < FINAL_STOP
    timer = threading.Timer(FINAL_STOP-time.time(), lambda: os._exit(124)); timer.daemon = True; timer.start()
    try:
        for stage, base, pool in (('human_screen', 1950000000, 'human'), ('human_confirmation', 1970000000, 'human'),
                                  ('greedy_screen', 1960000000, 'greedy'), ('greedy_confirmation', 1980000000, 'greedy')):
            games(ROUND/'M5', stage, base, 1000, pool, models, frozen_digest, FINAL_STOP)
        log(ROUND/'M5', 'M5 all fixed shift targets complete; no fitting/adaptation; independent audit remains.')
    finally: timer.cancel()


if __name__ == '__main__': main()
