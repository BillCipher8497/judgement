"""Separate full counterfactual replay, model-selection and game/CI audit."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import copy
import json
import math
import statistics
import sys
import time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; ENGINE = HERE.parents[2]; ROOT = ENGINE.parent
sys.path[:0] = [str(ROUND/'M4'), str(ROUND), str(ENGINE)]
from common import save, sha, stamp, log
from residual import design, features, MODEL_PATH, WIDTH
from judgement import Rules, play_game
from judgement.engine import make_rngs, deal, run_round
from judgement.state import RoundState
from judgement.bidding.model import BidModel, ModelBidder, Agent
from judgement.policies.ta_plus import ta_plus_v2
from judgement.policies.human_guide import HumanBidder
from judgement.policies.greedy import GreedyPolicy
MODEL = BidModel.load(MODEL_PATH); RULES = Rules(); NT = ROUND/'M4/results'


def records(stage):
    with (NT/f'{stage}_contexts.jsonl').open(encoding='utf-8') as f:
        for line in f: yield json.loads(line)


def replay_labels(digest):
    reports = {}; mu_sum = np.zeros(WIDTH); square_sum = np.zeros(WIDTH); bce_normalizer_sum = 0.
    for stage, base, count in (('pilot', 1940000000, 200), ('train', 1900000000, 40000), ('selection', 1910000000, 12000)):
        complete = 0; rounds = 0
        for i, row in enumerate(records(stage)):
            assert row['base'] == base and row['index'] == i and row['source_freeze_sha256'] == digest
            cell, replicate = i % 20, i//20; n = (9, 4, 3, 8, 13)[cell//4]; position = cell % 4
            expected_seed = [base, n, position, replicate]; assert row['seed'] == expected_seed
            dealer = (-position-1) % 4; assert row['hero'] == 0 and row['dealer'] == dealer
            rng_deal, rngs = make_rngs(np.random.SeedSequence(expected_seed), 4)
            hands = deal(RULES, n, rng_deal); assert hands == row['initial_hands']
            state = RoundState(RULES, n, -1, dealer, hands)
            agents = [Agent(ModelBidder(MODEL), ta_plus_v2()) for _ in range(4)]
            while state.to_act != 0:
                player = state.to_act; value = agents[player].bid(state.info_state(player), state.legal_bids(), rngs[player]); state.apply_bid(value)
            context = {'hand': int(hands[0]), 'n': n, 'position': position,
                       'prior': [int(b) for _, b in state.bid_seq], 'legal': sorted(state.legal_bids())}
            assert context == row['context']
            x, offset, baseline = design(context, MODEL)
            original_evs, original_p = ModelBidder(MODEL).evs(state.info_state(0))
            np.testing.assert_array_equal(baseline, np.diag(original_p)[context['legal']])
            np.testing.assert_allclose(baseline*(10+np.array(context['legal'])), original_evs[context['legal']], rtol=0, atol=1e-14)
            if stage == 'train':
                mu_sum += x.mean(0); square_sum += (x*x).mean(0)
                bce_normalizer_sum += float(np.mean((10+np.array(context['legal']))/23))
            assert [o['bid'] for o in row['outcomes']] == context['legal']
            tapes = [copy.deepcopy(r.bit_generator.state) for r in rngs]
            for outcome in row['outcomes']:
                s = state.copy(); s.apply_bid(outcome['bid']); rs = [np.random.default_rng(0) for _ in range(4)]
                for random, tape in zip(rs, tapes): random.bit_generator.state = copy.deepcopy(tape)
                policies = [Agent(ModelBidder(MODEL), ta_plus_v2()) for _ in range(4)]
                run_round(s, policies, rs, check=True)
                assert s.bids == outcome['bids'] and s.tricks_won == outcome['tricks'] and s.scores() == outcome['scores']
                assert outcome['hero_hit'] == int(s.bids[0] == s.tricks_won[0]) and outcome['hero_score'] == s.scores()[0]
                rounds += 1
            complete += 1
            if complete % 1000 == 0: save(HERE/'results/nt_replay_progress.json', {'stage': stage, 'verified_contexts': complete, 'verified_rounds': rounds, 'utc': stamp()})
        assert complete == count
        reports[stage] = {'contexts': complete, 'counterfactual_rounds': rounds, 'all_seed_deals_bids_hits_scores_match': True, 'raw_sha256': sha(NT/f'{stage}_contexts.jsonl')}
        save(HERE/'results/nt_labels_verification.json', {'stages': reports, 'status': 'in progress', 'utc': stamp()})
    normalization = json.loads((NT/'training_normalization.json').read_text())
    mu = mu_sum/40000; sd = np.sqrt(np.maximum(square_sum/40000-mu*mu, 0)); sd[sd < 1e-10] = 1
    np.testing.assert_allclose(mu, normalization['mu'], atol=3e-11, rtol=0)
    np.testing.assert_allclose(sd, normalization['sd'], atol=3e-10, rtol=0)
    assert abs(bce_normalizer_sum/40000-normalization['bce_fixed_norm']) < 2e-12
    save(HERE/'results/nt_labels_verification.json', {'stages': reports, 'status': 'pass', 'all_training_mean_sd_recomputed': True, 'utc': stamp(), 'verifier_sha256': sha(__file__)})


def independent_probabilities(x, offset, params):
    result = np.empty(offset.shape)
    for start in range(0, len(x), 256):
        z = x[start:start+256].reshape(-1, WIDTH)
        for k in range(0, len(params), 2):
            z = z @ params[k]+params[k+1]
            if k < len(params)-2: z = np.tanh(z)
        z = z.reshape(-1, 14)+offset[start:start+256]
        result[start:start+256] = np.exp(-np.logaddexp(0., -z))
    return result


def verify_model_selection(packet):
    norm = json.loads((NT/'training_normalization.json').read_text()); mu, sd = np.array(norm['mu']), np.array(norm['sd'])
    rows = list(records('selection')); count = len(rows)
    x = np.zeros((count, 14, WIDTH)); offset = np.zeros((count, 14)); base = np.zeros_like(offset); mask = np.zeros_like(offset, bool); score = np.zeros_like(offset)
    for i, row in enumerate(rows):
        bids = row['context']['legal']; xx, off, pp = design(row['context'], MODEL)
        x[i, bids] = xx; offset[i, bids] = off; base[i, bids] = pp; mask[i, bids] = True
        score[i, bids] = [o['hero_score'] for o in row['outcomes']]
    x -= mu; x /= sd
    def scores(p):
        action = np.argmax(np.where(mask, p*(10+np.arange(14)), -np.inf), axis=1)
        return int(score[np.arange(count), action].sum())
    checked = 0
    for objective in ('bce', 'direct'):
        selected_probabilities = []
        for seed in range(5):
            folder = NT/f'{objective}_s{seed}'; receipt = json.loads((folder/'frozen.json').read_text())
            traces = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]; assert [r['epoch'] for r in traces] == list(range(1, 51))
            candidates = []
            for epoch in (10, 25, 50):
                a = dict(np.load(folder/f'candidate_{epoch}.npz')); params = [a[f'p{i}'] for i in range(6)]
                probabilities = independent_probabilities(x, offset, params); total = scores(probabilities)
                assert next(r['selection_score_sum'] for r in traces if r['epoch'] == epoch) == total
                candidates.append((-total, epoch))
                if epoch == receipt['selected_epoch']:
                    selected_probabilities.append(probabilities)
                    selected = dict(np.load(folder/'selected.npz'))
                    for i, p in enumerate(params): np.testing.assert_array_equal(p, selected[f'p{i}'])
            assert min(candidates) == (-receipt['selection_score_sum'], receipt['selected_epoch']); checked += 1
        ensemble = np.mean(selected_probabilities, axis=0); trials = []
        for alpha in (0., .25, .5, 1.): trials.append((-scores((1-alpha)*base+alpha*ensemble), alpha))
        chosen = min(trials); assert chosen[1] == packet['models'][objective]['alpha']
        recorded = packet['models'][objective]['selection_trials']
        assert all(-v[0] == r['score_sum'] and v[1] == r['alpha'] for v, r in zip(trials, recorded))
    assert checked == 10
    save(HERE/'results/nt_model_verification.json', {'status': 'pass', 'n_models': checked, 'all_candidate_scores_and_epoch_blend_choices_match': True, 'selection_contexts': count, 'utc': stamp()})


def paired_ci(values, seats, z=1.96):
    groups = [[float(x) for x, seat in zip(values, seats) if seat == s] for s in range(4)]
    means = [statistics.mean(g) for g in groups]; mean = statistics.mean(means)
    se = math.sqrt(math.fsum(statistics.variance(g)/len(g) for g in groups)/16)
    return {'n_games': len(values), 'n_by_seat': [len(g) for g in groups], 'mean': mean,
            'ci95': [mean-z*se, mean+z*se], 'halfwidth': z*se,
            'method': 'equal-seat paired normal95%CI' if z == 1.96 else 'Bonferroni4-stage sensitivity98.75%normalCI'}


def game_audit(folder, stage, base, count, pool, digest):
    data = []; round_count = 0
    for i in range(count):
        row = json.loads((folder/'results'/stage/f'{i:04d}.json').read_text()); hero = i % 4
        assert row['g'] == i and row['hero'] == hero and row['base'] == base and row['pool'] == pool and row['frozen_ensemble_sha256'] == digest
        assert set(row['arms']) == {'bce', 'direct', 'canonical'}
        for a in row['arms'].values():
            assert len(a['rounds']) == 25
            total = [0]*4
            for j, rnd in enumerate(a['rounds']):
                assert rnd['n'] == RULES.ladder()[j] and rnd['trump'] == RULES.trump_for_round(j)
                assert sum(rnd['tricks']) == rnd['n'] and sum(rnd['bids']) != rnd['n']
                assert rnd['scores'] == [10+b if b == t else 0 for b, t in zip(rnd['bids'], rnd['tricks'])]
                total = [x+y for x, y in zip(total, rnd['scores'])]; round_count += 1
            assert total == a['totals']
        for j in range(25):
            rounds = [row['arms'][a]['rounds'][j] for a in ('bce', 'direct', 'canonical')]
            if rounds[0]['trump'] != -1: assert rounds[0] == rounds[1] == rounds[2]
            else:
                first = (RULES.dealer_for_round(j)+1) % 4
                for p in [(first+k) % 4 for k in range((hero-first) % 4)]: assert len({r['bids'][p] for r in rounds}) == 1
        data.append(row)
    comparisons = {}
    for challenger, reference in (('bce', 'canonical'), ('direct', 'canonical'), ('bce', 'direct')):
        score = []; wins = []; seats = []
        for row in data:
            h = row['hero']; a = row['arms'][challenger]['totals']; b = row['arms'][reference]['totals']
            score.append(a[h]-b[h]); wins.append(int(a[h] == max(a))-int(b[h] == max(b))); seats.append(h)
        comparisons[challenger+'-'+reference] = {'score': paired_ci(score, seats), 'ties_as_win': paired_ci(wins, seats),
            'score_familywise_sensitivity': paired_ci(score, seats, 2.497705474412374)}
    return {'stage': stage, 'base': base, 'pool': pool, 'n_triplets': count, 'n_round_records_checked': round_count, 'contrasts': comparisons}


def main():
    digest = sha(NT/'source_freeze.json'); source = json.loads((NT/'source_freeze.json').read_text())
    assert not any('D5' in rel or 'mcmc' in rel for rel in source['sources']), 'Unused held dependencies entered freeze'
    for rel, h in source['sources'].items(): assert sha(ROOT/rel) == h
    packet = json.loads((NT/'frozen_ensembles.json').read_text()); assert packet['source_freeze_sha256'] == digest
    assert packet['source_sha256'] == source['sources']
    for rel, h in packet['model_sha256'].items(): assert sha(NT/rel) == h
    for rel, h in packet['raw_sha256'].items(): assert sha(NT/rel) == h
    replay_labels(digest); verify_model_selection(packet)
    games = {}
    for folder, stage, base, count, pool in ((ROUND/'M4', 'screen', 1920000000, 2000, 'model'),
        (ROUND/'M4', 'confirmation', 1930000000, 2000, 'model'),
        (ROUND/'M5', 'human_screen', 1950000000, 1000, 'human'),
        (ROUND/'M5', 'human_confirmation', 1970000000, 1000, 'human'),
        (ROUND/'M5', 'greedy_screen', 1960000000, 1000, 'greedy'),
        (ROUND/'M5', 'greedy_confirmation', 1980000000, 1000, 'greedy')):
        games[stage] = game_audit(folder, stage, base, count, pool, sha(NT/'frozen_ensembles.json'))
        save(HERE/'results/nt_game_audit.json', {'stages': games, 'utc': stamp(), 'status': 'in progress'})
    save(HERE/'results/nt_game_audit.json', {'stages': games, 'utc': stamp(), 'status': 'pass', 'verifier_sha256': sha(__file__),
        'scope': 'Separate same-investigator raw/replay/selection/CI checks, not an external audit.95%normal intervals approximate and marginal.'})
    log(HERE, 'Full NT labels/selection/raw game/paired CI checks passed; fullgame fresh replay path still required.')


if __name__ == '__main__': main()
