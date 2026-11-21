"""Fresh complete-game driver for fixed, outcome-independent replay addresses."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; ENGINE = HERE.parents[2]
sys.path[:0] = [str(ROUND/'M4'), str(ROUND), str(ENGINE)]
from common import save, sha, stamp, log
from residual import Ensemble, NTResidualBidder, MODEL_PATH
from judgement import Rules, play_game
from judgement.bidding.model import BidModel, ModelBidder, Agent
from judgement.policies.ta_plus import ta_plus_v2
from judgement.policies.human_guide import HumanBidder
from judgement.policies.greedy import GreedyPolicy


def main():
    directory = ROUND/'M4/results'; packet = json.loads((directory/'frozen_ensembles.json').read_text()); model = BidModel.load(MODEL_PATH)
    ensembles = {k: Ensemble(directory, v, model) for k, v in packet['models'].items()}; rules = Rules(); reports = {}
    for folder, stage, base, count, pool in ((ROUND/'M4', 'screen', 1920000000, 2000, 'model'),
        (ROUND/'M4', 'confirmation', 1930000000, 2000, 'model'),
        (ROUND/'M5', 'human_screen', 1950000000, 1000, 'human'),
        (ROUND/'M5', 'human_confirmation', 1970000000, 1000, 'human'),
        (ROUND/'M5', 'greedy_screen', 1960000000, 1000, 'greedy'),
        (ROUND/'M5', 'greedy_confirmation', 1980000000, 1000, 'greedy')):
        blocks = [0, 1, count//8, count//4-2, count//4-1]
        addresses = [4*k+p for k in blocks for p in range(4)]; completed = 0
        assert len(set(addresses)) == 20
        for g in addresses:
            saved = json.loads((folder/'results'/stage/f'{g:04d}.json').read_text()); hero = g % 4
            for arm in ('canonical', 'direct', 'bce'):
                if pool == 'model': bidders = [ModelBidder(model) for _ in range(4)]
                elif pool == 'human': bidders = [HumanBidder() for _ in range(4)]
                else: bidders = [GreedyPolicy() for _ in range(4)]
                bidders[hero] = ModelBidder(model) if arm == 'canonical' else NTResidualBidder(ensembles[arm], ModelBidder(model))
                agents = [Agent(b, ta_plus_v2()) for b in bidders]
                result = play_game(rules, agents, base+g, record_trace=False)
                rebuilt = {'totals': result.totals, 'rounds': [{'n': r.n_cards, 'trump': r.trump,
                    'bids': r.bids, 'tricks': r.tricks, 'scores': r.scores} for r in result.rounds]}
                assert rebuilt == saved['arms'][arm], (stage, g, arm)
                completed += 1
        reports[stage] = {'n_fresh_games': completed, 'n_triplets': 20, 'g_addresses': addresses,
            'seat_counts': [5]*4, 'all_totals_rounds_bids_tricks_scores_match': True}
        save(HERE/'results/fullgame_replay_verification.json', {'status': 'in progress', 'stages': reports, 'utc': stamp()})
    save(HERE/'results/fullgame_replay_verification.json', {'status': 'pass', 'stages': reports,
        'n_total_fresh_games': 360, 'source_sha256': sha(__file__), 'utc': stamp(),
        'limitation': 'Fresh driver and frozen predictors on the same validated core engine, not a second engine implementation.'})
    log(HERE, 'Fresh360full-game replays agree across all six stages, balanced5addresses/seat/stage.')
    import runpy
    runpy.run_path(str(HERE/'verify_optional.py'), run_name='__main__')


if __name__ == '__main__': main()
