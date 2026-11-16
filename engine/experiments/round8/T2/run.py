"""Fresh-seed paired top-pool replication, resumable per completed game/arm."""
import json
import os
from pathlib import Path
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments' / 'round6'))
from agents import make_agent
from judgement import Rules, play_game

HERE = Path(__file__).resolve().parent
RES = HERE/'results'
BASE = 1_000_000_000
STOP = datetime(2026, 11, 16, 12, 30, tzinfo=timezone.utc).timestamp()
KINDS = {'ref': 'hybrid_bi:k=32,c=6', 'sanity': 'hybrid_bi:k=32,c=6',
         'rb6': 'hybrid_rb:k=32,c=6', 'rb7': 'hybrid_rb:k=32,c=7'}


def log(s):
    with (HERE/'STATUS.txt').open('a', encoding='utf-8') as f:
        f.write(datetime.now(timezone.utc).isoformat()+' '+s+'\n')


def hard_stop():
    log('HARD DEADLINE: terminating current game; completed game files preserved. Run analyze.py before review.')
    os._exit(0)


def main():
    RES.mkdir(exist_ok=True)
    delay = STOP-time.time()
    if delay <= 0:
        return
    timer = threading.Timer(delay, hard_stop)
    timer.daemon = True
    timer.start()
    log('RUNNING: seed base 1000000000, g=0..299, hero=g%4; one CPU process. All-pool reference uses seat seeds [BASE,g,7,p]; challengers use [BASE,g,5].')
    for g in range(300):
        cs = g%4
        for arm, kind in KINDS.items():
            path = RES/f'{g:04d}_{arm}.json'
            if path.exists():
                continue
            agents = [make_agent(KINDS['ref'], int(np.random.SeedSequence([BASE,g,7,p]).generate_state(1)[0])) for p in range(4)]
            if arm != 'ref':
                agents[cs] = make_agent(kind, int(np.random.SeedSequence([BASE,g,5]).generate_state(1)[0]))
            start = time.time()
            game = play_game(Rules(), agents, seed=BASE+g, record_trace=False)
            out = {'g':g, 'arm':arm, 'seed_base':BASE, 'seat':cs, 'kind':kind,
                   'totals':list(map(float,game.totals)), 'seconds':time.time()-start,
                   'bids':[list(map(int,r.bids)) for r in game.rounds],
                   'tricks':[list(map(int,r.tricks)) for r in game.rounds],
                   'scores':[list(map(float,r.scores)) for r in game.rounds]}
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(out), encoding='utf-8')
            temp.replace(path)
            if arm == 'rb7' and (g+1)%10 == 0:
                log(f'CHECKPOINT: {g+1} complete paired games per arm; raw outcomes saved.')
    timer.cancel()
    log('COMPUTE COMPLETE: 300 games per arm; analyze and submit for review. No new selection was performed.')


if __name__ == '__main__':
    main()
