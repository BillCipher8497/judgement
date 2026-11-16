"""Mode comparison on original phase1 grid; target-aware laws are bid-conditional.

Original seed 4242 (Greedy baseline reproduced), independent replication 4243.
40 per cell x 13 sizes x 2 trump types x 4 positions; 300 worlds per situation.
For TA/TA+, all legal candidate bids use paired hidden worlds. Mode/mean are
evaluated in the law at the score-optimal bid, not an invented bid-free law.
"""
import json
from pathlib import Path
import sys
import os
import threading
from datetime import datetime, timezone
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from judgement import Rules, RoundState
from judgement.engine import deal
from judgement.bidding.model import BidModel, ModelBidder, Agent, info_features
from judgement.bidding.sim import bid_free_dist, bid_conditional_dists, model_bid
from judgement.policies.greedy import GreedyPolicy
from judgement.policies.target_aware import TargetAwarePolicy
from judgement.policies.ta_plus import ta_plus_v2
HERE=Path(__file__).resolve().parent

def main():
    (HERE/'results').mkdir(parents=True,exist_ok=True)
    stop=datetime(2026,11,16,12,30,tzinfo=timezone.utc).timestamp()
    if time.time()>=stop:return
    timer=threading.Timer(stop-time.time(),lambda:os._exit(0));timer.daemon=True;timer.start()
    rules=Rules()
    mg=BidModel.load(ROOT/'experiments/phase1/results/model_greedy_final.json')
    mt=BidModel.load(ROOT/'experiments/phase1/results/model_ta_final.json')
    ag=[Agent(ModelBidder(mg),GreedyPolicy())]*4
    policies={'ta': [Agent(ModelBidder(mt),TargetAwarePolicy())]*4,
              'ta_plus':[Agent(ModelBidder(mt),ta_plus_v2())]*4}
    for seed in [4242,4243]:
        fn=HERE/'results'/f'raw_{seed}.jsonl'
        lines=fn.read_bytes().splitlines(keepends=True) if fn.exists() else []
        if lines:
            try:
                json.loads(lines[-1])
            except json.JSONDecodeError:
                (HERE/'results'/f'incomplete_tail_{seed}.txt').write_bytes(lines.pop())
                temp=fn.with_suffix('.resume_tmp');temp.write_bytes(b''.join(lines));temp.replace(fn)
        done=len(lines)
        if done==4160:
            continue
        rng=np.random.default_rng(seed);cache={};idx=0
        for n in range(1,14):
            for trump in [0,-1]:
                for pos in range(4):
                    hero=(pos+1)%4
                    for rep in range(40):
                        st=RoundState(rules,n,trump,0,deal(rules,n,rng))
                        while st.to_act!=hero:
                            b,_=model_bid(mg,rules,info_features(st.info_state(st.to_act)),n,trump,st.legal_bids(),cache)
                            st.apply_bid(b)
                        legal=st.legal_bids()
                        # Always advance the original stream, including on resume.
                        world_rng_state=rng.bit_generator.state
                        dg=bid_free_dist(st,hero,ag,300,rng)
                        if idx<done:idx+=1;continue
                        distributions={'greedy':{b:dg for b in legal}}
                        for k,(name,players) in enumerate(policies.items()):
                            paired_rng=np.random.default_rng([seed,idx,91,k])
                            if seed==4243:
                                paired_rng.bit_generator.state=world_rng_state
                            distributions[name]=bid_conditional_dists(st,hero,players,legal,300,paired_rng)
                        record={'i':idx,'n':n,'trump':trump,'pos':pos,'rep':rep,'seed':seed,'legal':legal,
                                'world_pairing':('all policies share the pre-Greedy RNG state; deterministic policies'
                                    if seed==4243 else 'discovery uses independent policy MC streams; own contexts shared'),
                                'hand':int(st.hands[hero]),'prior_bids':list(st.bid_seq),'policies':{}}
                        for name,ds in distributions.items():
                            b=max(legal,key=lambda b:((10+b)*ds[b][b],-b))
                            law=ds[b];mu=float(np.arange(n+1)@law)
                            record['policies'][name]={'b':b,'mode':int(law.argmax()),'mu':mu,'floor':int(np.floor(mu)),
                                'ceil':int(np.ceil(mu)),'round':int(np.floor(mu+.5)),
                                'laws':{str(k):d.tolist() for k,d in ds.items()}}
                        with fn.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
                        idx+=1
                        if idx%40==0:
                            with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:f.write(f'{datetime.now(timezone.utc).isoformat()} seed={seed}: {idx}/4160 situations complete; all laws saved.\n')
    timer.cancel()
    with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:f.write('COMPUTE COMPLETE: discovery and fresh-seed replication. Run analyze.py; review bid-conditional interpretation.\n')

if __name__=='__main__':main()
