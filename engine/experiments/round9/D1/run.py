import json,sys,time,runpy
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ENGINE=HERE.parents[2]
sys.path[:0]=[str(ENGINE),str(HERE),str(HERE.parent)]
from common import log,save,status,timer,DEADLINE
from judgement import Rules,RoundState
from judgement.engine import make_rngs,deal,run_round
from judgement.bidding.model import BidModel,ModelBidder
from judgement.policies.ta_plus import ta_plus_v2
from judgement.policies.spoiler import OmniscientSpoilerPolicy
from intervention import InformationSpoiler

def main():
    (HERE/'results').mkdir(parents=True,exist_ok=True);cut=timer()
    rules=Rules();bidder=ModelBidder(BidModel.load(ENGINE/'experiments/phase1/results/model_ta_final.json'))
    for base in [1600000000,1610000000]:
        for g in range(1000):
            path=HERE/'results'/f'{base}_{g:04d}.json'
            if path.exists():continue
            if time.time()>=DEADLINE:break
            seed=[base,5,g];dr,pr=make_rngs(seed,4)
            st=RoundState(rules,5,0 if g%2==0 else -1,(g//2)%4,deal(rules,5,dr))
            initial=st.hands.copy()
            while st.phase==0:
                p=st.to_act;st.apply_bid(bidder.bid(st.info_state(p),st.legal_bids(),pr[p]))
            arms={};start=time.time()
            for name in ['self','own','hero','all','omni']:
                pols=[ta_plus_v2() if p==0 or name=='self' else
                      OmniscientSpoilerPolicy(0) if name=='omni' else InformationSpoiler(0,name,16)
                      for p in range(4)]
                world=st.copy();_,rngs=make_rngs(seed,4);run_round(world,pols,rngs,check=True)
                arms[name]={'bids':world.bids,'tricks':world.tricks_won,'scores':world.scores(),'history':world.history}
            assert arms['all']==arms['omni'],'All-info PIMC must degenerate to the exact reference'
            save(path,{'g':g,'base':base,'seed':seed,'hero':0,'n':5,'trump':st.trump,'dealer':st.dealer,
                       'initial_hands':initial,'fixed_bids':st.bids,'arms':arms,'seconds':time.time()-start})
            if (g+1)%10==0:log('D1',f'base={base} {g+1}/1000 complete five-arm deals')
    cut.cancel();runpy.run_path(str(HERE/'analyze.py'),run_name='__main__')
if __name__=='__main__':main()
