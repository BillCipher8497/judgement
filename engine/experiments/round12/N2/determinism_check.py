"""Synthetic C1 fixtures: no RNG call, repeatability and fixed bid/play ties."""
import ast,inspect,random
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'experiments/round9/D3'))
from unittest.mock import patch
import numpy as np
from judgement.cfr import heuristics
from judgement.cfr.toy import ToyGame
from bounded_ta import ta_card,clear_mid,initial
from bid_premise import assert_matched

def check():
    source=inspect.getsource(heuristics);tree=ast.parse(source)
    assert not any(isinstance(n,ast.Name) and n.id in ('random','rng','default_rng','RandomState') for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('random','default_rng','RandomState','rand','randint','choice') for n in ast.walk(tree))
    g=ToyGame(N=4,S=2,R=4,n=2,trump=-1);hands=((0,1),(2,3),(4,5),(6,7))
    def no_rng(*a,**k):raise AssertionError('Policy attempted RNG use')
    with patch.object(np.random,'default_rng',no_rng),patch.object(np.random,'random',no_rng),patch.object(random,'random',no_rng):
        bids=[]
        for p in range(4):
            repeated=[heuristics.myopic_bid(g,p,hands[p],tuple(bids)) for _ in range(3)];assert len(set(repeated))==1;bids.append(repeated[0])
        assert_matched(g,hands,tuple(bids));clear_mid();initial.cache_clear();cold=ta_card(g,0,hands[0],tuple(bids));warm=ta_card(g,0,hands[0],tuple(bids));clear_mid();initial.cache_clear();assert ta_card(g,0,hands[0],tuple(bids))==cold==warm
    # Exact artificial EV tie; lower hook-legal bid wins by fixed -bid key.
    with patch.object(heuristics,'trick_dist_greedy',lambda *a:np.array([11.,10.,0.])):
        assert heuristics.myopic_bid(g,0,(0,4),())==0
        assert heuristics.myopic_bid(g,3,(0,4),(0,0,2))==1 # hook removes0
    # Exact artificial p(win) ties: policy's own fixed card-value/card-index keys.
    with patch.dict(ta_card.__globals__,{'_pwin_cached':lambda *a:{0:.5,4:.5}}):assert ta_card(g,0,(0,4),(0,0,0,1))==0
    with patch.dict(ta_card.__globals__,{'_pwin_cached':lambda *a:{0:0.,4:0.}}):assert ta_card(g,0,(0,4),(0,0,0,1))==0
    return {'C1_no_RNG_source_check':True,'C1_no_RNG_dynamic_fixture':True,'C1_cold_warm_repeatability':True,'fixed_lower_bid_and_card_ties':True,'C2_manual_hook_argmax_fixture':True,'scope':'Synthetic fixtures only; no experiment observations or training labels collected; in-memory patches restored.'}
