"""Independently compare fitted vector bids to the deployed bidder; audit raw games."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from judgement.policies.human_guide_v2 import HumanBidderV2
from run import predicted_bids
HERE=Path(__file__).resolve().parent
fit=json.loads((HERE/'results/fit.json').read_text());config=fit['config']
digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
assert digest==fit['frozen_sha256']
bidder=HumanBidderV2(config['weights']);checked=0
for line in (HERE/'results/train.jsonl').open(encoding='utf-8'):
    r=json.loads(line)
    info=SimpleNamespace(n_cards=r['n'],n_players=4,seat=r['hero'],dealer=r['dealer'],hand=r['hand'],bids=r['prior_bids'],trump=r['trump'])
    legal=np.zeros((1,14),bool);legal[0,r['legal']]=True
    vector=predicted_bids(np.array([r['points']]),np.array([r['features']]),np.array([r['adjust']]),np.array([r['n']]),legal,config['weights'])[0]
    actual=bidder.bid(info,r['legal'],None)
    assert int(vector)==actual,(r['i'],vector,actual)
    checked+=1
assert checked==60000
out={'training_bids_checked':checked,'vector_deployment_mismatches':0,'frozen_sha256':digest,'games':{}}
ladder=np.array(list(range(13,0,-1))+list(range(2,14)))
for stage,seed,target in [('screen',1300000000,2000),('validation',1400000000,4000)]:
    count=0
    for g,line in enumerate((HERE/'results'/f'{stage}.jsonl').open(encoding='utf-8')):
        r=json.loads(line)
        assert r['g']==g and r['seat']==g%4 and r['seed_base']==seed and r['frozen_sha256']==digest
        for a in r['arms']:
            b,t,s=np.asarray(a['bids']),np.asarray(a['tricks']),np.asarray(a['scores'])
            assert b.shape==t.shape==s.shape==(25,4)
            np.testing.assert_array_equal(t.sum(1),ladder)
            assert np.all(b.sum(1)!=ladder)
            np.testing.assert_array_equal(s,np.where(b==t,10+b,0))
            np.testing.assert_array_equal(s.sum(0),a['totals'])
        count+=1
    assert count==target
    out['games'][stage]={'complete_paired_games':count,'all_seeds_seats_scores_and_hashes_valid':True}
(HERE/'results/verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:
    f.write('Independent deployment/data check: python verify.py from T6. '+json.dumps(out)+'\n')
print(json.dumps(out,indent=2))
