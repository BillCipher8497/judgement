import json,sys,time,runpy
from pathlib import Path
HERE=Path(__file__).resolve().parent;ENGINE=HERE.parents[2]
sys.path[:0]=[str(ENGINE),str(HERE),str(HERE.parent)]
from common import timer,log,DEADLINE
from judgement import Rules,play_game
from judgement.bidding.model import Agent,BidModel,ModelBidder
from judgement.policies.target_aware import TargetAwarePolicy
from judgement.policies.human_guide import GuideAgent,HumanBidder,HumanPlayPolicy

class Recorder:
    def __init__(self,bidder):self.bidder=bidder;self.records=[]
    def bid(self,info,legal,rng):
        b=self.bidder.bid(info,legal,rng)
        self.records.append({'hand':info.hand,'prior':list(info.bids),'legal':legal,'bid':b})
        return b

def main():
    (HERE/'results').mkdir(parents=True,exist_ok=True);cut=timer()
    rules=Rules();model=BidModel.load(ENGINE/'experiments/phase1/results/model_ta_final.json')
    for base in [1500000000,1510000000]:
        path=HERE/'results'/f'raw_{base}.jsonl'
        lines=path.read_bytes().splitlines(keepends=True) if path.exists() else []
        if lines:
            try:json.loads(lines[-1])
            except json.JSONDecodeError:
                path.with_suffix('.incomplete_tail').write_bytes(lines.pop());path.write_bytes(b''.join(lines))
        for g in range(len(lines),4000):
            if time.time()>=DEADLINE:break
            seat=g%4;arms=[]
            for kind in ['guide','model']:
                recorder=Recorder(HumanBidder() if kind=='guide' else ModelBidder(model))
                agents=[Agent(ModelBidder(model),TargetAwarePolicy()) for _ in range(4)]
                agents[seat]=GuideAgent(recorder,HumanPlayPolicy())
                game=play_game(rules,agents,seed=base+g,record_trace=False)
                arms.append({'totals':list(map(float,game.totals)),'hero_bid_contexts':recorder.records,
                             'bids':[list(map(int,r.bids)) for r in game.rounds],
                             'tricks':[list(map(int,r.tricks)) for r in game.rounds],
                             'scores':[list(map(float,r.scores)) for r in game.rounds]})
            for a,b in zip(arms[0]['hero_bid_contexts'],arms[1]['hero_bid_contexts']):
                assert a['hand']==b['hand'] and a['prior']==b['prior'] and a['legal']==b['legal']
            with path.open('a',encoding='utf-8') as f:
                f.write(json.dumps({'g':g,'base':base,'seat':seat,'arms':arms})+'\n')
            if (g+1)%100==0:log('D2',f'base={base} {g+1}/4000 paired games complete')
        # Freeze discovery top cells before the replication is computed.
        runpy.run_path(str(HERE/'analyze.py'),run_name='__main__')
    cut.cancel()
if __name__=='__main__':main()
