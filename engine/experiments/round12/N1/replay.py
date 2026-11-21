"""N1 validation: manual prefix transitions + saved-seed sampler replay.
New audit path; never modifies the approved/frozen generator, models or data.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,time,hashlib
from types import SimpleNamespace
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;ENGINE=H.parents[2];sys.path[:0]=[str(H),str(R),str(ENGINE)]
from judgement import Rules
from judgement.bidding.model import BidModel,ModelBidder
from judgement.policies.target_aware import TargetAwarePolicy
from judgement.policies.ta_plus import ta_plus_v2
from judgement.sampling import HandSampler
from judgement.bidding2.sampler import BidLikelihood
from judgement.policies.hybrid_bi import BidInformedWorlds
RULES=Rules();MODEL=BidModel.load(ENGINE/'experiments/phase1/results/model_ta_final.json');DOMAIN=0x4E31
O=H/'results'
STAGES={'pilot':(2100000000,0,176),'train':(2100000000,1,30800),'validation':(2100000000,2,4400),'confirmation':(2110000000,3,3520)}

def winner(trick,trump):
    led=trick[0][1]//13
    def strength(pc):
        s,r=divmod(pc[1],13);return (2 if trump!=-1 and s==trump else int(s==led),r)
    return max(trick,key=strength)

class ManualPrefix:
    def __init__(self,n,trump,dealer,hands):
        self.n=n;self.trump=trump;self.dealer=dealer;self.hands=hands[:];self.bids=[];self.history=[];self.trick=[];self.won=[0]*4;self.voids=[0]*4;self.winners=[];self.leader=(dealer+1)%4
    def view(self,p):
        left=[self.n-sum(int(pp==q) for pp,_ in self.history) for q in range(4)]
        assert left==[h.bit_count() for h in self.hands]
        wc,ws=-1,-1
        if self.trick:ws,wc=winner(self.trick,self.trump)
        played=0
        for _,c in self.history:played|=1<<c
        return SimpleNamespace(rules=RULES,n_players=4,seat=p,dealer=self.dealer,n_cards=self.n,trump=self.trump,hand=self.hands[p],bids=tuple(self.bids),
            tricks_won=tuple(self.won),current_trick=tuple(self.trick),history=tuple(self.history),voids=tuple(self.voids),trick_winners=tuple(self.winners),trick_len=len(self.trick),
            played_mask=played,winning_card=wc,winning_seat=ws,led_suit=self.trick[0][1]//13 if self.trick else -1,cards_left=lambda q:left[q],phase=0 if len(self.bids)<4 else 1)
    def legal(self,p):
        hand=self.hands[p]
        if self.trick:
            follow=hand&(((1<<13)-1)<<(13*(self.trick[0][1]//13)))
            if follow:return follow
        return hand
    def play(self,p,c):
        assert p==(self.leader+len(self.trick))%4 and self.legal(p)&(1<<c)
        if self.trick and c//13!=self.trick[0][1]//13:self.voids[p]|=1<<(self.trick[0][1]//13)
        self.hands[p]^=1<<c;self.trick.append((p,c));self.history.append((p,c))
        if len(self.trick)==4:
            self.leader=winner(self.trick,self.trump)[0];self.won[self.leader]+=1;self.winners.append(self.leader);self.trick=[]

def marginal(worlds,cards,opponents):
    counts=np.zeros((len(cards),4),np.int64)
    for world in worlds:
        for i,c in enumerate(cards):
            holders=[j for j,p in enumerate(opponents) if world[p]&(1<<c)];assert len(holders)<=1;counts[i,holders[0] if holders else 3]+=1
    return counts/len(worlds)

def rebuild(stage,g):
    base,code,_=STAGES[stage];cell=(g//4)%44;n=3+cell%11;trump=-1 if (cell//11)%2 else 0;table=cell//22;hero=g%4;dealer=(g//176)%4
    # Spawn and permutation reconstructed independently of make_rngs/deal.
    children=np.random.SeedSequence([base,DOMAIN,code,g,3]).spawn(5);permutation=np.random.default_rng(children[0]).permutation(52).tolist();rngs=[np.random.default_rng(s) for s in children[1:]]
    hands=[sum(1<<c for c in permutation[p*n:(p+1)*n]) for p in range(4)];state=ManualPrefix(n,trump,dealer,hands)
    bidder=ModelBidder(MODEL);policies=[ta_plus_v2() if table else TargetAwarePolicy() for _ in range(4)]
    for k in range(4):
        p=(dealer+1+k)%4;legal=list(range(n+1))
        if k==3:
            forbidden=n-sum(b for _,b in state.bids)
            if forbidden in legal:legal.remove(forbidden)
        bid=bidder.bid(state.view(p),legal,rngs[p]);assert bid in legal;state.bids.append((p,bid))
    target=int(np.random.default_rng([base,DOMAIN,code,g,4]).integers(1,n-1))
    while len(state.history)<4*n:
        p=(state.leader+len(state.trick))%4
        if p==hero and len(state.history)//4>=target:break
        c=int(policies[p].play(state.view(p),state.legal(p),rngs[p]));state.play(p,c)
    assert len(state.history)<4*n
    info=state.view(hero)
    obs=dict(seat=hero,n=n,trump=trump,dealer=dealer,hand=info.hand,history=[list(v) for v in info.history],bids=[list(v) for v in info.bids],voids=list(info.voids),won=list(info.tricks_won),left=[info.cards_left(p) for p in range(4)])
    return hands,obs,info,target

def main():
    deadline=float(os.environ['JUDGEMENT_N1_AUDIT_DEADLINE']);results={};sampler_checks=0;prefix_checks=0
    likelihood=BidLikelihood(MODEL,RULES,eps=.02,tau=0)
    for stage,(base,code,count) in STAGES.items():
        if not (O/stage).exists():continue
        wanted=sorted(set(list(range(min(704,count)))+list(range(count-176,count))))
        selected={}
        for start in range(0,count,64):
            if not any(start<=g<start+64 for g in wanted):continue
            p=O/stage/f'{start:05d}';receipt=json.loads(p.with_suffix('.json').read_text());a=np.load(p.with_suffix('.npz'));rows=[json.loads(s) for s in p.with_suffix('.jsonl').read_text().splitlines()]
            for suffix,field in (('.npz','arrays_sha256'),('.jsonl','records_sha256')):assert hashlib.sha256(p.with_suffix(suffix).read_bytes()).hexdigest()==receipt[field]
            for j,row in enumerate(rows):
                if row['g'] not in wanted:continue
                assert time.time()<deadline;g=row['g'];initial,obs,info,target=rebuild(stage,g)
                assert initial==row['initial_hands'] and obs==row['observation'] and target==row['target_trick'];prefix_checks+=1
                first,last=a['offset'][j:j+2];seen=info.hand|info.played_mask;cards=[c for c in range(52) if not seen>>c&1];assert cards==row['cards']
                opponents=[(g%4+k)%4 for k in (1,2,3)]
                # Four replication blocks cover all44cells×4seats×4dealers
                # in confirmation; the pilot has one fixed176record block.
                if stage in ('pilot','confirmation') and g<min(704,count):
                    seed=[base,DOMAIN,code,g,5];hs=HandSampler(info);rng=np.random.default_rng(seed);uniform=[hs.sample(rng) for _ in range(256)]
                    bi,stats=BidInformedWorlds(likelihood,batch=64,m_max=2048,ess_target=1).sample(info,256,np.random.default_rng(seed))
                    np.testing.assert_array_equal(marginal(uniform,cards,opponents),a['uniform'][first:last]);np.testing.assert_array_equal(marginal(bi,cards,opponents),a['bi'][first:last])
                    for k,v in stats.items():assert float(v)==row['sampler'][k]
                    sampler_checks+=1
                selected[g]=True
            (O/'fresh_prefix_replay_progress.json').write_text(json.dumps({'status':'partial audit, not task validation','current_stage':stage,'selected_prefixes_checked_in_stage':len(selected),'n_manual_prefixes_checked':prefix_checks,'n_sampler_decisions_checked':sampler_checks,'completed_stages':results,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2),encoding='utf-8')
        assert sorted(selected)==wanted;results[stage]={'n_manual_prefixes':len(selected),'selected_g':'first min(704,n) and last176, de-duplicated','n_sampler_replay_decisions':min(704,count) if stage in ('pilot','confirmation') else 0}
    value={'status':'ready-for-Claude-external','stages':results,'n_manual_prefix_replays':prefix_checks,'n_sampler_replay_decisions':sampler_checks,'K_per_sampler':256,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'scope':'New manual seed/deal/turn/follow-suit/void/trick transition path, frozen bidder/play functions reused. K256samplers replayed from same seeds with independent public-only view. Selected addresses, not an independent full-population simulator or exact posterior certificate.'}
    (O/'fresh_prefix_sampler_replay.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps(value))
if __name__=='__main__':main()
