"""N2 posterior corpus views and exact fixed-profile joint baseline costs.
No experiment starts on import; no world data enter the observable encoder.
"""
import hashlib,json,math
from itertools import combinations
from functools import lru_cache
from pathlib import Path
import numpy as np
from export import decode
from features import encode,observation
from joint import feasible_count,integer_partition
BASE=2100000000;DOMAIN=0x4E32

def hand_partitions(g):
    def priority(h):return hashlib.sha256(json.dumps([BASE,DOMAIN,'hands',g.S,g.R,g.n,h],separators=(',',':')).encode()).digest()
    hands=sorted(combinations(g.deck,g.n),key=priority);a=6*len(hands)//10;b=8*len(hands)//10
    return {h:0 if i<a else 1 if i<b else 2 for i,h in enumerate(hands)}

class Corpus:
    def __init__(self,g,folder):
        self.g=g;self.folder=Path(folder);manifest=json.loads((self.folder/'posterior_manifest.json').read_text());p=self.folder/'posterior_supports.npz'
        assert manifest['condition']=={'N':g.N,'S':g.S,'R':g.R,'n':g.n,'trump':g.trump}
        assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['posterior_sha256']
        self.data=dict(np.load(p));self.keys=self.data['keys'];self.count=self.data['reach_count'];self.partitions=hand_partitions(g)
        self.split=np.array([self.partitions[decode(k)[1]] for k in self.keys],np.uint8)
        assert sum(self.count)==manifest['n_recorded_decision_worlds']
        for split in (0,1,2):
            expected={h for h,s in self.partitions.items() if s==split}
            actual={decode(self.keys[i])[1] for i in np.flatnonzero(self.split==split)}
            assert actual==expected,'Full trajectory corpus must reach every split hand'
    def selected(self,split,cap=None):
        ids=np.flatnonzero(self.split==split)
        if cap is not None and len(ids)>cap:
            prefix=json.dumps([BASE,DOMAIN,'fit-key-v1',self.g.S,self.g.R,self.g.n,self.g.trump],separators=(',',':')).encode()
            def priority(i):return hashlib.sha256(prefix+bytes(self.keys[i]).ljust(20,b'\0')).digest()
            ids=np.array(sorted(ids.tolist(),key=priority)[:cap],np.int64)
        return ids
    def row(self,i):
        obs=observation(self.g,self.keys[i]);x,mask,prop,public=encode(self.g,obs);cards=public['cards'];target=np.zeros((len(cards),4))
        if len(cards):target[:,:3]=self.data['holder_counts'][i,list(cards),:]/int(self.count[i])
        assert np.all(target[~mask]==0)
        return x,mask,prop,target,public
    def pack(self,ids):
        fields=[[],[],[],[]];offset=[0]
        for i in ids:
            x,m,p,t,_=self.row(int(i))
            for field,value in zip(fields,(x,m,p,t)):field.append(value)
            offset.append(offset[-1]+len(x))
        return {'X':np.concatenate(fields[0]),'M':np.concatenate(fields[1]),'P':np.concatenate(fields[2]),'T':np.concatenate(fields[3]),'offset':np.array(offset),'count':self.count[ids].astype(float),'ids':np.asarray(ids)}
    def describe(self,ids):
        return dict(n_keys=len(ids),n_reached_own_hands=len({decode(self.keys[i])[1] for i in ids}),exact_reach_weight=int(self.count[ids].sum()))

def marginal_risk(pack,prob):
    # Zero-card keys contribute0 while retaining their chance weight denominator.
    card=-np.sum(pack['T']*np.log(np.maximum(prob,1e-300)),axis=1);total=0.
    for a,b,w in zip(pack['offset'][:-1],pack['offset'][1:],pack['count']):
        if b>a:total+=float(w)*math.fsum(card[a:b])/int(b-a)
    return total/float(pack['count'].sum())

def structural_and_bid_KL(g,obs,public,reach_count,myopic_bid,eps=.02):
    assert eps==.02 # Exact rational1/50 in this declared BI-style toy model.
    caps=public['capacities'];cards=public['cards'];opponents=public['opponents'];U=len(cards)
    allowed=np.array([[int(c//g.R not in public['voids'][p] and caps[j]>0) for j,p in enumerate(opponents)] for c in cards],bool).reshape(U,3)
    number=feasible_count(caps,allowed);assert number>=reach_count>=1
    uniform=math.log(number)-math.log(reach_count)
    bids=tuple(obs['history'][:4]);cache={};matches={};legal_counts=[len(g.legal_bids(bids[:p])) for p in opponents]
    maxweight=math.prod(49*k+1 for k in legal_counts)
    def integer_weight(j,mask):
        if (j,mask) not in cache:
            p=opponents[j];hand=tuple(sorted(public['played'][p]+[c for i,c in enumerate(cards) if mask>>i&1]));assert len(hand)==g.n
            predicted=myopic_bid(g,p,hand,bids[:p]);matches[j,mask]=predicted==bids[p]
            cache[j,mask]=49*legal_counts[j]*int(matches[j,mask])+1
        return cache[j,mask]
    numerator=integer_partition(caps,allowed,integer_weight)
    kl=math.log(numerator/(reach_count*maxweight))
    def hard_allowed(j,hand):
        integer_weight(j,hand)
        return matches[j,hand]
    hard_count=feasible_count(caps,allowed,hard_allowed)
    assert number>=hard_count>=reach_count and hard_count*maxweight<=numerator<=number*maxweight
    return {'uniform_KL_float':max(0.,uniform),'prop_joint_KL_float':max(0.,uniform),'BI_KL_float':max(0.,kl),'raw_BI_KL_float':kl,
        'feasible_world_count':number,'hard_bid_consistent_world_count':hard_count,'hard_bid_KL_float':math.log(hard_count)-math.log(reach_count),
        'posterior_world_count':int(reach_count),'BI_integer_partition':numerator,'BI_max_integer_weight':maxweight,
        'BI_exact_log_ratio':[numerator,int(reach_count)*maxweight]},allowed
