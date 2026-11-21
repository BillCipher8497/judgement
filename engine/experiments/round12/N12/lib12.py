"""Frozen-belief decision bridge; import has no experiment side effects."""
import os,sys,json,time,importlib.util
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
H=Path(__file__).resolve().parent;R=H.parent;O=H/'results';sys.path.insert(0,str(R/'N9'))
import n9lib as B
import numpy as np
from fractions import Fraction
from scipy.stats import t
from bounded_ta import ta_card,clear_mid
from features import public_replay
from common import save,sha,stamp,log
TAGS=B.TAGS;CONFIGS=B.CONFIGS;LOSSES=B.LOSSES

def check():assert time.time()<float(os.environ['JUDGEMENT_JOB_DEADLINE'])
def rows(tag):return B.rows(B.load(R/'N9/results'/tag/'data.npz'))
def eligible(g,r):
    actor,hand,hist=B.decode(r.key);pub=public_replay(g,hist);left=[c for c in hand if c not in hist[4:]]
    return len(left)==2 and len(g.legal_cards(left,pub['trick']))==2
def worlds(g,r):
    actor,hand,hist=B.decode(r.key);pub=public_replay(g,hist);cards=[c for c in g.deck if c not in hand and c not in hist[4:]];opps=[(actor+k)%4 for k in (1,2,3)];out=[]
    for owners in r.F:
        w=[None]*4;w[actor]=tuple(hand)
        for k,p in enumerate(opps):w[p]=tuple(sorted(pub['played'][p]+[c for c,a in zip(cards,owners) if a==k]))
        out.append(tuple(w))
    return tuple(out)
def rollout(g,w,hero,hist,action):
    h=hist+(int(action),)
    while len(h)<4+4*g.n:
        left,_,trick,_,_,p=g.replay(w,h);choices=g.legal_cards(left[p],trick)
        if p==hero:assert len(choices)==1
        c=choices[0] if len(choices)==1 else ta_card(g,p,w[p],h);assert c in choices;h+=(c,)
    _,b,_,_,won,_=g.replay(None,h);return int(g.scores(b,won)[hero])
def ci(v):
    v=np.asarray(v,float);m=float(v.mean());h=float(t.ppf(.975,4)*v.std(ddof=1)/np.sqrt(5))
    return {'mean':m,'ci95':[m-h,m+h],'n_optimizer_seeds':5,'method':'paired t95%df4, optimizer seeds ONLY on known deterministic key subset'}
def probability(z,r):
    q=z[np.arange(len(z))[None,:],r.F].sum(1);q-=q.max();q=np.exp(q);return q/q.sum()
def summarize(results):
    def vals(tag,cfg,loss,budget,metric):return np.array([next(r[metric] for r in results if (r['tag'],r['config'],r['loss'],r['budget'],r['seed'])==(tag,cfg,loss,budget,s)) for s in range(5)])
    arms={};objective={};duration={};features={};kl={}
    for tag in TAGS:
        for cfg in CONFIGS:
            for loss in LOSSES:
                for budget in (20,80):
                    key=f'{tag}/{cfg}/{loss}/{budget}';arms[key]=ci(vals(tag,cfg,loss,budget,'private_regret'));kl[key]=ci(vals(tag,cfg,loss,budget,'joint_KL'))
                duration[f'{tag}/{cfg}/{loss}']=ci(vals(tag,cfg,loss,80,'private_regret')-vals(tag,cfg,loss,20,'private_regret'))
            for budget in (20,80):objective[f'{tag}/{cfg}/{budget}']=ci(vals(tag,cfg,'joint',budget,'private_regret')-vals(tag,cfg,'marginal',budget,'private_regret'))
        for loss in LOSSES:
            for budget in (20,80):features[f'{tag}/{loss}/{budget}']=ci(vals(tag,'rich_mlp',loss,budget,'private_regret')-vals(tag,'summary_mlp',loss,budget,'private_regret'))
    return {'rows':results,'arm_regret':arms,'joint_minus_marginal':objective,'long_minus_short':duration,'rich_minus_summary':features,'arm_KL_same_keys':kl,'primary':objective['S3R4n3_t-1/summary_mlp/80']}
