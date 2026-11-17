"""Use a free slot to optimize the completed trump tree within the same box."""
import json,pickle,sys,time
from pathlib import Path
from fractions import Fraction
import numpy as np
HERE=Path(__file__).resolve().parent;ENGINE=HERE.parents[2]
sys.path[:0]=[str(ENGINE),str(HERE),str(HERE.parent),str(ENGINE/'experiments/round8/T5')]
from common import timer,save,replace,log,DEADLINE
from judgement.cfr.solve import CFRPlus,profile_value
from compact_tree import CompactTree
from certify import certify

def main():
    started=json.loads((HERE/'results/timebox.json').read_text())['started_unix']
    stop=min(started+10800,DEADLINE);cut=timer(stop)
    meta=json.loads((HERE/'results/tree_cache_0.json').read_text())
    with Path(meta['derived_cache']).open('rb') as f:t=pickle.load(f)
    base=json.loads((HERE/'results/trump_0.json').read_text())
    with np.load(HERE/'results/trump_0_profile.npz') as z:best=z['sigma'].copy();fixed=z['fixed'].copy()
    assert t.n_deals==369600 and best.shape==t.is_valid.shape
    value=float(profile_value(t,best)[0]);assert abs(value-base['best_feasible']['hero_score'])<1e-7
    learner=CFRPlus(t);best_iteration=0;checkpoints=[]
    def checkpoint(final=False):
        nonlocal best,value,best_iteration
        candidates={'average':learner.average(),'current':learner.current()}
        one=np.zeros_like(best);one[np.arange(t.n_is),np.where(t.is_valid,candidates['current'],-1).argmax(1)]=1
        candidates['one_hot']=one;observed={}
        for name,sigma in candidates.items():
            score=float(profile_value(t,sigma)[0]);observed[name]=score
            if score<value:value=score;best=sigma.copy();best_iteration=learner.T
        checkpoints.append({'iteration':learner.T,'candidate_scores':observed})
        item={'trump':0,'n_exact_deals':t.n_deals,'iteration':learner.T,'best_iteration':best_iteration,
              'PI_fraction':base['PI_fraction'],'baseline_TA':base['baseline_TA'],
              'best_feasible_score':value,'PI_achieved_interval':[base['PI_min'],value],
              'hidden_to_PI_surplus':(base['baseline_TA']-value)/(base['baseline_TA']-base['PI_min']),
              'checkpoints':checkpoints,'same_original_deadline':stop,
              'uncertainty':'Complete chance space, no sampling CI; bounded optimizer, not a convergence claim.'}
        if final:
            certificate,_=certify(t,fixed,np.ones(t.n_is,bool),best);item['certificate']=certificate
        save(HERE/'results/trump_0_optimization.json',item)
        temp=HERE/'results/trump_0_optimized_profile.tmp'
        with temp.open('wb') as f:np.savez_compressed(f,sigma=best,fixed=fixed,iteration=learner.T)
        replace(temp,HERE/'results/trump_0_optimized_profile.npz')
        log('D3',f'Second-slot trump optimization T{learner.T}, achieved {value:.10f}; final={final}')
    while learner.T<1000 and time.time()<stop-120:
        learner.iterate(1)
        if learner.T in [1,5,10,25,50,100,200,400,800,1000]:checkpoint()
    checkpoint(final=True);cut.cancel()
if __name__=='__main__':main()
