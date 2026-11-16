"""Exact toy evaluation of decentralized team CFR with a frozen TA hero.

Play-only intervention: all bidding and hero play frozen to TA/myopic profile.
Unilateral BR residuals do NOT certify joint team optimality. PI gives a lower
bound on achievable hero score, hence an upper bound on remaining joint gain.
"""
import json, os, threading, time, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import gc
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from judgement.cfr.toy import ToyGame, Tree
from judgement.cfr.heuristics import profile
from judgement.cfr.solve import reaches, cf_reach, values, profile_value, best_response
HERE=Path(__file__).resolve().parent

def pi_value(t,fixed,free):
    v=t.U[:,0].copy();sf=fixed.ravel()
    for a,b,seg,par in reversed(t.chunks):
        q=v[t.e_ch[a:b]]
        ordinary=np.add.reduceat(q*sf[t.e_ia[a:b]],seg)
        informed=np.minimum.reduceat(q,seg)
        v[par]=np.where(free[t.node_is[par]],informed,ordinary)
    return float(t.node_c[t.roots]@v[t.roots])

def assess(t,sig,free,pi):
    v=float(profile_value(t,sig)[0]);cf=cf_reach(t,reaches(t,sig))
    gains=[max(0.,best_response(t,sig,p,free=free,cf=cf[p])+v) for p in [1,2,3]]
    assert v>=pi-1e-8, 'PI relaxation must not exceed a feasible hidden-policy hero score'
    assert max(gains)<=v-pi+1e-8, 'Unilateral gain must fit inside PI joint-gain bound'
    return {'hero_score':v,'individual_spoiler_BR_gains':gains,'sum_unilateral_gains':sum(gains),
            'joint_gain_lower_bound':max(gains),'joint_gain_upper_bound_PI':max(0.,v-pi),
            'qualification':'Individual residual is not a joint coalition BR gap; PI upper bound may be loose.'}

def main():
    (HERE/'results').mkdir(parents=True,exist_ok=True)
    stop=min(time.time()+3*3600,datetime(2026,11,16,12,30,tzinfo=timezone.utc).timestamp())
    if time.time()>=stop:return
    lock=HERE/'worker.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        return
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    timer=threading.Timer(stop-time.time(),lambda:os._exit(0));timer.daemon=True;timer.start()
    for trump in [0,-1]:
        # The worker previously ran the test suite; release heuristic caches
        # before allocating the large exact tree, and between trump regimes.
        import judgement.cfr.heuristics as heuristics
        for obj in vars(heuristics).values():
            if hasattr(obj,'cache_clear'):
                obj.cache_clear()
        gc.collect()
        t=Tree(ToyGame(N=4,S=2,R=4,n=2,trump=trump));fixed=profile(t,play='ta',bid='myopic')
        free=(t.is_pl!=0)&~t.is_bid
        # Each spoiler has the shared objective -hero score; private information
        # sets stay separate. No pooled private cards and no mediator.
        t.U[:,1:]=-t.U[:,0,None]
        pi=pi_value(t,fixed,free)
        heuristic=fixed.copy()
        for p in [1,2,3]:
            _,br=best_response(t,fixed,p,free=free,return_policy=True)
            mask=free&(t.is_pl==p);heuristic[mask]=br[mask]
        summary={'trump':trump,'n_ordered_deals':len(t.roots),'nodes':t.n_nodes,
                 'fixed':'TA hero; all bids myopic fixed; optimize spoilers play only',
                 'PI_min_hero_score':pi,'baseline_TA_hero_score':float(profile_value(t,fixed)[0]),
                 'heuristic_independent_BR_to_TA':assess(t,heuristic,free,pi),
                 'uncertainty':'Exact full-deck enumeration, no sampling CI; no random seeds.', 'checkpoints':[]}
        R=np.zeros_like(fixed);S=np.zeros_like(fixed)
        for iteration in range(1,1001):
            z=R.sum(1,keepdims=True)
            sig=np.where(z>0,R/np.where(z>0,z,1),t.uniform());sig[~free]=fixed[~free]
            reach=reaches(t,sig);cf=cf_reach(t,reach);V=values(t,sig)
            p=t.e_pl;w=cf[p,t.e_par]*(V[t.e_ch,p]-V[t.e_par,p])
            inst=np.bincount(t.e_ia,weights=w,minlength=t.n_is*t.g.A).reshape(fixed.shape)
            R=np.maximum(R+inst,0)*t.is_valid;R[~free]=0
            own=reach[t.is_pl,t.is_rep];S+=iteration*own[:,None]*sig
            if iteration in [1,10,50,100,250,500,1000]:
                z=S.sum(1,keepdims=True);avg=np.where(z>0,S/np.where(z>0,z,1),t.uniform());avg[~free]=fixed[~free]
                np.testing.assert_array_equal(avg[~free],fixed[~free])
                np.testing.assert_allclose(avg.sum(1),1)
                entry={'iteration':iteration,**assess(t,avg,free,pi)}
                rz=R.sum(1,keepdims=True)
                current=np.where(rz>0,R/np.where(rz>0,rz,1),t.uniform());current[~free]=fixed[~free]
                entry['current_after_update']=assess(t,current,free,pi)
                entry['profile_file']=f'trump_{trump}_t{iteration}_profile.npz'
                summary['checkpoints'].append(entry)
                name=HERE/'results'/f'trump_{trump}'
                profile_path=HERE/'results'/entry['profile_file']
                temp=profile_path.with_suffix('.tmp')
                with temp.open('wb') as f:
                    np.savez_compressed(f,sigma=avg,current_sigma=current,fixed=fixed,free=free,iteration=iteration)
                temp.replace(profile_path)
                temp=name.with_suffix('.tmp_json')
                temp.write_text(json.dumps(summary,indent=2,allow_nan=False))
                temp.replace(name.with_suffix('.json'))
                with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:f.write(f'{datetime.now(timezone.utc).isoformat()} trump={trump}, T={iteration}: {json.dumps(entry)}\n')
        del t,fixed,free,heuristic,R,S,sig,reach,cf,V,avg,current
        gc.collect()
    timer.cancel()
    summaries={str(tr):json.loads((HERE/'results'/f'trump_{tr}.json').read_text()) for tr in [0,-1]}
    with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:
        f.write('READY FOR REVIEW\nFiles changed: T5/run.py, results/trump_{0,-1}.json and profile NPZs; engine/tests/test_round8_team.py.\nExact command: cd engine/experiments/round8/T5; python worker.py (focused test and compute in the freed slot); equivalent python run.py.\nSeed base: none, exact exhaustive enumeration of toy deals. n=2520 ordered deals per trump regime, n_cards=2, 4 players, 2 suits x 4 ranks. No sampling 95% CI applies to these exact values; approximation is optimization residual, not sample error. Results:\n'+json.dumps(summaries)+'\nLimitations: frozen TA hero and all myopic bids; play-only team. Separate private infosets, independent behavioral policies, no private-card pooling or correlated mediator. Heuristic baseline is simultaneous independent BR against TA teammates, not the full-deck heuristic spoiler. Sum of unilateral BR gains is NOT a joint coalition optimality certificate; report PI-based upper and individual-BR lower bounds on remaining joint gain.\n')
    board=HERE.parent/'BOARD.md';lines=board.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| T5 |'):
            cells=line.split('|');cells[-3]=' ready-for-review ';cells[-2]=' Exact toy evaluation complete; individual residuals and PI-based joint-gain bounds reported separately. Team optimality not certified. ';lines[i]='|'.join(cells)
    board.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lock.unlink(missing_ok=True)

if __name__=='__main__':main()
