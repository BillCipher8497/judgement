import gc,json,sys,time,os,ctypes,pickle,hashlib,tempfile
from itertools import combinations
from pathlib import Path
from fractions import Fraction
import numpy as np
HERE=Path(__file__).resolve().parent;ENGINE=HERE.parents[2]
sys.path[:0]=[str(ENGINE),str(HERE),str(HERE.parent),str(ENGINE/'experiments/round8/T5')]
from common import save,log,status,timer,DEADLINE,replace
from judgement.cfr.toy import ToyGame
from judgement.cfr.heuristics import trick_dist_greedy
from bounded_ta import ta_card,clear_mid,pwin
from judgement.cfr.solve import reaches,cf_reach,values,profile_value,best_response
from compact_tree import CompactTree as FrozenTree
from reachable_ta import reachable_ta_profile
from root_bound import root_information_bound
from certify import certify,integer_bound

def memory():
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
            ['peak_work','work','peak_pool','pool','peak_nonpool','nonpool','page','peak_page','private']]
    c=Counters();c.cb=ctypes.sizeof(c)
    current=ctypes.windll.kernel32.GetCurrentProcess;current.restype=wintypes.HANDLE
    func=ctypes.windll.psapi.GetProcessMemoryInfo
    func.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    assert func(current(),ctypes.byref(c),c.cb)
    return c.private

def ta_profile(t):
    sigma=np.zeros(t.is_valid.shape)
    for i,(p,hand,hist) in enumerate(t.is_key):
        sigma[i,hand.index(ta_card(t.g,p,hand,hist))]=1
        if (i+1)%10000==0:
            log('D3',f'TA profile {i+1}/{t.n_is} infosets')
    return sigma

def assess(t,sig,pi):
    score=float(profile_value(t,sig)[0]);cf=cf_reach(t,reaches(t,sig))
    gains=[max(0.,best_response(t,sig,p,cf=cf[p])+score) for p in [1,2,3]]
    return {'hero_score':score,'unilateral_BR_gains':gains,
            'remaining_joint_gain_bounds':[max(gains),max(0.,score-pi)]}

def main():
    (HERE/'results').mkdir(parents=True,exist_ok=True)
    timebox=HERE/'results/timebox.json'
    if timebox.exists():started=json.loads(timebox.read_text())['started_unix']
    else:started=time.time();save(timebox,{'started_unix':started})
    stop=min(started+3*3600,DEADLINE);cut=timer(stop);start=started
    rng=np.random.default_rng(1620000000);probes=[];shape=None
    preflight=HERE/'results/resource_preflight.json'
    if preflight.exists():
        previous=json.loads(preflight.read_text())
        if previous.get('latest_feasible'):
            last=previous['probes'][-1];shape=(last['S'],last['R'])
            log('D3','Reusing completed compact-storage resource preflight; no game outcomes informed this choice.')
    for S,R in ([] if shape else [(3,4),(2,6)]):
        worst_nodes=0;worst_infosets=0;construction=0
        for trump in [0,-1]:
            g=ToyGame(N=4,S=S,R=R,n=3,trump=trump)
            warm=time.time()
            for hand_index,hand in enumerate(combinations(g.deck,3)):
                for p in range(4):trick_dist_greedy(g,p,hand)
                pwin(g,0,hand,())
                if (hand_index+1)%20==0:log('D3',f'Warmup S{S}R{R} trump{trump}: {hand_index+1}/220 hands')
            warm_seconds=time.time()-warm
            deals=[]
            for _ in range(250):
                cards=rng.permutation(12);deals.append(tuple(tuple(sorted(cards[p*3:(p+1)*3])) for p in range(4)))
            tick=time.time();t=FrozenTree(g,deals=deals);elapsed=time.time()-tick
            worst_nodes=max(worst_nodes,t.n_nodes/250);worst_infosets=max(worst_infosets,t.n_is/250)
            construction=max(construction,warm_seconds+elapsed/250*369600)
            probes.append({'S':S,'R':R,'trump':trump,'deals':250,'nodes':t.n_nodes,'infosets':t.n_is,
                           'seconds':elapsed,'one_time_warmup_seconds':warm_seconds,
                           'projected_seconds_conservative':warm_seconds+elapsed/250*369600})
            del t;gc.collect();clear_mid()
        projected_bytes=369600*(worst_nodes*250+worst_infosets*500)
        feasible=projected_bytes<10*1024**3 and construction<45*60
        save(HERE/'results/resource_preflight.json',{'probes':probes,'latest_projected_GiB':projected_bytes/1024**3,'latest_feasible':feasible})
        log('D3',f'Resource-only preflight S{S}R{R}: projected {projected_bytes/1024**3:.2f}GiB, {construction/60:.1f}min cold-cache conservative build; feasible={feasible}')
        if feasible:shape=(S,R);break
    if shape is None:
        log('D3','READY FOR REVIEW: resource preflight failed both predeclared exact designs. No sampled-chance result is substituted. Files D3/frozen_tree.py, run.py, STATUS and resource_preflight.json; command python experiments/round9/D3/run.py. Seed1620M, 250 resource-only sample deals/regime/design, no game-theory estimates or statistical CIs. Limitation: cannot deliver exact bounds within resource box; task scoped too large.');status('D3','ready-for-review','Exact designs exceed preregistered resource cap; scope failure documented, no sampled substitute.');cut.cancel();return
    results={}
    for regime_index,trump in enumerate([0,-1]):
        if time.time()>stop-120:break
        regime_stop=time.time()+(stop-time.time())/(2-regime_index)
        g=ToyGame(N=4,S=shape[0],R=shape[1],n=3,trump=trump)
        def checkpoint(deals,nodes,infosets):
            used=memory()
            if deals%5000==0:
                save(HERE/'results'/f'build_{trump}.json',{'deals_complete':deals,'nodes':nodes,'infosets':infosets,'private_GiB':used/1024**3,'seconds':time.time()-start,'shape':shape,'pid':os.getpid()})
                log('D3',f'Build trump{trump}: {deals}/369600 deals, nodes{nodes}, private{used/1024**3:.2f}GiB')
            if used>10*1024**3 or time.time()>regime_stop-120:raise RuntimeError('D3 resource/time cap reached during exact tree construction')
        fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in [HERE/'compact_tree.py',HERE/'bounded_ta.py',ENGINE/'judgement/cfr/toy.py',ENGINE/'judgement/cfr/heuristics.py'])).hexdigest()
        cache_dir=Path(tempfile.gettempdir())/'judgement_round9_D3';cache_dir.mkdir(exist_ok=True)
        cached=cache_dir/f'tree_S{shape[0]}R{shape[1]}n3_t{trump}_{fingerprint[:16]}.pkl'
        try:
            if cached.exists():
                with cached.open('rb') as f:t=pickle.load(f)
                log('D3',f'Loaded derived local tree cache for trump{trump}; code fingerprint {fingerprint}')
            else:
                t=FrozenTree(g,checkpoint=checkpoint)
                try:
                    temporary=cached.with_suffix('.tmp')
                    with temporary.open('wb') as f:pickle.dump(t,f,protocol=5)
                    replace(temporary,cached)
                    save(HERE/'results'/f'tree_cache_{trump}.json',{'derived_cache':str(cached),'source_sha256':fingerprint,'exact_deals':t.n_deals})
                except OSError as cache_error:log('D3',f'Optional local cache write failed: {cache_error}; continuing exact evaluation.')
        except RuntimeError as e:
            log('D3',str(e)+'. Incomplete chance space is not an exact population bound.');break
        assert t.n_deals==369600
        fixed,ta_stats=reachable_ta_profile(t,progress=lambda done,total:log('D3',f'Reachable TA profile {done}/{total} infosets'))
        free=np.ones(t.n_is,bool)
        lower=integer_bound(t,fixed,free);pi=float(lower.mean());pi_fraction=str(Fraction(int(lower.sum()),t.n_deals))
        root_lower=root_information_bound(t)
        baseline=float(profile_value(t,fixed)[0]);heuristic=fixed.copy()
        for p in [1,2,3]:
            _,br=best_response(t,fixed,p,return_policy=True);mask=t.is_pl==p;heuristic[mask]=br[mask]
        item={'S':shape[0],'R':shape[1],'n':3,'trump':trump,'exact_deals':t.n_deals,'nodes':t.n_nodes,'infosets':t.n_is,
              'baseline_TA':baseline,'PI_min':pi,'PI_fraction':pi_fraction,'TA_profile_support':ta_stats,
              'root_information_relaxation':root_lower,
              'independent_BR_to_TA':assess(t,heuristic,pi),'checkpoints':[],
              'uncertainty':'Full chance enumeration; no sampling CI. [PI, achieved] bounds optimization, not sample error.'}
        R=np.zeros_like(fixed);S=np.zeros_like(fixed);best=heuristic.copy();bestvalue=item['independent_BR_to_TA']['hero_score']
        bestiteration=0;bestkind='independent_BR';valid=t.is_valid
        assert float(Fraction(root_lower['first_decision_hidden_lower_fraction']))<=bestvalue+1e-7
        sig=reach=cf=V=inst=weights=avg=current=candidate=z=p=None
        item['best_feasible']={'hero_score':bestvalue,'iteration':0,'kind':bestkind,'optimal_hidden_value_interval':[pi,bestvalue],
                              'hidden_to_PI_surplus':(baseline-bestvalue)/(baseline-pi) if baseline!=pi else None}
        save(HERE/'results'/f'trump_{trump}.json',item)
        for iteration in range(1,1001):
            if time.time()>regime_stop-120:break
            z=R.sum(1,keepdims=True);sig=np.where(z>0,R/np.where(z>0,z,1),t.uniform())
            reach=reaches(t,sig);cf=cf_reach(t,reach);V=values(t,sig)
            p=t.e_pl;weights=cf[p,t.e_par]*(V[t.e_ch,p]-V[t.e_par,p])
            inst=np.bincount(t.e_ia,weights=weights,minlength=t.n_is*g.A).reshape(R.shape)
            R=np.maximum(R+inst,0)*valid
            S+=iteration*reach[t.is_pl,t.is_rep,None]*sig
            if iteration in [1,10,50,100,250,500,1000]:
                z=S.sum(1,keepdims=True);avg=np.where(z>0,S/np.where(z>0,z,1),t.uniform())
                z=R.sum(1,keepdims=True);current=np.where(z>0,R/np.where(z>0,z,1),t.uniform())
                rounded=np.zeros_like(current);rounded[np.arange(t.n_is),np.where(valid,current,-1).argmax(1)]=1
                entry={'iteration':iteration,'candidates':{}}
                for kind,candidate in [('average',avg),('current',current),('one_hot',rounded)]:
                    score=float(profile_value(t,candidate)[0]);entry['candidates'][kind]=score
                    if score<bestvalue:best=candidate.copy();bestvalue=score;bestiteration=iteration;bestkind=kind
                item['checkpoints'].append(entry)
                item['best_feasible']={'hero_score':bestvalue,'iteration':bestiteration,'kind':bestkind,'optimal_hidden_value_interval':[pi,bestvalue],
                                       'hidden_to_PI_surplus':(baseline-bestvalue)/(baseline-pi) if baseline!=pi else None}
                save(HERE/'results'/f'trump_{trump}.json',item)
                with (HERE/'results'/f'trump_{trump}_profile.tmp').open('wb') as f:np.savez_compressed(f,sigma=best,fixed=fixed,iteration=iteration)
                replace(HERE/'results'/f'trump_{trump}_profile.tmp',HERE/'results'/f'trump_{trump}_profile.npz')
                log('D3',f'trump{trump} iteration{iteration}: '+json.dumps(item['best_feasible']))
        certificate,rounded=certify(t,fixed,free,best)
        assert bestvalue>=pi-1e-7
        item['certificate']=certificate;item['best_residual']=assess(t,best,pi)
        with (HERE/'results'/f'trump_{trump}_profile.tmp').open('wb') as f:np.savez_compressed(f,sigma=best,fixed=fixed,iteration=bestiteration)
        replace(HERE/'results'/f'trump_{trump}_profile.tmp',HERE/'results'/f'trump_{trump}_profile.npz')
        save(HERE/'results'/f'trump_{trump}.json',item);results[str(trump)]=item
        del t,fixed,free,R,S,best,heuristic,lower,sig,reach,cf,V,inst,weights,avg,current,rounded,candidate,z,p,valid,br,mask
        clear_mid();gc.collect()
    save(HERE/'results/summary.json',results);cut.cancel()
    log('D3','READY FOR REVIEW\nFiles D3/frozen_tree.py, run.py, STATUS, results/*.json/*.npz; tests/test_round9_frozen_toy.py.\nCommand from engine python -m pytest -q -p no:cacheprovider tests/test_round9_frozen_toy.py; python experiments/round9/D3/run.py.\nSeeds1620M only resource preflight, exact evaluation no seed. n369600 complete ordered deals per delivered regime; no statistical95% CI applies to full-population values. Results: '+json.dumps(results)+'\nLimitations: fixed TA hero and myopic bids, play-only private-infoset spoiler team. PI/achieved bounds not a sampling interval; unilateral gaps not joint residuals. Only a successful exact support certificate establishes zero joint gap. Any missing regime or resource cap is a target shortfall, not evidence. No full-deck extrapolation.')
    status('D3','ready-for-review','Exact frozen-tree results/bounds and any time-box shortfall saved; no pooling, no full-deck inference.')
if __name__=='__main__':main()
