"""Non-opening hero dependence; independent discovery and replication streams."""
from pathlib import Path
import sys, json, threading, os, time
from datetime import datetime, timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'math')]
from judgement import Rules, RoundState
from judgement.engine import deal, run_round
from judgement.state import state_from_info
from judgement.sampling import HandSampler
from m5_sim import make_agents
from m5_analyze import pb_pmf, hist, bid
from m10_analyze import chain_pmf
HERE=Path(__file__).resolve().parent

def best_legal_bid(pmf,legal):
    ev=(10+np.arange(len(pmf)))*pmf-1e-12*np.arange(len(pmf))
    return int(max(legal,key=lambda b:ev[b]))

def models(X,W):
    n=W.shape[1];pj=W.mean(0);L=W[:,:-1];Y=W[:,1:]
    a=[(Y[L==l].sum()+2*W.mean())/((L==l).sum()+2) for l in (0,1)]
    return {'PB':pb_pmf(X.mean(0)),'TPB':pb_pmf(pj),
            'LC0':chain_pmf(lambda j,s,t:pj[0] if j==0 else a[s[-1]],n)}

def metrics(cards,cm,tm,n,legal=None):
    X=((cm[:,None]>>np.arange(n))&1).astype(float)
    W=((tm[:,None]>>np.arange(n))&1).astype(float)
    p=X.mean(0);vpb=(p*(1-p)).sum()
    if vpb<1e-9:return None
    C=X.T@X/len(X)-np.outer(p,p)
    cross=(np.asarray(cards)//13)[:,None]!=(np.asarray(cards)//13)[None,:]
    same=(~cross)&(~np.eye(n,dtype=bool))
    out={'delta':float(X.sum(1).var()/vpb),'cross_suit_dispersion':float(C[cross].sum()/vpb),
         'same_suit_dispersion':float(C[same].sum()/vpb)}
    assert abs(out['delta']-1-out['cross_suit_dispersion']-out['same_suit_dispersion'])<1e-8
    Y=W[:,1:];L=W[:,:-1]
    out['persistence']=float(Y[L==1].mean()-Y[L==0].mean()) if (L==1).any() and (L==0).any() else None
    legal=list(range(n+1)) if legal is None else list(legal)
    def choose(pmf):
        return best_legal_bid(pmf,legal)
    R=len(X)//2;target=choose(hist(W[R:].sum(1).astype(int),n))
    out['floor_match']=float(choose(hist(W[:R].sum(1).astype(int),n))==target)
    for name,pmf in models(X[:R],W[:R]).items():out[name+'_match']=float(choose(pmf)==target)
    return out

def main():
    (HERE/'results').mkdir(parents=True,exist_ok=True)
    stop=datetime(2026,11,16,12,30,tzinfo=timezone.utc).timestamp()
    if stop<=time.time():return
    lock=HERE/'worker.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        return  # Another compute slot already owns this task.
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    timer=threading.Timer(stop-time.time(),lambda:os._exit(0));timer.daemon=True;timer.start()
    rules=Rules();agents=make_agents()['ta']
    for base in [1_100_000_000,1_110_000_000]:
      for n in [5,8]:
       for pos in [1,2,3]:
        hero=(pos+1)%4
        for i in range(500):
            path=HERE/'results'/f'{base}_{n}_{pos}_{i:04d}.npz'
            if path.exists():continue
            rng=np.random.default_rng([base,n,pos,i]);hands=deal(rules,n,rng)
            st=RoundState(rules,n,-1,0,hands)
            while st.to_act!=hero:
                st.apply_bid(agents[st.to_act].bid(st.info_state(st.to_act),st.legal_bids(),rng))
            legal=st.legal_bids()
            st.apply_bid(agents[hero].bid(st.info_state(hero),legal,rng))
            info=st.info_state(hero);sampler=HandSampler(info)
            cards=[c for c in range(52) if hands[hero]>>c&1];indices={c:j for j,c in enumerate(cards)}
            cm=np.zeros(300,np.int32);tm=np.zeros(300,np.int32)
            for r in range(300):
                world=state_from_info(info,sampler.sample(rng));run_round(world,agents,[rng]*4,check=False)
                for j,w in enumerate(world.trick_winners):
                    if w==hero:
                        tm[r]|=1<<j
                        for p,c in world.history[4*j:4*j+4]:
                            if p==hero:cm[r]|=1<<indices[c]
            tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,cards=cards,cm=cm,tm=tm,base=base,n=n,pos=pos,i=i,bids=st.bids,legal=legal)
            tmp.replace(path)
            if (i+1)%50==0:
                with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:f.write(f'{datetime.now(timezone.utc).isoformat()} base={base}, n={n}, pos={pos}: {i+1}/500 hands; R=300.\n')
    timer.cancel()
    lock.unlink(missing_ok=True)
    with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:f.write('COMPUTE COMPLETE. Run analyze.py; no interpretive claim before replication and review.\n')

if __name__=='__main__':main()
