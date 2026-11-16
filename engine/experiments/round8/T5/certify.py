"""Exact integer support certificate for a saved hidden-team strategy.

With fixed deterministic hero/bids, PI min is a pointwise lower bound. Maximize
hero score over EVERY action in the candidate's infoset support. If this worst
support score equals PI min on every deal, the feasible hidden policy attains
the PI bound, certifying zero remaining JOINT team gain. No CFR theorem needed.
"""
from pathlib import Path
from fractions import Fraction
import json,sys,gc,os,threading,time
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from judgement.cfr.toy import ToyGame,Tree
from judgement.cfr.heuristics import profile
HERE=Path(__file__).resolve().parent

def integer_bound(t,fixed,free,support=None):
    v=t.U[:,0].astype(np.int64)
    np.testing.assert_array_equal(v,t.U[:,0])
    np.testing.assert_array_equal(fixed,fixed.astype(bool))
    forced=fixed.astype(np.int64).ravel()
    for a,b,seg,par in reversed(t.chunks):
        q=v[t.e_ch[a:b]]
        ordinary=np.add.reduceat(q*forced[t.e_ia[a:b]],seg)
        if support is None:
            informed=np.minimum.reduceat(q,seg)
        else:
            allowed=support.ravel()[t.e_ia[a:b]]
            informed=np.maximum.reduceat(np.where(allowed,q,-1000),seg)
        v[par]=np.where(free[t.node_is[par]],informed,ordinary)
    return v[t.roots]

def certify(t,fixed,free,sigma,tolerance=1e-12):
    support=(sigma>tolerance)&t.is_valid
    support[~free]=fixed[~free].astype(bool)
    assert np.all(support.sum(1)>0)
    lower=integer_bound(t,fixed,free)
    worst=integer_bound(t,fixed,free,support)
    assert np.all(worst>=lower)
    n=len(t.roots)
    np.testing.assert_allclose(t.node_c[t.roots],np.full(n,1/n),rtol=0,atol=1e-15)
    out={'certified':bool(np.array_equal(lower,worst)),'n_exact_ordered_deals':n,
         'PI_value_fraction':str(Fraction(int(lower.sum()),n)),
         'worst_support_value_fraction':str(Fraction(int(worst.sum()),n)),
         'deals_with_support_excess':int(np.count_nonzero(worst>lower)),
         'max_pointwise_excess':int((worst-lower).max()),'support_threshold':tolerance,
         'scope':'Fixed TA hero and fixed myopic bids; separate hidden infosets; play-only team.',
         'uncertainty':'Exact integer DP over the complete chance space; no sampling CI.'}
    rounded=support.astype(float)/support.sum(1)[:,None]
    rounded[~free]=fixed[~free]
    return out,rounded

def main():
    stop=datetime(2026,11,16,12,30,tzinfo=timezone.utc).timestamp()
    if time.time()>=stop:return
    lock=HERE/'worker.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    timer=threading.Timer(stop-time.time(),lambda:os._exit(0));timer.daemon=True;timer.start()
    out={}
    import judgement.cfr.heuristics as h
    for trump in [0,-1]:
        for obj in vars(h).values():
            if hasattr(obj,'cache_clear'):obj.cache_clear()
        gc.collect()
        summary=json.loads((HERE/'results'/f'trump_{trump}.json').read_text())
        checkpoint=summary['checkpoints'][-1]
        with np.load(HERE/'results'/checkpoint['profile_file']) as z:
            fixed,free,sigma=z['fixed'],z['free'],z['current_sigma']
        t=Tree(ToyGame(N=4,S=2,R=4,n=2,trump=trump))
        np.testing.assert_array_equal(free,(t.is_pl!=0)&~t.is_bid)
        np.testing.assert_array_equal(fixed,profile(t,play='ta',bid='myopic'))
        result,rounded=certify(t,fixed,free,sigma)
        result['iteration']=checkpoint['iteration'];result['source_profile']=checkpoint['profile_file']
        if result['certified']:
            target=HERE/'results'/f'certified_trump_{trump}.npz'
            temp=target.with_suffix('.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,sigma=rounded,fixed=fixed,free=free)
            temp.replace(target)
            result['joint_optimality_gap_exact']=0
        out[str(trump)]=result
        temp=HERE/'results/certificate.tmp'
        temp.write_text(json.dumps(out,indent=2));temp.replace(HERE/'results/certificate.json')
        with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:
            f.write('Support certificate: '+json.dumps(result)+'\n')
        del t,fixed,free,sigma,rounded
        gc.collect()
    print(json.dumps(out,indent=2))
    with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:
        f.write('READY FOR REVIEW — supplemental exact certificate\nFiles changed: T5/certify.py, results/certificate.json, results/certified_trump_*.npz; tests/test_round8_team.py.\nExact commands from engine: python -m pytest -q -p no:cacheprovider tests/test_round8_team.py; python experiments/round8/T5/certify.py.\nSeed base: none; complete chance space, n=2520 ordered deals per trump regime. Integer DP, no sampling CI. Results: '+json.dumps(out)+'\nLimitations: fixes all myopic bids and the TA hero; certifies the explicitly saved support-rounded hidden strategy, not a multiplayer CFR convergence theorem. No claim about a 52-card round or unrestricted bidding. Only claim zero joint gap where certified=true.\n')
    board=HERE.parent/'BOARD.md';lines=board.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| T5 |'):
            cells=line.split('|');cells[-3]=' ready-for-review ';cells[-2]=' CFR and integer support certificates complete; n2520 exact deals/regime. Check certificate.json before any joint-optimality claim. Fixed bids/TA hero only. ';lines[i]='|'.join(cells)
    board.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    timer.cancel();lock.unlink(missing_ok=True)

if __name__=='__main__':main()
