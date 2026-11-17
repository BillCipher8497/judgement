"""Integer replay of previously evaluated support-rounded saved candidates.

No training/new action search. Establish exact feasible upper bounds in
addition to the separate audit's rational lower bounds.
"""
import gc,hashlib,json,pickle,sys
from fractions import Fraction
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;D3=HERE.parent/'D3';ENGINE=HERE.parents[2]
sys.path[:0]=[str(ENGINE),str(D3)]
out={}
for trump in [0,-1]:
    meta=json.loads((D3/'results'/f'tree_cache_{trump}.json').read_text())
    with Path(meta['derived_cache']).open('rb') as f:t=pickle.load(f)
    filename='trump_0_optimized_profile.npz' if trump==0 else 'trump_-1_profile.npz'
    with np.load(D3/'results'/filename) as z:
        actions=np.where(t.is_valid,z['sigma'],-1).argmax(1).astype(np.int8)
    assert np.all(t.is_valid[np.arange(t.n_is),actions])
    v=t.U[:,0].astype(np.int64);lengths=t.node_len[t.e_par]
    cuts=np.r_[0,np.flatnonzero(np.diff(lengths))+1,t.n_edges]
    for a,b in reversed(list(zip(cuts[:-1],cuts[1:]))):
        parents=t.e_par[a:b];starts=np.r_[0,np.flatnonzero(np.diff(parents))+1]
        selected=actions[t.e_is[a:b]]==t.e_a[a:b]
        assert np.all(np.add.reduceat(selected.astype(np.int8),starts)==1)
        v[parents[starts]]=np.add.reduceat(v[t.e_ch[a:b]]*selected,starts)
    total=int(v[t.roots].sum());fraction=Fraction(total,t.n_deals)
    # These candidates were already evaluated in the timebox, not selected now.
    rawname='trump_0_optimization.json' if trump==0 else 'trump_-1.json'
    raw=json.loads((D3/'results'/rawname).read_text())
    entry=next(c for c in raw['checkpoints'] if c['iteration']==50)
    original=entry['candidate_scores']['one_hot'] if trump==0 else entry['candidates']['one_hot']
    assert abs(float(fraction)-original)<1e-8
    dest=D3/'results'/f'review_pure_trump_{trump}.npz'
    np.savez_compressed(dest,actions=actions)
    out[str(trump)]={'n_exact_deals':t.n_deals,'exact_feasible_upper_fraction':str(fraction),
        'hero_score_sum':total,'policy_file':dest.name,'policy_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
        'original_one_hot_checkpoint':50,'original_one_hot_score':original,
        'source_profile_sha256':hashlib.sha256((D3/'results'/filename).read_bytes()).hexdigest(),
        'uncertainty':'Integer replay over complete chance space; no sampling CI; feasible upper bound, not team optimum.'}
    (HERE/'results/D3_exact_upper_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({str(trump):out[str(trump)]}),flush=True)
    del t,v,actions,selected,parents,starts,lengths,cuts;gc.collect()
