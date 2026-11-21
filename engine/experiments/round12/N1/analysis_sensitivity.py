"""Claude-predeclared finite-K sparsity diagnostics; saved-data analysis only.
This new module never modifies the approved/frozen N1 generator or models.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,statistics,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
H=Path(__file__).resolve().parent;O=H/'results';DOMAIN=0x4E31

def ci(values,groups):
    parts=[np.asarray(values,dtype=float)[groups==s] for s in sorted(set(groups))]
    assert all(len(p)>1 for p in parts)
    mean=statistics.mean([statistics.mean(v) for v in parts]);half=1.96*math.sqrt(math.fsum(statistics.variance(v)/len(v) for v in parts))/len(parts)
    return dict(mean=mean,ci95=[mean-half,mean+half],n_decisions=len(values),n_strata=len(parts),method='equal-stratum normal95%CI; independent rounds, fixed fitted ensemble')

def seed_check():
    new=np.random.SeedSequence([2110000000,DOMAIN,91]).generate_state(4).tobytes();count=0
    def check(a):
        nonlocal count
        assert np.random.SeedSequence(a).generate_state(4).tobytes()!=new;count+=1
    for g in range(6000):check([2100000000,g])
    for g in range(600):check([2100000000,10000000+g])
    for stage,base,n in ((0,2100000000,176),(1,2100000000,30800),(2,2100000000,4400),(3,2110000000,3520)):
        for g in range(n):
            for tail in (3,4,5):check([base,DOMAIN,stage,g,tail])
    for seed in range(5):
        check([2100000000,DOMAIN,20,seed])
        for epoch in range(1,51):check([2100000000,DOMAIN,21,seed,epoch])
    check([2110000000,DOMAIN,90]);assert count==123544
    return dict(address=[2110000000,DOMAIN,91],checked_prior_roots=count,collisions=0,scope='bounded known N0/N1 roots; prior distinct-root audit preserved')

def main():
    assert (O/'summary.json').exists(),'Run only after fixed confirmation complete'
    seed=seed_check();data={k:[] for k in ('P','Y','uniform','bi')};offset=[0];records=[];hashes={}
    for start in range(0,3520,64):
        p=O/'confirmation'/f'{start:05d}';receipt=json.loads(p.with_suffix('.json').read_text())
        for suffix,field in (('.npz','arrays_sha256'),('.jsonl','records_sha256')):
            got=hashlib.sha256(p.with_suffix(suffix).read_bytes()).hexdigest();assert got==receipt[field];hashes[str(p.with_suffix(suffix).relative_to(H))]=got
        a=np.load(p.with_suffix('.npz'))
        for k in data:data[k].append(a[k])
        offset.extend((a['offset'][1:]+offset[-1]).tolist())
        records.extend(json.loads(s) for s in p.with_suffix('.jsonl').read_text().splitlines())
    data={k:np.concatenate(v) for k,v in data.items()};offset=np.array(offset);n=3520;assert len(records)==n
    assert [r['g'] for r in records]==list(range(n));groups=np.arange(n)%176
    stored=np.load(O/'confirmation_predictions.npz');net=stored['net_full'];y=data['Y'];prop=data['P'];idx=np.arange(len(y))
    def round_mean(v):return np.array([math.fsum(v[a:b])/int(b-a) for a,b in zip(offset[:-1],offset[1:])])
    def ll(p):return round_mean(-np.log(p[idx,y]))
    netloss=ll(net);out={'seed_collision_check':seed,'sensitivity':{},'zero_count':{},'BI_by_n':{},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'raw_hashes':hashes,
        'primary_floor_effective_pseudocount':256*.001/.999,'n_confirmation_rounds':3520,'n_fit_seeds':5,'n_scored_cards':len(y),'scope':'pre-data Claude secondary sensitivity; primary unchanged; round intervals conditional on selected fixed ensemble; not a posterior proof'}
    for method in ('bi','uniform'):
        fraction=data[method];counts=np.rint(256*fraction).astype(int);np.testing.assert_allclose(counts/256,fraction,rtol=0,atol=1e-12);assert np.all(counts.sum(1)==256)
        base=.999*fraction+.001*prop;baseline=ll(base)
        np.testing.assert_allclose(base,stored[method],rtol=0,atol=1e-12)
        out['sensitivity'][method]={'declared_floor':{'net_minus_sampler':ci(netloss-baseline,groups),'sampler_loss':ci(baseline,groups)}}
        for a in (1,4,16):
            smooth=(counts+a*prop)/(256+a);values=ll(smooth)
            out['sensitivity'][method][str(a)]={'net_minus_sampler':ci(netloss-values,groups),'sampler_loss':ci(values,groups)}
        zeros=(counts[idx,y]==0);fractions=round_mean(zeros);num=np.array([int(zeros[a:b].sum()) for a,b in zip(offset[:-1],offset[1:])]);den=np.diff(offset);ratio=float(num.sum()/den.sum())
        influence=(num-ratio*den)/den.mean();effect=ci(influence,groups);half=(effect['ci95'][1]-effect['ci95'][0])/2
        out['zero_count'][method]={'round_equal_fraction':ci(fractions,groups),'pooled_card_fraction':{'mean':ratio,'ci95':[max(0,ratio-half),min(1,ratio+half)],'n_round_clusters':n,'n_cards':len(y),'method':'equal-stratum paired delta-method ratio95%CI'},'n_zero_card_labels':int(zeros.sum())}
    # Paired primary recomputation: distinct CI path from run.py.
    primary=out['sensitivity']['bi']['declared_floor']['net_minus_sampler'];original=json.loads((O/'summary.json').read_text())['primary_net_minus_BI'];np.testing.assert_allclose(primary['ci95'],original['ci95'],rtol=0,atol=1e-11)
    rng=np.random.default_rng([2110000000,DOMAIN,91])
    for size in range(3,14):
        ids=np.array([i for i,r in enumerate(records) if r['n']==size]);g=groups[ids];assert len(ids)==320
        local=[np.flatnonzero(g==s) for s in sorted(set(g))];boot=np.array([np.concatenate([rng.choice(q,len(q),replace=True) for q in local]) for _ in range(2000)])
        stats={}
        for field in ('ess','M'):
            v=np.array([records[i]['sampler'][field] for i in ids]);quantiles=np.quantile(v,[.25,.5,.75]);draws=np.quantile(v[boot],[.25,.5,.75],axis=1)
            stats[field]={'mean':ci(v,g),'observed_sample_range_exact':[float(v.min()),float(v.max())],'quartiles':[{'quantile':q,'value':float(z),'ci95':np.quantile(draws[k],[.025,.975]).tolist(),'n_rounds':320,'bootstrap_reps':2000} for k,(q,z) in enumerate(zip((.25,.5,.75),quantiles))]}
        out['BI_by_n'][str(size)]=stats
    # Replay the already-declared ECE bootstrap stream90, with the same draws
    # paired across methods. No new data, model, seed stage or hypothesis.
    strata=[np.arange(s,n,176) for s in range(176)];rng=np.random.default_rng([2110000000,DOMAIN,90])
    draws=[np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]) for _ in range(2000)]
    reliability={}
    for method in ('prop','uniform','bi','net_bids','net_full'):
        probability=stored[method];bins=[];raw_bin_counts=np.zeros(10,np.int64)
        for a,b in zip(offset[:-1],offset[1:]):
            pp=probability[a:b];confidence=pp.max(1);correct=pp.argmax(1)==y[a:b];ix=np.minimum((10*confidence).astype(int),9)
            count=np.bincount(ix,minlength=10);raw_bin_counts+=count
            bins.append(np.stack([count,np.bincount(ix,weights=confidence,minlength=10),np.bincount(ix,weights=correct,minlength=10)])/(b-a))
        bins=np.array(bins);aggregate=bins.mean(0);sample=np.array([bins[d].mean(0) for d in draws]);entries=[]
        for k in range(10):
            support=int((bins[:,0,k]>0).sum());valid=sample[:,0,k]>0
            entry={'bin':[k/10,(k+1)/10],'n_round_clusters':n,'n_rounds_with_cards_in_bin':support,'n_raw_cards':int(raw_bin_counts[k]),'round_weighted_mass':ci(bins[:,0,k],groups),'n_nonempty_bootstrap_draws':int(valid.sum())}
            for axis,name in ((1,'predicted_confidence'),(2,'observed_accuracy')):
                entry[name]={'mean':float(aggregate[axis,k]/aggregate[0,k]) if aggregate[0,k] else None,
                    'ci95':np.quantile(sample[valid,axis,k]/sample[valid,0,k],[.025,.975]).tolist() if valid.any() else None}
            entry['scope']='Paired176-stratum round bootstrap,2000draws; marginal not simultaneous; sparse bins exploratory and intervals conditional on nonempty resamples.'
            entries.append(entry)
        ece=float(abs(aggregate[2]-aggregate[1]).sum());ece_draws=abs(sample[:,2,:]-sample[:,1,:]).sum(1)
        original=json.loads((O/'summary.json').read_text())['calibration'][method]
        assert abs(ece-original['ECE'])<1e-12
        np.testing.assert_allclose(np.quantile(ece_draws,[.025,.975]),original['ci95'],rtol=0,atol=1e-12)
        reliability[method]={'ECE':ece,'ci95':original['ci95'],'n_rounds':n,'bins':entries}
    out['reliability_with_CIs']=reliability
    out['utc']=datetime.now(timezone.utc).isoformat();path=O/'sparsity_diagnostics.json';path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print('Saved',path,'nrounds',n,'nlabels',len(y))
if __name__=='__main__':main()
