"""Independent N2 raw grouping, model selection, direct posterior/KL audit.
Prototype, not executed. New conditional proof notes remain pending external.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,math,statistics,sys,time,hashlib
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;sys.path[:0]=[str(H),str(H.parent),str(H.parents[2]),str(H.parents[2]/'experiments/round9/D3')]
from judgement.cfr.toy import ToyGame
from judgement.cfr.heuristics import myopic_bid
from bounded_ta import ta_card,clear_mid
from corpus import Corpus
from export import decode,digest
from reference import direct_posterior,deal_index,public,legal,independent_deals
from bid_premise import manual_hook_argmax,assert_matched
O=H/'results'

def forward(x,mask,prop,p):
    hidden=x.reshape(-1,35)@p[0]+p[1];hidden[hidden<0]=0
    z=(hidden@p[2]+p[3]).reshape(-1,4);z+=np.log(np.maximum(prop,1e-300));z[~mask]=-np.inf
    z-=np.max(z,axis=1,keepdims=True);prob=np.exp(z);prob/=prob.sum(1,keepdims=True);return prob

def model(path):
    a=np.load(path);return [a[f'p{k}'] for k in range(4)]

def risk(data,prob):
    losses=-np.sum(data['T']*np.log(np.maximum(prob,1e-300)),axis=1);parts=[]
    for a,b,c in zip(data['offset'][:-1],data['offset'][1:],data['count']):parts.append(float(c)*math.fsum(losses[a:b])/int(b-a) if b>a else 0.)
    return math.fsum(parts)/float(data['count'].sum())

def tinterval(v):
    mean=statistics.mean(v);half=2.7764451051977987*statistics.stdev(v)/math.sqrt(5);return [mean-half,mean+half]

def every_world_matched_bid_premise(g,corpus,deadline):
    generator=independent_deals(g);world_count=0;key_world_count=0
    for path in sorted((corpus.folder/'raw').glob('*.npz')):
        a=np.load(path);r=json.loads(path.with_suffix('.json').read_text())
        for j in range(r['n_deals']):
            assert time.time()<deadline;hands=next(generator);bids=[]
            for p in range(4):bids.append(manual_hook_argmax(g,p,hands[p],tuple(bids)))
            assert_matched(g,hands,tuple(bids));prefix=tuple(bids)
            for event in range(4*g.n):
                index=j*4*g.n+event;seat,hand,hist=decode(a['keys'][index])
                assert tuple(hand)==hands[seat] and hist==prefix and int(a['deal_index'][index])==world_count
                played,voids,actor=public(g,hist);assert actor==seat
                tl=(len(hist)-4)%4;trick=[((seat-tl+k)%4,c) for k,c in enumerate(hist[-tl:])] if tl else []
                left=[c for c in hands[seat] if c not in hist[4:]];chosen=int(a['chosen_card'][index]);assert chosen in legal(g,left,trick)
                labels=0;opps=[(seat+k)%4 for k in (1,2,3)]
                for c in range(g.S*g.R):
                    if c not in hand and c not in hist[4:]:labels|=next(k for k,p in enumerate(opps) if c in hands[p])<<(2*c)
                assert labels==int(a['holder_bits'][index]);prefix+=(chosen,);key_world_count+=1
            world_count+=1
    assert next(generator,None) is None
    return {'n_exact_chance_deals':world_count,'n_recorded_key_worlds':key_world_count,'R_subset_F_bid_every_exported_key':True,'all_hook_legal_argmax_and_true_holders':True,'independent_manual_turn_follow_suit_and_prefix_chaining':True}

def raw_grouping(g,corpus):
    d=corpus.data;n=len(corpus.keys);count=np.zeros(n,np.int64);holders=np.zeros_like(d['holder_counts']);groups=[];deals=[];low=np.full(n,255,np.uint8);high=np.zeros(n,np.uint8)
    unknown=[]
    for k in corpus.keys:
        p,h,hist=decode(k);played,voids,actor=public(g,hist);assert actor==p and set(played[p])<=set(h)
        unknown.append(sum(1<<c for c in range(g.S*g.R) if c not in h and c not in hist[4:]))
    unknown=np.array(unknown,np.uint16);np.testing.assert_array_equal(unknown,d['unseen_mask'])
    for path in sorted((corpus.folder/'raw').glob('*.npz')):
        a=np.load(path);r=json.loads(path.with_suffix('.json').read_text());assert digest(path)==r['raw_sha256'];idx=np.searchsorted(corpus.keys,a['keys']);np.testing.assert_array_equal(corpus.keys[idx],a['keys'])
        count+=np.bincount(idx,minlength=n);groups.append(idx.astype(np.uint32));deals.append(a['deal_index']);np.minimum.at(low,idx,a['chosen_card']);np.maximum.at(high,idx,a['chosen_card'])
        for c in range(g.S*g.R):
            take=((unknown[idx]>>c)&1).astype(bool);truth=(a['holder_bits'][take]>>(2*c))&3;assert np.all(truth<3);np.add.at(holders[:,c,:],(idx[take],truth),1)
    np.testing.assert_array_equal(count,d['reach_count']);np.testing.assert_array_equal(holders,d['holder_counts']);np.testing.assert_array_equal(low,high)
    # Same private key must prescribe one legal action in every reached world.
    for i,k in enumerate(corpus.keys):
        p,h,hist=decode(k);tl=(len(hist)-4)%4;trick=[((p-tl+j)%4,c) for j,c in enumerate(hist[-tl:])] if tl else []
        left=[c for c in h if c not in hist[4:]];assert int(low[i]) in legal(g,left,trick)
    groups=np.concatenate(groups);world=np.concatenate(deals);order=np.lexsort((world,groups));ordered_g=groups[order];ordered_w=world[order]
    assert not np.any((ordered_g[1:]==ordered_g[:-1])&(ordered_w[1:]==ordered_w[:-1]))
    np.testing.assert_array_equal(ordered_w,d['support_deal_index']);np.testing.assert_array_equal(np.r_[0,np.cumsum(count)],d['support_offset'])
    return {'all_raw_records_checked':int(count.sum()),'all_private_key_actions_consistent':True,'all_counts_labels_and_CSR_match':True}

def selected_models(corpus):
    f=corpus.folder/'models';packet=json.loads((f/'frozen_models.json').read_text());keys=np.load(f/'fitting_keys.npz');np.testing.assert_array_equal(keys['train'],corpus.selected(0,30000));np.testing.assert_array_equal(keys['validation'],corpus.selected(1,4000));val=corpus.pack(keys['validation']);parameters=[];alphas=[]
    for seed,r in enumerate(packet['models']):
        folder=f/f's{seed}';choices=[];traces=[json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()];assert [x['epoch'] for x in traces]==list(range(1,21))
        for epoch in (5,10,20):
            p=model(folder/f'candidate_{epoch}.npz');pred=forward(val['X'],val['M'],val['P'],p);loss=risk(val,.999*pred+.001*val['P']);assert abs(loss-next(x['selection_marginal_loss'] for x in traces if x['epoch']==epoch))<1e-11;choices.append((loss,epoch))
        expected=min(choices);assert r['selected_epoch']==expected[1]
        p=model(folder/'selected.npz');candidate=model(folder/f'candidate_{r["selected_epoch"]}.npz')
        for aa,bb in zip(p,candidate):np.testing.assert_array_equal(aa,bb)
        assert digest(folder/'selected.npz')==r['selected_model_sha256'];net=forward(val['X'],val['M'],val['P'],p)
        trials=[(risk(val,(1-.999*a)*val['P']+.999*a*net),a) for a in (0.,.25,.5,1.)];assert min(trials)[1]==r['alpha'];parameters.append(p);alphas.append(r['alpha'])
    return parameters,alphas

def direct_checks(g,corpus,parameters,alphas,deadline):
    n=len(corpus.keys);selected=range(n) if g.n==2 else sorted(set(np.linspace(0,n-1,64,dtype=int).tolist()));checked=0
    for i in selected:
        assert time.time()<deadline;p,h,hist=decode(corpus.keys[i]);worlds,reached=direct_posterior(g,p,h,hist,myopic_bid,ta_card)
        ids=sorted(deal_index(g,w) for w in reached);a,b=corpus.data['support_offset'][i:i+2];np.testing.assert_array_equal(ids,corpus.data['support_deal_index'][a:b])
        chunk=corpus.folder/'joint_KL'/f'{(i//1024)*1024:07d}.npz';saved=np.load(chunk);row=i%1024;opps=[(p+k)%4 for k in (1,2,3)];counts=[g.n+1-int(q==3 and 0<=g.n-sum(hist[:3])<=g.n and g.hook) for q in opps]
        weights=[];hard=0
        for w in worlds:
            match=[myopic_bid(g,q,w[q],tuple(hist[:q]))==hist[q] for q in opps];hard+=int(all(match));weights.append(math.prod(49*c*int(m)+1 for c,m in zip(counts,match)))
        Z=sum(weights);maxw=math.prod(49*c+1 for c in counts);np.testing.assert_array_equal(saved['world_counts'][row],[len(worlds),hard,len(reached)]);np.testing.assert_array_equal(saved['BI_integer_partition_and_maxweight'][row],[Z,maxw])
        x,m,prop,_,obs=corpus.row(i);cards=obs['cards'];net=[]
        for params,alpha in zip(parameters,alphas):
            pp=(1-.999*alpha)*prop+.999*alpha*forward(x,m,prop,params);logs=[];reached_logs=[];reach_set=set(ids)
            for w in worlds:
                value=math.fsum(math.log(pp[j,next(k for k,q in enumerate(opps) if c in w[q])]) for j,c in enumerate(cards));logs.append(value)
                if deal_index(g,w) in reach_set:reached_logs.append(value)
            maximum=max(logs);logz=maximum+math.log(math.fsum(math.exp(v-maximum) for v in logs));value=logz-statistics.mean(reached_logs)-math.log(len(reached));net.append(value)
        np.testing.assert_allclose(net,saved['raw_net_KL'][row],rtol=0,atol=1e-10);checked+=1
    return {'direct_normalizer_and_full_reach_contexts':checked,'selection':'allsmallinfosets;64fixedsortedlargekeys','independent_rules':True,'policy_functions_reused_as_definition':True}

def summaries(corpus):
    value=json.loads((corpus.folder/'joint_summary.json').read_text());parts={s:[] for s in ('all','train','validation','test')};marginal={s:[] for s in parts}
    for path in sorted((corpus.folder/'joint_KL').glob('*.npz')):
        a=np.load(path);r=json.loads(path.with_suffix('.json').read_text());assert digest(path)==r['arrays_sha256']
        # Exact integer baseline reconstruction, independent of DP floats.
        w=a['world_counts'];b=a['BI_integer_partition_and_maxweight'];recomputed=np.column_stack([np.log(w[:,0]/w[:,2]),np.log(b[:,0]/(w[:,2]*b[:,1])),np.log(w[:,1]/w[:,2])]);np.testing.assert_allclose(recomputed,a['baseline_KL'],rtol=0,atol=1e-11)
        for name,split in (('all',None),('train',0),('validation',1),('test',2)):
            take=np.ones(len(w),bool) if split is None else a['split']==split
            parts[name].append((a['reach'][take].astype(float),a['net_KL'][take],recomputed[take]))
            marginal[name].append((a['reach'][take].astype(float),a['raw_marginal_card_logloss'][take]))
    for name,rows in parts.items():
        den=math.fsum(float(w.sum()) for w,n,b in rows);net=[math.fsum(float((w*n[:,s]).sum()) for w,n,b in rows)/den for s in range(5)];baseline=[math.fsum(float((w*b[:,s]).sum()) for w,n,b in rows)/den for s in range(3)]
        q=value['summaries'][name];assert den==q['exact_reach_weight'];np.testing.assert_allclose(net,q['net_KL_by_seed_exact_population_float'],rtol=0,atol=1e-10);np.testing.assert_allclose(tinterval([v-baseline[1] for v in net]),q['primary_net_minus_BI']['ci95'],rtol=0,atol=1e-10)
        card=[math.fsum(float((w*c[:,j]).sum()) for w,c in marginal[name])/den for j in range(6)]
        np.testing.assert_allclose(card[1:],q['raw_marginal_card_logloss']['net_by_seed'],rtol=0,atol=1e-10)
        assert abs(card[0]-q['raw_marginal_card_logloss']['capacity_prop'])<1e-10
        if baseline[0]>0:
            shares=q['share_uniform_KL_removed'];np.testing.assert_allclose([1-v/baseline[0] for v in net],shares['net_by_seed'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(tinterval([1-v/baseline[0] for v in net]),shares['net_seed_summary']['ci95'],rtol=0,atol=1e-10)
    return True

def main():
    deadline=float(os.environ['JUDGEMENT_N2_AUDIT_DEADLINE']);freeze=json.loads((O/'source_freeze.json').read_text())
    for rel,h in freeze['raw_sources'].items():assert digest(H.parents[3]/rel)==h
    results={}
    for S,n,tr in ((2,2,0),(2,2,-1),(3,3,0),(3,3,-1)):
        assert time.time()<deadline;tag=f'S{S}R4n{n}_t{tr}';g=ToyGame(N=4,S=S,R=4,n=n,trump=tr);corpus=Corpus(g,O/tag)
        premise=every_world_matched_bid_premise(g,corpus,deadline);raw=raw_grouping(g,corpus);parameters,alphas=selected_models(corpus);direct=direct_checks(g,corpus,parameters,alphas,deadline);assert summaries(corpus);clear_mid();results[tag]={'matched_bid_premise':premise,'raw':raw,'models_checked':5,'direct':direct,'summary_CIs_recomputed':True}
        (O/'separate_verification_progress.json').write_text(json.dumps({'conditions':results,'status':'partial verification, not external validation'},indent=2))
    report={'conditions':results,'status':'self-verified(Codex), implementation/results pending external audit; M16 conditional proof externally checked17Nov','verifier_sha256':digest(__file__),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'limitations':'Every raw grouping/label/action consistency/model selection/summary checked. Fresh policy-conditioned posterior direct enumeration covers all small keys and64large contexts, not every large posterior; existing private policy functions reused, rules/normalizers independent.'}
    (O/'separate_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
