"""Separate cache/feature/DP-normalizer/forward/selection/CI audit. No fitting."""
from n9lib import *
import math
from itertools import combinations
from joint import log_partition
from reference import public as replay_public,direct_posterior,reaches
from judgement.cfr.heuristics import myopic_bid
from bounded_ta import ta_card,clear_mid
def independent_score(r,p,isrich):
    a=r.x.copy()
    if not isrich:a[:,:,35:]=0
    for i in range(0,len(p),2):
        a=np.einsum('...j,jk->...k',a,p[i])+p[i+1]
        if i+2<len(p):a=np.maximum(a,0)
    return np.where(r.mask,a.reshape(-1,4)+np.log(np.maximum(r.prop,1e-300)),-np.inf)
def context_support(g,r):
    seat,own,hist=decode(r.key);played,voids,p=replay_public(g,hist);assert p==seat;cards=tuple(c for c in g.deck if c not in own and c not in hist[4:]);opps=[(seat+k)%4 for k in (1,2,3)];caps=[g.n-len(played[p]) for p in opps];allowed=np.array([[c//4 not in voids[p] and caps[j]>0 for j,p in enumerate(opps)] for c in cards]);assert len(cards)==len(r.x)
    np.testing.assert_array_equal(r.mask,np.c_[allowed,np.zeros(len(cards),bool)])
    prop=np.c_[allowed*np.array(caps),np.zeros(len(cards))];prop/=prop.sum(1,keepdims=True);np.testing.assert_allclose(prop,r.prop,rtol=0,atol=1e-15)
    # Reconstruct the new rich fields independently, retaining the unchanged,
    # previously audited N2summary encoder for the first35columns.
    obs=observation(g,r.key);oldx,_,_,_=encode(g,obs);ctx=[int(c in own) for c in range(12)]
    ctx += [int(b==j) for b in hist[:4] for j in range(4)]
    ctx += [int((hist[4+k] if 4+k<len(hist) else 12)==j) for k in range(12) for j in range(13)]
    ctx += [int(seat==j) for j in range(4)]+[int((3 if g.trump==-1 else g.trump)==j) for j in range(4)]+[g.n/3,g.S/3]
    expected=np.zeros_like(r.x);expected[:,:,:35]=oldx;expected[:,:,35:229]=ctx
    for i,c in enumerate(cards):
        expected[i,:,229:241]=np.eye(12)[c]
        for j,p in enumerate(opps):expected[i,j,241:245]=np.eye(4)[p]
    np.testing.assert_array_equal(expected,r.x)
    return seat,own,hist,cards,opps,caps,allowed
def independent_kl(r,z,caps,allowed):
    def handweight(j,mask):return math.fsum(float(z[i,j]) for i in range(len(z)) if mask>>i&1)
    logz=log_partition(caps,allowed,handweight);expect=math.fsum(float(z[i,int(a)]) for owners in r.F[r.posterior] for i,a in enumerate(owners))/r.count
    return float(logz-expect-math.log(r.count))
def interval(v):
    m=math.fsum(v)/5;h=2.7764451051977987*math.sqrt(math.fsum((a-m)**2 for a in v)/20);return m,[m-h,m+h]
def main():
    O.mkdir(exist_ok=True);complete=json.loads((O/'training_complete.json').read_text());evaluation=json.loads((O/'evaluation.json').read_text());assert len(evaluation['rows'])==240
    sources=json.loads((O/'sources.json').read_text());assert all(sha(Path(p))==v for p,v in sources.items())
    adapter=json.loads((O/'api_adapter.json').read_text());assert all(sha(Path(p))==v for p,v in adapter['sources'].items())
    checked=0;cachechecks=[];near_ties=0
    for tag in TAGS:
        check();g=game(tag);rs=rows(load(O/tag/'data.npz'));oldfolder=R/'N2/results'/tag;manifest=json.loads((oldfolder/'posterior_manifest.json').read_text());assert sha(oldfolder/'posterior_supports.npz')==manifest['posterior_sha256'];raw=load(oldfolder/'posterior_supports.npz');worlds=list(independent_deals(g));contexts=[];part=hand_partitions(g);handsets={s:set() for s in range(3)}
        # Independently reconstruct the declared hand-balanced SHA priority
        # selector, not merely the split labels on the already selected keys.
        packed=np.frombuffer(raw['keys'].tobytes(),np.uint8).reshape(-1,20);masks=packed[:,2].astype(int)+256*packed[:,3];eligible=np.array([i.bit_count()>=2 for i in range(4096)])[raw['unseen_mask']];expected_ids=[];expected_splits=[]
        for split,cap in ((0,2048),(1,512),(2,1024)):
            pools=[]
            for own in sorted((h for h,s in part.items() if s==split),key=lambda h:sum(1<<c for c in h)):
                hm=sum(1<<c for c in own);pool=np.flatnonzero(eligible&(masks==hm)).tolist()
                def priority(i):
                    msg=[2170000000,0x4e39,g.S,g.n,g.trump,split,bytes(raw['keys'][i]).ljust(20,b'\0').hex()]
                    return int.from_bytes(hashlib.sha256(json.dumps(msg,separators=(',',':')).encode()).digest(),'big'),i
                pools.append(heapq.nsmallest(cap,pool,key=priority))
            take=[]
            for index in range(max(map(len,pools))):
                for pool in pools:
                    if index<len(pool):take.append(pool[index])
                    if len(take)>=cap:break
                if len(take)>=cap:break
            expected_ids.extend(take);expected_splits.extend([split]*len(take))
        assert [r.kid for r in rs]==expected_ids and [r.split for r in rs]==expected_splits
        sample=set(np.linspace(0,len(rs)-1,32,dtype=int).tolist());original_chunks={}
        for j,r in enumerate(rs):
            check();seat,own,hist,cards,opps,caps,allowed=context_support(g,r);contexts.append((caps,allowed));assert part[own]==r.split;handsets[r.split].add(own)
            assert bytes(raw['keys'][r.kid])==bytes(r.key) and int(raw['reach_count'][r.kid])==r.count
            a,b=map(int,raw['support_offset'][r.kid:r.kid+2]);np.testing.assert_array_equal(raw['support_deal_index'][a:b],r.world_ids)
            fresh=list(structural_worlds(g,seat,own,hist));fowners={tuple(next(k for k,p in enumerate(opps) if c in w[p]) for c in cards) for w in fresh};assert fowners==set(map(tuple,r.F))
            posterior=[]
            for wid in r.world_ids:
                w=worlds[int(wid)];assert w[seat]==own;posterior.append(tuple(next(k for k,p in enumerate(opps) if c in w[p]) for c in cards))
            assert posterior==[tuple(a) for a in r.F[r.posterior]] and len(set(posterior))==r.count
            target=np.zeros_like(r.target)
            for owners in posterior:target[np.arange(len(cards)),owners]+=1/r.count
            np.testing.assert_allclose(target,r.target,rtol=0,atol=1e-12)
            if g.n==2 or j in sample:
                reached=[w for w in fresh if reaches(g,w,seat,hist,myopic_bid,ta_card)];assert {deal_index(g,w) for w in reached}==set(r.world_ids.tolist());clear_mid()
            if r.split==2:
                start=r.kid//1024*1024
                if start not in original_chunks:
                    with np.load(oldfolder/'joint_KL'/f'{start:07d}.npz') as z:original_chunks[start]=(z['group_index'],z['baseline_KL'])
                ids,base=original_chunks[start];assert ids[r.kid-start]==r.kid and abs(float(base[r.kid-start,1])-r.BI)<1e-10
        assert not(handsets[0]&handsets[1] or handsets[0]&handsets[2] or handsets[1]&handsets[2]);testids=[j for j,r in enumerate(rs) if r.split==2];valids=[j for j,r in enumerate(rs) if r.split==1]
        for cfg in CONFIGS:
            for loss in LOSSES:
                for seed in range(5):
                    check();folder=O/tag/'models'/f'{cfg}_{loss}_s{seed}';f=json.loads((folder/'frozen.json').read_text());assert sha(folder/'frozen.json')==complete['receipts'][str((folder/'frozen.json').relative_to(O))]
                    history=[json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()];recorded={r['epoch']:r['val_joint_KL'] for r in history if r['epoch'] in EPOCHS};assert set(recorded)==set(EPOCHS)
                    for epoch in EPOCHS:
                        z=load(folder/f'candidate_{epoch}.npz');p=[z[f'p{i}'] for i in range(len(z))];vals=[independent_kl(rs[j],independent_score(rs[j],p,cfg=='rich_mlp'),*contexts[j]) for j in valids];value=math.fsum(rs[j].count*v for j,v in zip(valids,vals))/sum(rs[j].count for j in valids);assert abs(value-recorded[epoch])<2e-9
                    for budget in (20,80):
                        expected=min((e for e in EPOCHS if e<=budget),key=lambda e:(recorded[e],e));assert expected==f['models'][str(budget)]['selected_epoch'];path=folder/f'selected{budget}.npz';assert sha(path)==f['models'][str(budget)]['sha256'];pp=load(path);p=[pp[f'p{i}'] for i in range(len(pp))];old=load(folder/f'candidate_{expected}.npz');assert all(np.array_equal(v,old[k]) for k,v in pp.items())
                        held=load(folder/f'held_{budget}.npz');assert held['key_ids'].tolist()==[rs[j].kid for j in testids];values=[];offset=0
                        for j in testids:
                            r=rs[j];z=independent_score(r,p,cfg=='rich_mlp');np.testing.assert_allclose(z,held['logits'][offset:offset+len(z)],rtol=0,atol=2e-9);offset+=len(z);values.append(independent_kl(r,z,*contexts[j]))
                        np.testing.assert_allclose(values,held['KL'],rtol=0,atol=2e-9);mean=math.fsum(rs[j].count*v for j,v in zip(testids,values))/sum(rs[j].count for j in testids)
                        row=next(r for r in evaluation['rows'] if (r['tag'],r['config'],r['loss'],r['seed'],r['budget'])==(tag,cfg,loss,seed,budget));assert abs(mean-row['joint_KL'])<2e-9;checked+=1
                    save(O/'verification_progress.json',{'selected_models_checked':checked,'tag':tag,'utc':stamp()})
        cachechecks.append({'tag':tag,'n_keys':len(rs),'n_test_hands':len(handsets[2]),'all_structural_supports_targets_richfeatures_checked':True,'private_reach_reenumeration':'allsmall selectedkeys;32fixedlarge keys, remainingRfrom unchanged previouslyvalidated N2hash'})
    def vals(tag,cfg,loss,budget):return [next(r['joint_KL'] for r in evaluation['rows'] if (r['tag'],r['config'],r['loss'],r['budget'],r['seed'])==(tag,cfg,loss,budget,s)) for s in range(5)]
    checks=0
    for group,kind in (('arm_summaries','arm'),('joint_minus_marginal','loss'),('long_minus_short','duration'),('rich_minus_summary','features')):
        for key,c in evaluation[group].items():
            a=key.split('/')
            if kind=='arm':v=vals(a[0],a[1],a[2],int(a[3]))
            elif kind=='loss':v=np.array(vals(a[0],a[1],'joint',int(a[2])))-vals(a[0],a[1],'marginal',int(a[2]))
            elif kind=='duration':v=np.array(vals(a[0],a[1],a[2],80))-vals(a[0],a[1],a[2],20)
            else:v=np.array(vals(a[0],'rich_mlp',a[1],int(a[2])))-vals(a[0],'summary_mlp',a[1],int(a[2]))
            mean,b=interval(v);assert abs(mean-c['mean'])<1e-12 and np.allclose(b,c['ci95'],atol=1e-12,rtol=0);checks+=1
    save(O/'verification.json',{'utc':stamp(),'n_selected_models':checked,'n_arm_and_contrast_intervals':checks,'cache_checks':cachechecks,'all_independent_forward_DP_normalizers_selection_splits_raw_and_CIs_pass':True,'source_sha256':sha(Path(__file__)),'evaluation_sha256':sha(O/'evaluation.json'),'scope':'Codex separate check, not independent investigator; newrich fields/F/Rmembership allkeys, fullprivateRcheck allsmall/32large, frozenpriorN2posteriorassumption otherwise; pending Claude external'})
    log(H,'N9separate verifier complete,240selectedmodels andallreportedintervals pass; pending external/local.')
if __name__=='__main__':main()
