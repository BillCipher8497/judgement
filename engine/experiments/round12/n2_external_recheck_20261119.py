"""Independent check of Claude's marginal floors/small product witnesses.
No new fits or research seeds. IPF is a numerical oracle diagnostic, not a
certified global optimum or a deployable predictor.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,hashlib,math,itertools
from pathlib import Path
import numpy as np
from scipy.stats import t as t_law
H=Path(__file__).resolve().parent;ROOT=H.parents[2];N2=H/'N2/results';OUT=H/'CODEX_REVIEW_20261119';OUT.mkdir(exist_ok=True);DEADLINE=float(os.environ['N123_REVIEW_DEADLINE'])
def check():
    if time.time()>=DEADLINE:raise TimeoutError('N2 recheck hard stop')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):(OUT/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def interval(v):
    n=len(v);mean=float(np.mean(v));half=float(t_law.ppf(.975,n-1)*np.std(v,ddof=1)/np.sqrt(n));return {'n':n,'mean':mean,'ci95':[mean-half,mean+half]}
def decode(k):
    b=bytes(k).ljust(20,b'\0');return b[0],int.from_bytes(b[2:4],'little'),tuple(b[4:4+b[1]])
def fields(hist,R,tr):
    leader=0;trick=[];played=[[] for _ in range(4)];void=[set() for _ in range(4)]
    for c in hist[4:]:
        p=(leader+len(trick))%4
        if trick and c//R!=trick[0][1]//R:void[p].add(trick[0][1]//R)
        played[p].append(c);trick.append((p,c))
        if len(trick)==4:
            p0,c0=trick[0]
            for p1,c1 in trick[1:]:
                if (c1//R==c0//R and c1>c0) or (c1//R!=c0//R and c1//R==tr):p0,c0=p1,c1
            leader=p0;trick=[]
    return played,void,(leader+len(trick))%4
def assignments(S,n,tr,key):
    p,mask,hist=decode(key);played,void,actor=fields(hist,4,tr);assert p==actor;opps=[(p+k)%4 for k in (1,2,3)];cards=[c for c in range(S*4) if not(mask>>c&1) and c not in hist[4:]];cap=[n-len(played[q]) for q in opps];rows=[]
    def rec(j,left,owners):
        if j==3:
            assert not left;rows.append([owners[c] for c in cards]);return
        for take in itertools.combinations(left,cap[j]):
            if any(c//4 in void[opps[j]] for c in take):continue
            chosen=set(take);rec(j+1,[c for c in left if c not in chosen],{**owners,**{c:j for c in take}})
    rec(0,cards,{});assert rows;return cards,np.array(rows,np.int8).reshape(-1,len(cards)) if cards else np.empty((1,0),np.int8)
def oracle(target,worlds,reach):
    U=len(target)
    if U==0:assert reach==1;return 0.,0.,0.,0
    target=np.asarray(target,float);positive=target>0;keep=positive[np.arange(U)[None,:],worlds].all(1);w=worlds[keep];assert len(w)>=reach
    def objective(theta):
        values=theta[np.arange(U)[None,:],w].sum(1);m=float(values.max());logZ=m+math.log(float(np.exp(values-m).sum()));expected=math.fsum(float(target[i,h]*theta[i,h]) for i,h in zip(*np.where(positive)));return logZ-expected-math.log(reach)
    plug=np.zeros_like(target);plug[positive]=np.log(target[positive]);plugin=objective(plug)
    theta=np.zeros_like(target);q=np.full(len(w),1/len(w));gap=1.
    for iteration in range(1000):
        for card in range(U):
            current=np.bincount(w[:,card],weights=q,minlength=3);ratio=np.divide(target[card],current,out=np.ones(3),where=current>0);assert np.all(current[target[card]>0]>0);q*=ratio[w[:,card]];theta[card,positive[card]]+=np.log(ratio[positive[card]])
        q/=q.sum();gap=max(float(np.max(np.abs(np.bincount(w[:,card],weights=q,minlength=3)-target[card]))) for card in range(U))
        if gap<1e-10:break
    fitted=objective(theta);assert fitted<=plugin+1e-7
    return plugin,fitted,gap,iteration+1
def main():
    f=json.loads((N2/'source_freeze.json').read_text())
    for rel,h in f['raw_sources'].items():assert sha(ROOT/rel)==h
    results={}
    for S,n,tr in ((2,2,0),(2,2,-1),(3,3,0),(3,3,-1)):
        check();tag=f'S{S}R4n{n}_t{tr}';folder=N2/tag;manifest=json.loads((folder/'posterior_manifest.json').read_text());assert sha(folder/'posterior_supports.npz')==manifest['posterior_sha256']
        with np.load(folder/'posterior_supports.npz') as a:data={k:a[k] for k in a.files}
        chunks=[]
        for path in sorted((folder/'joint_KL').glob('*.npz')):
            with np.load(path) as a:
                take=a['split']==2;chunks.append({k:a[k][take] for k in ('group_index','reach','net_KL','baseline_KL','raw_marginal_card_logloss')})
        d={k:np.concatenate([a[k] for a in chunks]) for k in chunks[0]};ids=d['group_index'];reach=d['reach'].astype(float);den=reach.sum();keys=data['keys'][ids];hands=np.array([decode(k)[1] for k in keys]);counts=data['holder_counts'][ids].astype(float);prob=counts/reach[:,None,None];entropy=np.zeros(len(ids))
        for c in range(S*4):
            pp=prob[:,c,:];terms=np.zeros_like(pp);np.log(pp,out=terms,where=pp>0);entropy-=(pp*terms).sum(1)
        U=np.array([int(x).bit_count() for x in data['unseen_mask'][ids]]);entropy=np.divide(entropy,U,out=np.zeros_like(entropy),where=U>0);floor=float(reach@entropy/den);ce=np.sum(reach[:,None]*d['raw_marginal_card_logloss'],axis=0)/den;closed=(ce[0]-ce[1:])/(ce[0]-floor)
        handdiff=[]
        for hand in np.unique(hands):
            take=hands==hand;v=d['net_KL'][take].mean(1)-d['baseline_KL'][take,1];handdiff.append(float(reach[take]@v/reach[take].sum()))
        out={'exact_test_decision_world_weight':int(den),'test_hands':len(handdiff),'exact_full_info_marginal_entropy_floor':floor,'capacity_prop_card_CE':float(ce[0]),'net_card_CE_by_seed':ce[1:].tolist(),'fraction_marginal_gap_closed_seed_summary':interval(closed),'hand_sensitivity_t95':interval(handdiff),'hands_net_better_than_BI':sum(v<0 for v in handdiff),'scope':'Complete fixed test population quantities, no chance-sampling CI. Seed tCI only optimizer variation. Hand tCI is sensitivity to a hypothetical hand population, not a sampling interval over enumerated worlds.'}
        if S==2:
            plugin=[];ipf=[];gaps=[];iters=[]
            for j,key in enumerate(keys):
                check();cards,worlds=assignments(S,n,tr,key);target=prob[j,cards,:];a,b,gap,it=oracle(target,worlds,int(reach[j]));plugin.append(a);ipf.append(b);gaps.append(gap);iters.append(it)
            plugin=np.array(plugin);ipf=np.array(ipf);bi=float(reach@d['baseline_KL'][:,1]/den)
            out.update(exact_feasible_plugin_product_KL=float(reach@plugin/den),numerical_IPF_candidate_KL=float(reach@ipf/den),matched_soft_BI_KL=bi,IPF_worst_marginal_gap=max(gaps),IPF_nonconverged=sum(g>=1e-10 for g in gaps),IPF_keys_worse_than_BI=int((ipf>d['baseline_KL'][:,1]+1e-8).sum()),oracle_limit='Oracle uses full posterior marginals, not learned/deployable inputs. Plugin is a feasible family witness, proving attainable aggregate KL belowBI. IPF is numerical optimization without a rigorous global-optimum certificate; not pointwise dominance.')
            assert out['exact_feasible_plugin_product_KL']<bi
        results[tag]=out;save('n2_external_numbers.json',results);print('N2checked',tag,flush=True)
    save('n2_complete.json',{'status':'PASS for independently recomputed quantities; no optimizer-cause or certified-global-optimum claim','results':results,'new_training_or_seed_draws':False})
if __name__=='__main__':main()
