"""Matched new conditional-product objectives. No experiment starts on import."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,sys,time,heapq
from pathlib import Path
from types import SimpleNamespace
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;ENGINE=H.parents[2];ROOT=ENGINE.parent;O=H/'results'
sys.path[:0]=[str(R),str(R/'N2'),str(ENGINE),str(ENGINE/'experiments/round9/D3')]
from common import save,sha,stamp,replace,log
from learning import Adam
from judgement.cfr.toy import ToyGame
from judgement.cfr.heuristics import myopic_bid
from export import decode
from features import observation,encode
from corpus import hand_partitions,structural_and_bid_KL
from joint import allocations,product_kl
from reference import independent_deals,deal_index,structural_worlds
BASE=2170000000;DOMAIN=0x4e39;TAGS=('S2R4n2_t0','S2R4n2_t-1','S3R4n3_t0','S3R4n3_t-1')
CONFIGS=('summary_linear','summary_mlp','rich_mlp');LOSSES=('marginal','joint');EPOCHS=(5,10,20,40,80)
def check():assert time.time()<float(os.environ['JUDGEMENT_JOB_DEADLINE'])
def npz(path,**v):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**v)
    replace(tmp,path)
def load(path):
    with np.load(path) as a:return {k:a[k] for k in a.files}
def digest(v):return hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).digest()
def game(tag):return ToyGame(N=4,S=2 if tag.startswith('S2') else 3,R=4,n=2 if tag.startswith('S2') else 3,trump=0 if tag.endswith('t0') else -1)
def rich(g,obs,x,public):
    ctx=np.zeros(194);j=0;ctx[obs['hand']]=1;j=12
    for b in obs['history'][:4]:ctx[j+b]=1;j+=4
    for slot in range(12):
        hist=obs['history'][4:];ctx[j+(hist[slot] if slot<len(hist) else 12)]=1;j+=13
    ctx[j+obs['seat']]=1;j+=4;ctx[j+(3 if g.trump==-1 else g.trump)]=1;j+=4;ctx[j:j+2]=[g.n/3,g.S/3];j+=2;assert j==194
    out=np.zeros((len(x),4,245),np.float32);out[:,:,:35]=x;out[:,:,35:229]=ctx
    for c,card in enumerate(public['cards']):
        out[c,:,229+card]=1
        for k,p in enumerate(public['opponents']):out[c,k,241+p]=1
    return out
def select(g,keys,unseen):
    parts=hand_partitions(g);table={sum(1<<c for c in h):s for h,s in parts.items()};b=np.frombuffer(keys.tobytes(),np.uint8).reshape(-1,20);masks=b[:,2].astype(np.uint16)+256*b[:,3].astype(np.uint16);lut=np.array([i.bit_count() for i in range(4096)],np.uint8)
    ids=np.flatnonzero(lut[unseen]>=2);buckets={s:{} for s in range(3)}
    for i in ids:
        mask=int(masks[i]);buckets[table[mask]].setdefault(mask,[]).append(int(i))
    selected=[];splits=[]
    for split,cap in ((0,2048),(1,512),(2,1024)):
        pools=[]
        for mask,values in sorted(buckets[split].items()):
            prefix=[BASE,DOMAIN,g.S,g.n,g.trump,split]
            ordered=heapq.nsmallest(cap,values,key=lambda i:digest(prefix+[bytes(keys[i]).ljust(20,b'\0').hex()]))
            pools.append(ordered)
        take=[]
        for k in range(max(map(len,pools))):
            for pool in pools:
                if k<len(pool):take.append(pool[k])
                if len(take)==cap:break
            if len(take)==cap:break
        assert take;selected.extend(take);splits.extend([split]*len(take))
    return np.array(selected,np.int64),np.array(splits,np.uint8)
def build(tag,sources):
    folder=O/tag;folder.mkdir(exist_ok=True);path=folder/'data.npz';receipt=folder/'data.json'
    p=R/'N2/results'/tag;m=json.loads((p/'posterior_manifest.json').read_text());assert sha(p/'posterior_supports.npz')==m['posterior_sha256'];g=game(tag)
    if receipt.exists():
        r=json.loads(receipt.read_text());assert r['sources']==sources and r['posterior_sha256']==m['posterior_sha256'] and sha(path)==r['data_sha256'];return rows(load(path)),r
    check();d=load(p/'posterior_supports.npz');ids,split=select(g,d['keys'],d['unseen_mask']);save(folder/'selection.json',{'ids':ids.tolist(),'split':split.tolist(),'posterior_sha256':m['posterior_sha256'],'utc':stamp()})
    worlds=list(independent_deals(g));assert len(worlds)==m['n_exact_chance_deals']
    for j in np.linspace(0,len(worlds)-1,64,dtype=int):assert deal_index(g,worlds[int(j)])==j
    xs=[];ms=[];ps=[];ts=[];fs=[];rs=[];ws=[];co=[0];fo=[0];ro=[0];counts=[];bikl=[]
    for j,i in enumerate(ids):
        check();obs=observation(g,d['keys'][i]);x,mask,prop,_,public=encode(g,obs);cards=public['cards'];U=len(cards);assert U>=2
        assigned=[]
        for hands in allocations(public['capacities'],mask[:,:3]):
            owners=np.empty(U,np.uint8)
            for k,subset in enumerate(hands):owners[list(subset)]=k
            assigned.append(owners)
        F=np.array(assigned,np.uint8);assert len(F);lookup={tuple(a):k for k,a in enumerate(F)};a,b=map(int,d['support_offset'][i:i+2]);wids=d['support_deal_index'][a:b].astype(int);count=int(d['reach_count'][i]);assert len(wids)==count
        posterior=[]
        for w in wids:
            world=worlds[w];assert tuple(world[obs['seat']])==tuple(obs['hand'])
            owners=tuple(next(k for k,p in enumerate(public['opponents']) if card in world[p]) for card in cards);posterior.append(lookup[owners])
        assert len(set(posterior))==count
        target=np.zeros((U,4));target[:,:3]=d['holder_counts'][i,list(cards),:]/count
        manual=np.zeros_like(target)
        for owners in F[posterior]:manual[np.arange(U),owners]+=1/count
        np.testing.assert_allclose(manual,target,rtol=0,atol=1e-12)
        xs.append(rich(g,obs,x,public));ms.append(mask);ps.append(prop);ts.append(target);fs.append(F.ravel());rs.extend(posterior);ws.extend(wids.tolist());co.append(co[-1]+U);fo.append(fo[-1]+F.size);ro.append(ro[-1]+count);counts.append(count)
        if split[j]==2:bikl.append(structural_and_bid_KL(g,obs,public,count,myopic_bid)[0]['BI_KL_float'])
        else:bikl.append(float('nan'))
        if (j+1)%128==0:save(folder/'data_progress.json',{'keys':j+1,'target':len(ids),'utc':stamp()})
    npz(path,X=np.concatenate(xs),mask=np.concatenate(ms),prop=np.concatenate(ps),target=np.concatenate(ts),owners=np.concatenate(fs),posterior=np.array(rs,np.int32),world_ids=np.array(ws,np.int32),card_offset=np.array(co),owner_offset=np.array(fo),reach_offset=np.array(ro),count=np.array(counts),ids=ids,split=split,keys=d['keys'][ids],BI_KL=np.array(bikl))
    desc={str(s):{'n_keys':int(np.sum(split==s)),'n_hands':len({decode(d['keys'][i])[1] for i in ids[split==s]}),'reach':sum(counts[j] for j in range(len(ids)) if split[j]==s)} for s in range(3)}
    r={'tag':tag,'sources':sources,'posterior_sha256':m['posterior_sha256'],'data_sha256':sha(path),'selection':desc,'utc':stamp()};save(receipt,r);return rows(load(path)),r
def rows(d):
    result=[]
    for j in range(len(d['ids'])):
        a,b=map(int,d['card_offset'][j:j+2]);f,k=map(int,d['owner_offset'][j:j+2]);r,s=map(int,d['reach_offset'][j:j+2]);U=b-a
        result.append(SimpleNamespace(x=d['X'][a:b],mask=d['mask'][a:b],prop=d['prop'][a:b],target=d['target'][a:b],F=d['owners'][f:k].reshape(-1,U),posterior=d['posterior'][r:s],world_ids=d['world_ids'][r:s],count=int(d['count'][j]),split=int(d['split'][j]),kid=int(d['ids'][j]),key=d['keys'][j],BI=float(d['BI_KL'][j])))
    return result
def initialize(arch,seed):
    rng=np.random.default_rng(seed)
    return [np.zeros((245,1)),np.zeros(1)] if arch==0 else [rng.normal(0,1/np.sqrt(245),(245,32)),np.zeros(32),np.zeros((32,1)),np.zeros(1)]
def forward(x,p):
    flat=x.reshape(-1,245)
    if len(p)==2:return (flat@p[0]+p[1]).reshape(-1,4),(flat,None,None)
    pre=flat@p[0]+p[1];hidden=np.maximum(0,pre);return (hidden@p[2]+p[3]).reshape(-1,4),(flat,pre,hidden)
def score(row,p,rich_input):
    x=row.x.copy()
    if not rich_input:x[:,:,35:]=0
    z,_=forward(x,p);return np.where(row.mask,z+np.log(np.maximum(row.prop,1e-300)),-np.inf)
def joint(z,row):
    U=len(z);values=z[np.arange(U)[None,:],row.F].sum(1);mx=values.max();e=np.exp(values-mx);prob=e/e.sum();logz=float(mx+np.log(e.sum()));expected=float(np.sum(row.target[row.mask]*z[row.mask]));kl=logz-expected-np.log(row.count)
    assert kl>=-1e-9
    return float(kl),prob,logz
def loss_gradient(batch,p,objective,rich_input,meanreach,l2=1e-4):
    x=np.concatenate([r.x for r in batch]).copy()
    if not rich_input:x[:,:,35:]=0
    mask=np.concatenate([r.mask for r in batch]);prop=np.concatenate([r.prop for r in batch]);raw,cache=forward(x,p);z=np.where(mask,raw+np.log(np.maximum(prop,1e-300)),-np.inf);dz=np.zeros_like(z);value=0.;a=0
    for row in batch:
        U=len(row.x);b=a+U;zz=z[a:b];weight=row.count/(meanreach*len(batch)*U)
        if objective=='joint':
            _,prob,lz=joint(zz,row);marg=np.bincount((row.F+4*np.arange(U)).ravel(),weights=np.repeat(prob,U),minlength=4*U).reshape(U,4)
            value+=weight*(lz-float(np.sum(row.target[row.mask]*zz[row.mask])));dz[a:b]=weight*(marg-row.target)
        else:
            mx=zz.max(1,keepdims=True);e=np.exp(zz-mx);pp=e/e.sum(1,keepdims=True);lp=zz-mx-np.log(e.sum(1,keepdims=True));value-=weight*float(np.sum(row.target[row.mask]*lp[row.mask]));dz[a:b]=weight*(pp-row.target)
        a=b
    flat,pre,hidden=cache;dd=dz.reshape(-1,1)
    if len(p)==2:gr=[flat.T@dd+l2*p[0],dd.sum(0)]
    else:
        dh=(dd@p[2].T)*(pre>0);gr=[flat.T@dh+l2*p[0],dh.sum(0),hidden.T@dd+l2*p[2],dd.sum(0)]
    value+=.5*l2*sum(float((p[i]**2).sum()) for i in range(0,len(p),2));return float(value),gr
def evaluate(rs,p,rich_input):
    vals=np.array([joint(score(r,p,rich_input),r)[0] for r in rs]);weights=np.array([r.count for r in rs]);return float(vals@weights/weights.sum()),vals
