"""New late-play experiment plumbing; immutable policy sources only imported."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,sys,time,types,importlib.util
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;ENGINE=H.parents[2];ROOT=ENGINE.parent;O=H/'results';DOMAIN=0x4e40
sys.path[:0]=[str(R),str(ENGINE)]
from common import save,sha,stamp,replace,log
from learning import initialize,forward,head_loss,Adam
from judgement.engine import make_rngs,deal,play_game
from judgement.state import RoundState,InfoState,BID,PLAY,state_from_info
from judgement.sampling import HandSampler
from judgement.bidding.model import Agent,ModelBidder
def import_observables():
    prior=sys.modules.get('common');shim=types.ModuleType('common');shim.np=np;shim.ENGINE=ENGINE;sys.modules['common']=shim
    try:
        spec=importlib.util.spec_from_file_location('n10_frozen_private_observables',R/'N3/observables.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    finally:
        if prior is None:del sys.modules['common']
        else:sys.modules['common']=prior
OBS=import_observables();RULES=OBS.RULES;MODEL=OBS.MODEL
COUNTS={'pilot':96,'train':6144,'validation':1536,'held':1536,'fixtures':768};STAGES={name:i for i,name in enumerate(COUNTS)};ROUNDS=[i for i,n in enumerate(RULES.ladder()) if n>1]
ALPHAS=(0.,.25,.5,1.,2.,4.);EPOCHS=(5,10,25,50);ARMS=('hard','set');BASE=2180000000;FITBASE=2190000000
def check():assert time.time()<float(os.environ['JUDGEMENT_JOB_DEADLINE'])
def npz(p,**v):
    p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**v)
    replace(tmp,p)
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def partition(hand):
    h=hashlib.sha256(json.dumps([BASE,DOMAIN,'hand',int(hand)],separators=(',',':')).encode()).digest();b=int.from_bytes(h[:8],'big')%10
    return 0 if b<6 else 1 if b<8 else 2
def interval(v,strata):
    v=np.asarray(v,float);s=np.asarray(strata);n=len(v);assert n>1;var=0.
    for k in np.unique(s):
        a=v[s==k];assert len(a)>1;var+=(len(a)/n)**2*float(a.var(ddof=1)/len(a))
    mean=float(v.mean());h=1.959963984540054*np.sqrt(var)
    return {'n':n,'mean':mean,'ci95':[float(mean-h),float(mean+h)],'method':'paired/equal-stratum normal95%; fixed selected ensemble, selection validation optimistic where used'}
class Registry:
    def __init__(self):
        self.states=set();self.addresses=set();O.mkdir(exist_ok=True);self.path=O/'seed_addresses.jsonl'
        def old(a):
            q=np.random.SeedSequence(a)
            for r in [q]+q.spawn(5):self.states.add(r.generate_state(4).tobytes())
        for g in range(6000):old([2100000000,g])
        for g in range(600):old([2100000000,10000000+g])
        for stage,base,n in ((0,2100000000,176),(1,2100000000,30800),(2,2100000000,4400),(3,2110000000,3520)):
            for g in range(n):
                for tail in (3,4,5):old([base,0x4e31,stage,g,tail])
        for ns,epochs in ((0x4e31,50),(0x4e32,20)):
            for s in range(5):
                old([2100000000,ns,20,s])
                for e in range(1,epochs+1):old([2100000000,ns,21,s,e])
        for tail in (90,91):old([2110000000,0x4e31,tail])
        for line in (R/'N3/results/seed_addresses.jsonl').read_text().splitlines():
            for h in json.loads(line)['root_and_5children_hex']:self.states.add(bytes.fromhex(h))
        for b,n in ((2150020000,4),(2150000000,300),(2150010000,300)):
            for g in range(n):old(b+g)
        for r in json.loads((R/'N5/preflight.json').read_text())['new']:self.states.add(bytes.fromhex(r['state']))
        for r in json.loads((R/'N9/preflight.json').read_text())['roots']:
            for h in r['root_and5children']:self.states.add(bytes.fromhex(h))
        self.prior=len(self.states)
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                r=json.loads(line);self.addresses.add(tuple(r['address']))
                for h in r['states']:self.states.add(bytes.fromhex(h))
    def add(self,address):
        a=tuple(map(int,np.atleast_1d(address)))
        if a in self.addresses:return
        q=np.random.SeedSequence(list(a));states=[r.generate_state(4).tobytes() for r in [q]+q.spawn(5)];assert len(set(states))==6 and not set(states)&self.states,('seed collision',a)
        self.states.update(states);self.addresses.add(a)
        with self.path.open('a',encoding='utf-8') as f:f.write(json.dumps({'address':a,'states':[s.hex() for s in states]})+'\n')
    def receipt(self):return {'utc':stamp(),'prior_known_states':self.prior,'new_root_addresses':len(self.addresses),'collisions':0,'scope':'known N0-N9roots/journals and declaredchildren, not unknown external uses; N6/N11noRNG; intentional CRN roots counted once'}
def private_rollout(info,hidden,card,hero,address):
    st=state_from_info(info,hidden);st.apply_card(card);policies=OBS.agents();policies[hero]=Agent(ModelBidder(MODEL),OBS.baseline());root=np.random.SeedSequence(address);rngs=[np.random.default_rng(s) for s in root.spawn(4)]
    while st.phase==PLAY:
        p=st.to_act;legal=st.legal_cards();private=OBS.detach(InfoState(st,p));assert not hasattr(private,'_st');choice=policies[p].play(private,legal,rngs[p]);assert legal>>choice&1;st.apply_card(choice)
    return float(st.scores()[hero])
def context(name,g,registry):
    stage=STAGES[name];base=2220000000 if name=='fixtures' else BASE;hero=g%4;ri=ROUNDS[(g//4)%24];n=RULES.ladder()[ri];want=None if name=='fixtures' else 0 if name in ('pilot','train') else 1 if name=='validation' else 2
    for attempt in range(1000):
        check();prefix=[base,DOMAIN,stage,g,attempt,3];registry.add(prefix);dr,pr=make_rngs(prefix,4);hands=deal(RULES,n,dr);original=hands[hero]
        if want is not None and partition(original)!=want:continue
        st=RoundState(RULES,n,RULES.trump_for_round(ri),RULES.dealer_for_round(ri),hands);policies=OBS.agents();policies[hero]=Agent(ModelBidder(MODEL),OBS.baseline())
        while st.phase==BID:
            p=st.to_act;st.apply_bid(policies[p].bid(InfoState(st,p),st.legal_bids(),pr[p]))
        while st.phase==PLAY:
            p=st.to_act;info=InfoState(st,p);legal=st.legal_cards()
            if p==hero and OBS.active(info,legal):
                private=OBS.detach(info);bc=OBS.baseline().play(private,legal,pr[p]);x,b,cards=OBS.encode(private,legal,bc);assert max(private.cards_left(p) for p in range(4))<=2
                row={'x':x,'base':b/8,'cards':cards,'hit':10+dict(private.bids)[hero],'g':g,'attempt':attempt,'stratum':g%96,'original_hand':original,'round_index':ri,'observation':OBS.observation(private)}
                if name=='fixtures':return row
                # The primary reward vector uses the ACTUAL current hidden deal
                # from this context generator, offline only. Future policies
                # remain private/fixed; no actual hidden array enters x/base.
                truth={p:int(st.hands[p]) for p in range(4) if p!=hero};ta=[base,DOMAIN,stage,g,attempt,7];registry.add(ta)
                reward=[private_rollout(private,truth,c,hero,ta) for c in cards];assert set(reward)<={0.,float(row['hit'])}
                row.update(q=[[v/row['hit'] for v in reward]],true_worlds=[int(h) for h in st.hands])
                wa=[base,DOMAIN,stage,g,attempt,5];registry.add(wa);rng=np.random.default_rng(wa);sampler=HandSampler(private);worlds=[];q=[]
                for w in range(16):
                    hidden=sampler.sample(rng);worlds.append([hidden.get(p,private.hand) for p in range(4)]);ra=[base,DOMAIN,stage,g,attempt,w,6];registry.add(ra);v=[private_rollout(private,hidden,c,hero,ra) for c in cards];assert set(v)<={0.,float(row['hit'])};q.append([a/row['hit'] for a in v])
                row.update(proxy_q=q,worlds=worlds);return row
            st.apply_card(policies[p].play(info,legal,pr[p]))
    raise RuntimeError(f'Bounded rejection exhausted {name}/{g}; preserve scope failure, no acceptance retuning')
def generate(name,registry,sources):
    folder=O/'data'/name;folder.mkdir(parents=True,exist_ok=True);N=COUNTS[name]
    for start in range(0,N,64):
        check();p=folder/f'{start:06d}.npz';meta=p.with_suffix('.json')
        if meta.exists():
            r=json.loads(meta.read_text());assert r['sources']==sources and sha(p)==r['data_sha256'];continue
        tick=time.perf_counter();rows=[context(name,g,registry) for g in range(start,min(start+64,N))];elapsed=time.perf_counter()-tick;fields=('x','base','cards','hit','g','attempt','stratum','original_hand','round_index')+(() if name=='fixtures' else ('q','proxy_q','worlds','true_worlds'))
        if p.exists():
            old=load(p)
            for k in fields:np.testing.assert_array_equal(old[k],np.asarray([r[k] for r in rows]))
        else:npz(p,**{k:np.asarray([r[k] for r in rows]) for k in fields})
        save(meta,{'stage':name,'first_g':start,'n':len(rows),'observations':[r['observation'] for r in rows],'sources':sources,'data_sha256':sha(p),'generation_wall_seconds':elapsed,'utc':stamp()});save(O/'seed_receipt.json',registry.receipt());log(H,f'Data {name} {start+len(rows)}/{N}')
    chunks=[load(p) for p in sorted(folder.glob('*.npz'))];data={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]};assert len(data['g'])==N;np.testing.assert_array_equal(data['g'],np.arange(N));return data
def init(seed):
    p=initialize([683,64,32,2],np.random.SeedSequence([FITBASE,DOMAIN,20,seed]));p[-2][:]=0;p[-1][:]=0;return p
def targets(q,arm):
    return np.eye(2)[q.argmax(2)].mean(1) if arm=='hard' else (q==q.max(2,keepdims=True)).mean(1)
def grad(x,base,target,w,p,arm):
    f,acts=forward(x,p);value,dz=head_loss(base+f,np.ones(f.shape,bool),target,w,'hard_ce' if arm=='hard' else 'set_bce');gr=[None]*len(p)
    for i in range(len(p)-2,-1,-2):
        value+=.5e-4*float((p[i]**2).sum());gr[i]=acts[i//2].T@dz+1e-4*p[i];gr[i+1]=dz.sum(0)
        if i:dz=(dz@p[i].T)*(1-acts[i//2]**2)
    return value,gr
def scores(data,p,alpha):return data['base']+alpha*forward(data['x'],p)[0]
def ensemble(data,models):return np.mean([scores(data,m['params'],m['alpha']) for m in models],axis=0)
def identity_bound(models):
    from fractions import Fraction as F
    bound=F(0)
    for m in models:
        p=m['params'];width=p[-2].shape[0];last=sum((abs(F(float(p[-2][j,0]))-F(float(p[-2][j,1]))) for j in range(width)),F(0))+abs(F(float(p[-1][0]))-F(float(p[-1][1])))
        bound+=F(float(m['alpha']))*last
    bound/=len(models);slack=F(1,4)-bound
    return {'residual_difference_bound_exact':str(bound),'baseline_margin_exact':'1/4','strict_slack_exact':str(slack),'global_baseline_identity_by_tanh_margin':slack>0,'scope':'Same bounded-tanh argument as auditedN3, exact binary coefficients; new numericalbound pending external check, not a new convergence theorem'}
