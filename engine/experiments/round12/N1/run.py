"""N1 approved-source-only pipeline. --preflight never generates experiment data."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,subprocess,sys,time,threading
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;ENGINE=R.parents[1];ROOT=ENGINE.parent
sys.path[:0]=[str(H),str(R),str(ENGINE)]
from belief import encode,ablate,init,predict,loss_grad,decision_losses,F
from common import save,sha,stamp,replace,log
from learning import Adam
from seed_audit import audit as audit_seeds
from judgement import Rules,RoundState
from judgement.engine import make_rngs,deal
from judgement.bidding.model import BidModel,ModelBidder,Agent
from judgement.bidding2.sampler import BidLikelihood
from judgement.policies.target_aware import TargetAwarePolicy
from judgement.policies.ta_plus import ta_plus_v2
from judgement.policies.hybrid_bi import BidInformedWorlds
from judgement.sampling import HandSampler
MODEL_PATH=ENGINE/'experiments/phase1/results/model_ta_final.json';MODEL=BidModel.load(MODEL_PATH)
RULES=Rules();LIK=BidLikelihood(MODEL,RULES,eps=.02,tau=0);O=H/'results';DOMAIN=0x4E31
STAGES={'pilot':(2100000000,0,176,256),'train':(2100000000,1,30800,0),'validation':(2100000000,2,4400,0),'confirmation':(2110000000,3,3520,256)}
SOURCES=sorted({Path(__file__),H/'belief.py',H/'PROTOCOL.md',H/'seed_audit.py',H/'verify.py',R/'common.py',R/'learning.py',MODEL_PATH}|
 {Path(m.__file__).resolve() for name,m in list(sys.modules.items()) if name.startswith('judgement') and getattr(m,'__file__',None) and str(m.__file__).endswith('.py')})
STOP=None
def source_packet():
    raw={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in SOURCES}
    canonical={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in SOURCES}
    fingerprint=hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()
    return {'raw_source_sha256':raw,'canonical_source_sha256':canonical,'source_fingerprint':fingerprint,'numpy':np.__version__,'BLAS':str(np.__config__.CONFIG)}
def approved(packet):
    token='N1_SOURCE_APPROVED '+packet['source_fingerprint']
    assert token in (R/'REVIEW.md').read_text(encoding='utf-8'),'Claude external source preflight approval required before experimental data'
    commit=os.environ.get('N1_APPROVED_COMMIT','');assert commit,'Set the approved pre-run local source commit'
    for rel,want in packet['canonical_source_sha256'].items():
        blob=subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT)
        assert hashlib.sha256(blob.replace(b'\r\n',b'\n')).hexdigest()==want,('Uncommitted/changed dependency',rel)
    return commit
def check_stop():
    if time.time()>=STOP:raise TimeoutError('N1 absolute hard stop')
def check_sources():
    frozen=json.loads((O/'source_freeze.json').read_text())
    for rel,h in frozen['raw_source_sha256'].items():assert sha(ROOT/rel)==h,('Frozen source changed',rel)
def checkpoint(path,**arrays):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    replace(temp,path)
def data_manifest(stage):
    receipts={}
    for start in range(0,STAGES[stage][2],64):
        prefix=O/stage/f'{start:05d}';path=prefix.with_suffix('.json');meta=json.loads(path.read_text())
        assert sha(prefix.with_suffix('.npz'))==meta['arrays_sha256'] and sha(prefix.with_suffix('.jsonl'))==meta['records_sha256']
        receipts[path.name]=sha(path)
    return hashlib.sha256(json.dumps(receipts,sort_keys=True).encode()).hexdigest()
def observe(info):
    return dict(seat=int(info.seat),n=int(info.n_cards),trump=int(info.trump),dealer=int(info.dealer),hand=int(info.hand),
        history=[list(p) for p in info.history],bids=[list(p) for p in info.bids],voids=list(info.voids),won=list(info.tricks_won),left=[info.cards_left(p) for p in range(4)])
def world_marginals(worlds,cards,opps):
    count=np.zeros((len(cards),4),np.int64)
    for world in worlds:
        for i,c in enumerate(cards):
            holder=[j for j,p in enumerate(opps) if (world[p]>>c)&1]
            assert len(holder)<=1
            count[i,holder[0] if holder else 3]+=1
    return count/len(worlds)
def case(stage,g):
    base,code,_,K=STAGES[stage];cell=(g//4)%44;rep=g//176;n=3+cell%11;tr=-1 if (cell//11)%2 else 0;table=cell//22;hero=g%4;dealer=rep%4
    seed=[base,DOMAIN,code,g,3];d,rs=make_rngs(np.random.SeedSequence(seed),4);hands=deal(RULES,n,d)
    state=RoundState(RULES,n,tr,dealer,hands)
    agents=[Agent(ModelBidder(MODEL),ta_plus_v2() if table else TargetAwarePolicy()) for _ in range(4)]
    for _ in range(4):
        p=state.to_act;state.apply_bid(agents[p].bid(state.info_state(p),state.legal_bids(),rs[p]))
    target=int(np.random.default_rng([base,DOMAIN,code,g,4]).integers(1,n-1))
    while not state.is_terminal:
        p=state.to_act
        if p==hero and len(state.history)//4>=target:break
        state.apply_card(agents[p].play(state.info_state(p),state.legal_cards(),rs[p]))
    assert not state.is_terminal
    info=state.info_state(hero);obs=observe(info);x,mask,prop,cards=encode(obs,table);opps=[(hero+k)%4 for k in (1,2,3)]
    # True hands are labels only; no hidden filtering or state passed to encode.
    y=[]
    for c in cards:
        holders=[j for j,p in enumerate(opps) if (state.hands[p]>>c)&1];assert len(holders)<=1
        y.append(holders[0] if holders else 3)
    y=np.array(y,np.uint8);assert mask[np.arange(len(y)),y].all()
    record={'g':g,'base':base,'stage_code':code,'seed':seed,'hero':hero,'n':n,'trump':tr,'table':table,'cell':cell,'rep':rep,
        'observation':obs,'initial_hands':hands,'target_trick':target,'cards':cards}
    data={'X':x,'M':mask,'P':prop,'Y':y};tick=time.perf_counter()
    if K:
        sampler_seed=[base,DOMAIN,code,g,5];u_rng=np.random.default_rng(sampler_seed);b_rng=np.random.default_rng(sampler_seed)
        hs=HandSampler(info);u=[hs.sample(u_rng) for _ in range(K)]
        b,stats=BidInformedWorlds(LIK,batch=64,m_max=2048,ess_target=1).sample(info,K,b_rng)
        data['uniform']=world_marginals(u,cards,opps);data['bi']=world_marginals(b,cards,opps)
        for name in ('uniform','bi'):
            assert np.allclose(data[name].sum(1),1) and np.all(data[name][~mask]==0)
        record['sampler']={k:float(v) for k,v in stats.items()};record['K']=K
    record['baseline_seconds']=time.perf_counter()-tick
    return data,record
def generate(stage,freeze):
    folder=O/stage;folder.mkdir(exist_ok=True);count=STAGES[stage][2]
    for start in range(0,count,64):
        check_stop();check_sources();prefix=folder/f'{start:05d}';receipt=prefix.with_suffix('.json')
        if receipt.exists():
            r=json.loads(receipt.read_text());assert r['freeze_sha256']==freeze and r['count']==min(64,count-start)
            assert sha(prefix.with_suffix('.npz'))==r['arrays_sha256'] and sha(prefix.with_suffix('.jsonl'))==r['records_sha256'];continue
        for suffix in ('.npz','.jsonl','.tmp'):
            pending=prefix.with_suffix(suffix)
            if pending.exists():
                destination=pending.with_name(pending.name+'.orphan_'+str(time.time_ns()));assert not destination.exists()
                replace(pending,destination)
        arrays=[];records=[];tick=time.perf_counter()
        for g in range(start,min(start+64,count)):
            check_stop();data,record=case(stage,g);arrays.append(data);records.append(record)
        offset=np.cumsum([0]+[len(a['Y']) for a in arrays]);pack={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]};pack['offset']=offset
        checkpoint(prefix.with_suffix('.npz'),**pack)
        path=prefix.with_suffix('.jsonl');temp=path.with_suffix('.tmp');temp.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in records),encoding='utf-8');replace(temp,path)
        save(receipt,{'start':start,'count':len(records),'freeze_sha256':freeze,'arrays_sha256':sha(prefix.with_suffix('.npz')),'records_sha256':sha(path),
            'wall_seconds':time.perf_counter()-tick,'baseline_seconds':sum(v['baseline_seconds'] for v in records),'utc':stamp()})
        log(H,f'{stage}: saved{start+len(records)}/{count}rounds')
def load(stage):
    data_manifest(stage)
    arrays=[];records=[];offset=[0]
    for start in range(0,STAGES[stage][2],64):
        p=O/stage/f'{start:05d}';data=dict(np.load(p.with_suffix('.npz')));records.extend(json.loads(s) for s in p.with_suffix('.jsonl').read_text().splitlines())
        offset.extend((data['offset'][1:]+offset[-1]).tolist());arrays.append(data)
    packed={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0] if k!='offset'};packed['offset']=np.array(offset)
    assert len(records)==STAGES[stage][2]
    return packed,records
def probability(data,p,variant):
    out=np.empty_like(data['P'])
    for start in range(0,len(out),8192):
        sl=slice(start,start+8192);out[sl]=predict(ablate(data['X'][sl],variant),data['M'][sl],data['P'][sl],p)
    return out
def strata_ci(values,groups):
    labels=sorted(set(groups));parts=[np.asarray(values)[np.asarray(groups)==s] for s in labels];means=np.array([v.mean() for v in parts])
    mean=float(means.mean());half=float(1.96*np.sqrt(sum(v.var(ddof=1)/len(v) for v in parts))/len(parts))
    return {'n_decisions':len(values),'mean':mean,'ci95':[mean-half,mean+half],'half95':half,'n_strata':len(parts),'scope':'Marginal equal-stratum paired normal95%CI across independent rounds, conditional on frozen ensemble.'}
def fit(train,val,freeze):
    data_hashes={s:data_manifest(s) for s in ('train','validation')}
    path=O/'frozen_models.json'
    if path.exists():
        packet=json.loads(path.read_text());assert packet['source_freeze_sha256']==freeze and packet['data_manifests']==data_hashes
        for rel,h in packet['model_sha256'].items():assert sha(O/rel)==h
        return packet
    groups=np.arange(STAGES['validation'][2])%176;baseline=decision_losses(val['P'],val['Y'],val['offset']);models={};hashes={}
    for variant in ('bids','full'):
        for seed in range(5):
            folder=O/f'{variant}_s{seed}';folder.mkdir(exist_ok=True)
            if (folder/'frozen.json').exists():
                receipt=json.loads((folder/'frozen.json').read_text())
                assert receipt['source_freeze_sha256']==freeze and receipt['model_sha256']==sha(folder/'selected.npz') and receipt['data_manifests']==data_hashes
                continue
            p=init(np.random.SeedSequence([2100000000,DOMAIN,20,seed]));adam=Adam(p);first=1;best=None;best_epoch=None;active=folder/'checkpoint.npz'
            if active.exists():
                a=dict(np.load(active));first=int(a['epoch'])+1;p=[a[f'p{i}'] for i in range(4)];adam=Adam(p);adam.t=int(a['step']);adam.m=[a[f'm{i}'] for i in range(4)];adam.v=[a[f'v{i}'] for i in range(4)];best=None if float(a['best'])<0 else float(a['best']);best_epoch=int(a['best_epoch'])
                trace=folder/'epochs.jsonl';raw=trace.read_text() if trace.exists() else '';history=[]
                for line in raw.splitlines():
                    try:history.append(json.loads(line))
                    except ValueError:
                        trace.with_name('interrupted_trace_'+str(time.time_ns())+'.txt').write_text(raw);break
                history=[v for v in history if v['epoch']<first]
                if not history or history[-1]['epoch']!=first-1:
                    restored=json.loads(str(a['trace_row']));restored['recovered_from_atomic_checkpoint']=True;history.append(restored)
                assert [v['epoch'] for v in history]==list(range(1,first))
                trace.write_text(''.join(json.dumps(v)+'\n' for v in history))
            for epoch in range(first,51):
                check_stop();check_sources();order=np.random.default_rng([2100000000,DOMAIN,21,seed,epoch]).permutation(STAGES['train'][2])
                for start in range(0,len(order),32):
                    ids=order[start:start+32];indices=np.concatenate([np.arange(train['offset'][j],train['offset'][j+1]) for j in ids])
                    w=np.concatenate([np.full(train['offset'][j+1]-train['offset'][j],1/(len(ids)*(train['offset'][j+1]-train['offset'][j]))) for j in ids])
                    _,grad=loss_grad(ablate(train['X'][indices],variant),train['M'][indices],train['P'][indices],train['Y'][indices],w,p);adam.step(p,grad)
                row={'epoch':epoch,'utc':stamp()}
                if epoch in (10,25,50):
                    pred=.999*probability(val,p,variant)+.001*val['P'];value=float(decision_losses(pred,val['Y'],val['offset']).mean());row['validation_loss']=value
                    checkpoint(folder/f'candidate_{epoch}.npz',**{f'p{i}':v for i,v in enumerate(p)})
                    if best is None or value<best:
                        best,best_epoch=value,epoch;checkpoint(folder/'selected.npz',**{f'p{i}':v for i,v in enumerate(p)})
                checkpoint(active,epoch=epoch,step=adam.t,trace_row=np.array(json.dumps(row)),best=-1 if best is None else best,best_epoch=-1 if best_epoch is None else best_epoch,
                    **{f'p{i}':v for i,v in enumerate(p)},**{f'm{i}':v for i,v in enumerate(adam.m)},**{f'v{i}':v for i,v in enumerate(adam.v)})
                with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                log(H,f'fit{variant}_s{seed}: epoch{epoch}/50')
            save(folder/'frozen.json',{'variant':variant,'seed':seed,'selected_epoch':best_epoch,'validation_loss':best,'model_sha256':sha(folder/'selected.npz'),'source_freeze_sha256':freeze,'data_manifests':data_hashes,'utc':stamp()})
        predictions=[]
        for seed in range(5):
            rel=f'{variant}_s{seed}/selected.npz';a=dict(np.load(O/rel));p=[a[f'p{i}'] for i in range(4)];predictions.append(probability(val,p,variant));hashes[rel]=sha(O/rel)
        ensemble=np.mean(predictions,axis=0);choices=[];trials=[]
        for alpha in (0.,.25,.5,1.):
            pred=(1-.999*alpha)*val['P']+.999*alpha*ensemble;loss=decision_losses(pred,val['Y'],val['offset']);choices.append((float(loss.mean()),alpha));trials.append({'alpha':alpha,'loss':float(loss.mean()),'net_minus_prop':strata_ci(loss-baseline,groups)})
        _,alpha=min(choices);models[variant]={'alpha':alpha,'selected_models':[f'{variant}_s{s}/selected.npz' for s in range(5)],'selection_trials':trials}
    gate=next(q['net_minus_prop'] for q in models['full']['selection_trials'] if q['alpha']==models['full']['alpha'])
    packet={'models':models,'model_sha256':hashes,'source_freeze_sha256':freeze,'data_manifests':data_hashes,'selection_gate':gate,
        'confirmation_allowed':models['full']['alpha']>0 and gate['ci95'][1]<=0,'frozen_before_confirmation_utc':stamp()};save(path,packet);return packet
def evaluate(packet):
    data,records=load('confirmation');groups=np.arange(len(records))%176;probabilities={'prop':data['P'],'uniform':.999*data['uniform']+.001*data['P'],'bi':.999*data['bi']+.001*data['P']}
    for variant,model in packet['models'].items():
        pp=[]
        for rel in model['selected_models']:
            a=dict(np.load(O/rel));pp.append(probability(data,[a[f'p{i}'] for i in range(4)],variant))
        alpha=model['alpha'];probabilities['net_'+variant]=(1-.999*alpha)*data['P']+.999*alpha*np.mean(pp,axis=0)
    losses={k:decision_losses(p,data['Y'],data['offset']) for k,p in probabilities.items()};summary={};calibration={};conditional={};per_cell={}
    for name,p in probabilities.items():
        summary[name]=strata_ci(losses[name],groups);conditional_loss=[];bins=[]
        for a,b in zip(data['offset'][:-1],data['offset'][1:]):
            y=data['Y'][a:b];pr=p[a:b];active=y<3;cp=pr[active,:3];cp/=cp.sum(1,keepdims=True)
            conditional_loss.append(float(-np.log(cp[np.arange(len(cp)),y[active]]).mean()))
            conf=pr.max(1);correct=(pr.argmax(1)==y);ix=np.minimum((10*conf).astype(int),9)
            bins.append(np.stack([np.bincount(ix,minlength=10),np.bincount(ix,weights=conf,minlength=10),np.bincount(ix,weights=correct,minlength=10)])/len(y))
        conditional[name]=strata_ci(conditional_loss,groups);bins=np.array(bins);aggregate=bins.mean(0);ece=float(abs(aggregate[2]-aggregate[1]).sum());boot=[]
        rng=np.random.default_rng([2110000000,DOMAIN,90]);strata=[np.arange(s,len(records),176) for s in range(176)]
        for _ in range(2000):
            ids=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]);v=bins[ids].mean(0);boot.append(float(abs(v[2]-v[1]).sum()))
        mass=aggregate[0];calibration[name]={'ECE':ece,'ci95':np.quantile(boot,[.025,.975]).tolist(),'n_decisions':len(records),'round_weighted_bins':aggregate.tolist(),'bootstrap_reps':2000,
            'bin_confidence':[float(c/w) if w else None for c,w in zip(aggregate[1],mass)],'bin_accuracy':[float(c/w) if w else None for c,w in zip(aggregate[2],mass)]}
    primary=strata_ci(losses['net_full']-losses['bi'],groups)
    for n in range(3,14):
        for nt in (0,1):
            for table in (0,1):
                ids=[i for i,q in enumerate(records) if q['n']==n and int(q['trump']==-1)==nt and q['table']==table]
                per_cell[f'n{n}/NT{nt}/table{table}']=strata_ci((losses['net_full']-losses['bi'])[ids],groups[ids])
    checkpoint(O/'confirmation_predictions.npz',**probabilities)
    save(O/'summary.json',{'primary_net_minus_BI':primary,'primary_supported':primary['ci95'][1]<0,'unconditional':summary,'conditional_active':conditional,
        'calibration':calibration,'per_cell_net_minus_BI':per_cell,'confirmation_data_manifest':data_manifest('confirmation'),'raw_decision_losses':{k:v.tolist() for k,v in losses.items()},'utc':stamp(),
        'scope':'Fixed five-fit ensembles; independent round intervals not card-IID/optimization uncertainty; marginal cells/ECE descriptive. No exact posterior claim.'})
def main():
    global STOP
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args();packet=source_packet()
    if args.preflight:
        audit_seeds();save(H/'source_preflight.json',packet);print(packet['source_fingerprint']);return
    commit=approved(packet);STOP=float(os.environ['JUDGEMENT_N1_DEADLINE']);check_stop();O.mkdir(exist_ok=True)
    freeze_path=O/'source_freeze.json';packet['pre_run_commit']=commit
    if freeze_path.exists():assert json.loads(freeze_path.read_text())==packet
    else:save(freeze_path,packet)
    timer=threading.Timer(STOP-time.time(),lambda:os._exit(124));timer.daemon=True;timer.start()
    try:
        audit_seeds();digest=sha(freeze_path);generate('pilot',digest)
        receipts=[json.loads(p.read_text()) for p in (O/'pilot').glob('*.json')];wall=sum(p['wall_seconds'] for p in receipts);baseline=sum(p['baseline_seconds'] for p in receipts)
        projection=(wall-baseline)*(30800+4400+3520)/176+baseline*3520/176+7200
        timing={'projected_seconds':projection,'remaining_seconds':STOP-time.time(),'accepted':projection<=STOP-time.time(),'utc':stamp()}
        if (O/'pilot_resource.json').exists():timing=json.loads((O/'pilot_resource.json').read_text())
        else:save(O/'pilot_resource.json',timing)
        if not timing['accepted']:save(O/'scope_failure.json',{'reason':'resource-only pilot gate','timing':timing,'utc':stamp()});return
        generate('train',digest);generate('validation',digest);train,_=load('train');val,_=load('validation');model=fit(train,val,digest);del train,val
        if not model['confirmation_allowed']:save(O/'scope_failure.json',{'reason':'Predeclared selection NI filter; no confirmation seeds consumed','gate':model['selection_gate'],'utc':stamp()});return
        generate('confirmation',digest);evaluate(model);log(H,'N1 fixed experiment complete; separate audit and Claude external required.')
    finally:timer.cancel()
if __name__=='__main__':main()
