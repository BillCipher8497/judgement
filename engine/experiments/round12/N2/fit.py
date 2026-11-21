"""N2 finite posterior-marginal learner prototype; no self-starting job."""
import json,time,hashlib,sys
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent))
from learning import Adam
from neural import initialize,probabilities,loss_gradient
from corpus import marginal_risk,BASE,DOMAIN
from export import atomic_npz,digest

def save(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    if temp.exists():temp.rename(temp.with_name(temp.name+'.orphan_'+str(time.time_ns())))
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)

def predict_pack(data,params):
    out=np.empty_like(data['P'])
    for start in range(0,len(out),8192):
        sl=slice(start,start+8192);out[sl]=probabilities(data['X'][sl],data['M'][sl],data['P'][sl],params)
    return out

def params(path):
    a=np.load(path);return [a[f'p{k}'] for k in range(4)]

def check_deadline(deadline):
    if time.time()>=deadline:raise TimeoutError('N2 absolute hard stop')

def fit_models(corpus,folder,freeze,deadline,source_check,status):
    folder=Path(folder);folder.mkdir(exist_ok=True)
    trainids=corpus.selected(0,30000);valids=corpus.selected(1,4000);selection=folder/'fitting_keys.npz'
    if selection.exists():
        a=np.load(selection);np.testing.assert_array_equal(a['train'],trainids);np.testing.assert_array_equal(a['validation'],valids)
    else:atomic_npz(selection,train=trainids,validation=valids)
    keyhash=digest(selection);posteriorhash=digest(corpus.folder/'posterior_supports.npz')
    train=corpus.pack(trainids);val=corpus.pack(valids);meanreach=float(train['count'].mean());out=[]
    for seed in range(5):
        check_deadline(deadline);source_check();f=folder/f's{seed}';f.mkdir(exist_ok=True);receipt=f/'frozen.json'
        if receipt.exists():
            r=json.loads(receipt.read_text());assert r['source_freeze_sha256']==freeze and r['selected_keys_sha256']==keyhash and r['posterior_sha256']==posteriorhash and r['selected_model_sha256']==digest(f/'selected.npz');out.append(r);continue
        p=initialize(np.random.SeedSequence([BASE,DOMAIN,20,seed]));adam=Adam(p);first=1;best=None;best_epoch=None;active=f/'checkpoint.npz';trace=f/'epochs.jsonl'
        if active.exists():
            a=np.load(active);first=int(a['epoch'])+1;p=[a[f'p{i}'] for i in range(4)];adam=Adam(p);adam.t=int(a['step']);adam.m=[a[f'm{i}'] for i in range(4)];adam.v=[a[f'v{i}'] for i in range(4)];best=None if float(a['best'])<0 else float(a['best']);best_epoch=int(a['best_epoch'])
            raw=trace.read_text() if trace.exists() else '';history=[];repair=False
            for line in raw.splitlines():
                try:history.append(json.loads(line))
                except ValueError:repair=True;break
            if any(r['epoch']>=first for r in history):repair=True
            history=[r for r in history if r['epoch']<first]
            if not history or history[-1]['epoch']!=first-1:history.append(json.loads(str(a['trace_row'])));repair=True
            assert [r['epoch'] for r in history]==list(range(1,first))
            if repair:
                if trace.exists():trace.rename(trace.with_name('interrupted_trace_'+str(time.time_ns())+'.txt'))
                trace.write_text(''.join(json.dumps(r)+'\n' for r in history))
        for epoch in range(first,21):
            check_deadline(deadline);source_check();order=np.random.default_rng([BASE,DOMAIN,21,seed,epoch]).permutation(len(trainids))
            for start in range(0,len(order),128):
                check_deadline(deadline);ids=order[start:start+128];index=[];weights=[]
                for j in ids:
                    a,b=train['offset'][j:j+2];index.extend(range(a,b))
                    if b>a:weights.extend([train['count'][j]/(meanreach*len(ids)*(b-a))]*(b-a))
                index=np.asarray(index,np.int64);w=np.asarray(weights,float)
                _,gradient=loss_gradient(train['X'][index],train['M'][index],train['P'][index],train['T'][index],w,p);adam.step(p,gradient)
            row={'epoch':epoch,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            if epoch in (5,10,20):
                pred=.999*predict_pack(val,p)+.001*val['P'];risk=marginal_risk(val,pred);row['selection_marginal_loss']=risk;atomic_npz(f/f'candidate_{epoch}.npz',**{f'p{i}':v for i,v in enumerate(p)})
                if best is None or risk<best:best,best_epoch=risk,epoch;atomic_npz(f/'selected.npz',**{f'p{i}':v for i,v in enumerate(p)})
            atomic_npz(active,epoch=epoch,step=adam.t,best=-1 if best is None else best,best_epoch=-1 if best_epoch is None else best_epoch,trace_row=np.array(json.dumps(row)),**{f'p{i}':v for i,v in enumerate(p)},**{f'm{i}':v for i,v in enumerate(adam.m)},**{f'v{i}':v for i,v in enumerate(adam.v)})
            with trace.open('a',encoding='utf-8') as writer:writer.write(json.dumps(row)+'\n')
            status(f'N2fit{corpus.g.S}suit/trump{corpus.g.trump}/s{seed}:epoch{epoch}/20')
        selected=params(f/'selected.npz');net=predict_pack(val,selected);trials=[]
        for alpha in (0.,.25,.5,1.):trials.append({'alpha':alpha,'validation_marginal_loss':marginal_risk(val,(1-.999*alpha)*val['P']+.999*alpha*net)})
        chosen=min(trials,key=lambda r:(r['validation_marginal_loss'],r['alpha']));baseline=marginal_risk(val,val['P'])
        result={'seed_index':seed,'source_freeze_sha256':freeze,'posterior_sha256':posteriorhash,'selected_keys_sha256':keyhash,'selected_epoch':best_epoch,'alpha':chosen['alpha'],'selection_trials':trials,'baseline_validation_marginal_loss':baseline,
            'selection_noninferior_on_selected_keys':chosen['validation_marginal_loss']<=baseline+1e-12,'selection_scope':'Exact selected validation-key population; alpha0 included; no independent noninferiority guarantee; all held joint outcomes reported',
            'training':corpus.describe(trainids),'validation':corpus.describe(valids),'selected_model_sha256':digest(f/'selected.npz'),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        save(receipt,result);out.append(result)
    final={'models':out,'source_freeze_sha256':freeze,'selected_keys_sha256':keyhash,'posterior_sha256':posteriorhash,'data_scope':'Own-hand/key split; different actor observations may share latent deals; all joint test contexts retained'}
    packet=folder/'frozen_models.json'
    if packet.exists():assert json.loads(packet.read_text())==final
    else:save(packet,final)
    return out
