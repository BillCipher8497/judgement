"""Fixed, paired optimization schedules; validation-only selection."""
from common import *
from network import init,gradient,targets,scores,policy_scores,Adam
CANDIDATES=(5,10,25,50);ALPHAS=(0.,.25,.5,1.)
def read_params(path):
    a=np.load(path);return [a[f'p{i}'].copy() for i in range(int(a['nparams']))]
def write_params(path,p,**extra):arrays(path,nparams=len(p),**{f'p{i}':v for i,v in enumerate(p)},**extra)
def fit(train,val,freeze,registry,deadline):
    registry.fit();folder=O/'models';frozen=folder/'frozen_models.json';N=len(train['g']);labels=[targets(train['q'],a) for a in range(8)]
    if frozen.exists():
        value=json.loads(frozen.read_text());assert value['source_freeze_sha256']==digest(O/'source_freeze.json')
        for arm in value['arms']:
            for r in arm['models']:assert digest(O/r['path'])==r['sha256']
        return value
    result=[]
    for arm,name in enumerate(ARMS):
        rows=[]
        for seed in range(5):
            check_time(deadline);verify_sources(freeze);d=folder/name/f's{seed}';d.mkdir(exist_ok=True,parents=True);selected=d/'selected.json'
            if selected.exists():
                r=json.loads(selected.read_text());assert digest(O/r['path'])==r['sha256'];rows.append(r);continue
            p=init(arm,seed);adam=Adam(p);first=1
            if (d/'checkpoint.json').exists():
                receipt=json.loads((d/'checkpoint.json').read_text());checkpoint=d/receipt['checkpoint_name'];assert receipt['checkpoint_sha256']==digest(checkpoint) and receipt['source_freeze_sha256']==digest(O/'source_freeze.json')
                a=np.load(checkpoint);p=[a[f'p{i}'].copy() for i in range(len(p))];adam=Adam(p);adam.m=[a[f'm{i}'].copy() for i in range(len(p))];adam.v=[a[f'v{i}'].copy() for i in range(len(p))];adam.t=int(a['adam_t']);first=int(a['epoch'])+1
            for epoch in range(first,51):
                check_time(deadline);verify_sources(freeze);rng=np.random.default_rng(np.random.SeedSequence([BASE,DOMAIN,21,seed,epoch]));order=rng.permutation(N);loss=0.
                for start in range(0,N,128):
                    check_time(deadline);ids=order[start:start+128];value,grads=gradient(train['x'][ids],train['base'][ids],labels[arm][ids],train['hit'][ids],p,arm);adam.step(p,grads);loss+=value*len(ids)/N
                if epoch in CANDIDATES:write_params(d/f'epoch{epoch:02d}.npz',p)
                checkpoint=d/f'checkpoint{epoch:02d}.npz';write_params(checkpoint,p,epoch=epoch,adam_t=adam.t,**{f'm{i}':v for i,v in enumerate(adam.m)},**{f'v{i}':v for i,v in enumerate(adam.v)})
                save(d/'checkpoint.json',{'epoch':epoch,'training_loss':loss,'checkpoint_name':checkpoint.name,'checkpoint_sha256':digest(checkpoint),'source_freeze_sha256':digest(O/'source_freeze.json')});status(f'N3fit {name} seed{seed} epoch{epoch}/50')
            candidates=[];meanq=val['q'].mean(1)*val['hit'][:,None]
            for epoch in CANDIDATES:
                params=read_params(d/f'epoch{epoch:02d}.npz')
                for alpha in ALPHAS:
                    action=scores(val['x'],val['base'],params,arm,alpha).argmax(1);values=meanq[np.arange(len(action)),action];estimate=interval(values,val['stratum']);candidates.append({'epoch':epoch,'alpha':alpha,'teacher_score':estimate['mean'],'teacher_score_ci95':estimate['ci95'],'n_validation_contexts':len(action),'interval_scope':'descriptive reused selection contexts, not an unbiased selected-model interval'})
            best=min(candidates,key=lambda r:(-r['teacher_score'],r['epoch'],r['alpha']));path=d/f'epoch{best["epoch"]:02d}.npz'
            record={**best,'seed':seed,'path':str(path.relative_to(O)).replace('\\','/'),'sha256':digest(path),'n_validation_contexts':len(val['g']),'source_freeze_sha256':digest(O/'source_freeze.json')}
            save(d/'candidates.json',candidates);save(selected,record);rows.append(record)
        models=[{'params':read_params(O/r['path']),'alpha':r['alpha']} for r in rows];action=policy_scores(val['x'],val['base'],models,arm).argmax(1);q=val['q'].mean(1)*val['hit'][:,None];estimate=interval(q[np.arange(len(q)),action],val['stratum']);result.append({'name':name,'models':rows,'validation_teacher_score':estimate['mean'],'validation_teacher_score_descriptive':estimate})
    best=max(range(8),key=lambda a:result[a]['validation_teacher_score'])
    packet={'arms':result,'best_arm_index':best,'best_arm_name':ARMS[best],'n_validation_contexts':len(val['g']),'candidate_hashes':{str(p.relative_to(O)):digest(p) for p in sorted(folder.glob('*/s*/epoch*.npz'))},'data_chunks_sha256':{str(p.relative_to(O)):digest(p) for stage in ('train','validation') for p in sorted((O/'data'/stage).glob('*.npz'))},'source_freeze_sha256':digest(O/'source_freeze.json'),'selection':'validation surrogate only; frozen before held labels/screen/confirmation','utc':utc()};save(frozen,packet);return packet
def load_models(frozen):
    allmodels=[]
    for a in frozen['arms']:
        models=[]
        for r in a['models']:
            assert digest(O/r['path'])==r['sha256'];models.append({'params':read_params(O/r['path']),'alpha':r['alpha']})
        allmodels.append(models)
    return allmodels
