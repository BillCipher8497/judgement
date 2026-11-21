from n10lib import *
def fit(train,val,sources,registry):
    target={a:targets(train['q'],a) for a in ARMS};weights=train['hit']/train['hit'].mean();vmean=val['q'].mean(1)*val['hit'][:,None];rows=np.arange(len(vmean));out={a:[] for a in ARMS}
    for arm in ARMS:
        for seed in range(5):
            folder=O/'models'/f'{arm}_s{seed}';folder.mkdir(parents=True,exist_ok=True);receipt=folder/'frozen.json'
            if receipt.exists():
                r=json.loads(receipt.read_text());assert r['sources']==sources and sha(folder/'selected.npz')==r['model_sha256'];z=load(folder/'selected.npz');out[arm].append({'params':[z[f'p{i}'] for i in range(len(z))],'alpha':r['alpha']});continue
            registry.add([FITBASE,DOMAIN,20,seed]);p=init(seed);opt=Adam(p);first=1;best=None;be=None;ba=None;pointer=folder/'state.json'
            if pointer.exists():
                m=json.loads(pointer.read_text());assert sha(folder/m['state'])==m['sha256'];z=load(folder/m['state']);p=[z[f'p{i}'] for i in range(len(p))];opt=Adam(p);opt.t=int(z['step']);opt.m=[z[f'm{i}'] for i in range(len(p))];opt.v=[z[f'v{i}'] for i in range(len(p))];first=m['epoch']+1;best,be,ba=m['best'],m['best_epoch'],m['alpha']
                if be is not None:npz(folder/'selected.npz',**load(folder/f'candidate_{be}.npz'))
            for epoch in range(first,51):
                check();assert all(sha(Path(k))==v for k,v in sources.items());address=[FITBASE,DOMAIN,21,seed,epoch];registry.add(address);order=np.random.default_rng(address).permutation(len(train['x']))
                for start in range(0,len(order),128):
                    ix=order[start:start+128];_,gr=grad(train['x'][ix],train['base'][ix],target[arm][ix],weights[ix],p,arm);opt.step(p,gr)
                trials=[]
                if epoch in EPOCHS:
                    npz(folder/f'candidate_{epoch}.npz',**{f'p{i}':v for i,v in enumerate(p)})
                    f=forward(val['x'],p)[0]
                    for alpha in ALPHAS:
                        a=(val['base']+alpha*f).argmax(1);score=float(vmean[rows,a].mean());trials.append({'alpha':alpha,'validation_rollout_score':score})
                        if best is None or score>best:best,be,ba=score,epoch,alpha;npz(folder/'selected.npz',**{f'p{i}':v for i,v in enumerate(p)})
                state=folder/f'epoch_{epoch}.npz';npz(state,step=opt.t,**{f'p{i}':v for i,v in enumerate(p)},**{f'm{i}':v for i,v in enumerate(opt.m)},**{f'v{i}':v for i,v in enumerate(opt.v)})
                save(pointer,{'epoch':epoch,'state':state.name,'sha256':sha(state),'best':best,'best_epoch':be,'alpha':ba})
                with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,'trials':trials,'utc':stamp()})+'\n')
                save(O/'progress.json',{'stage':'fit','arm':arm,'seed':seed,'epoch':epoch,'utc':stamp()})
            save(receipt,{'arm':arm,'seed':seed,'base':FITBASE,'domain':DOMAIN,'sources':sources,'selected_epoch':be,'alpha':ba,'validation_rollout_score':best,'model_sha256':sha(folder/'selected.npz'),'data_freeze_sha256':sha(O/'data_frozen.json'),'utc':stamp()});z=load(folder/'selected.npz');out[arm].append({'params':[z[f'p{i}'] for i in range(len(z))],'alpha':ba});log(H,f'Frozen {arm}/s{seed},epoch{be},alpha{ba}')
    receipts=list((O/'models').glob('*/frozen.json'));assert len(receipts)==10
    save(O/'models/frozen_models.json',{'utc':stamp(),'receipts':{str(p.relative_to(O)):sha(p) for p in receipts},'sources':sources,'scope':'Both five-fit ensembles frozen before fixtures/held/games; no replacement after gates'})
    return out
