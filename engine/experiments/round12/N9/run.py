"""Frozen new N9 recipe, incremental states, held evaluation after allfits."""
from n9lib import *
SOURCES=[H/p for p in ('n9lib.py','run.py','evaluate.py','PROTOCOL.md','preflight.py','self_check.py')]+[R/'learning.py',R/'common.py']+[R/'N2'/p for p in ('features.py','corpus.py','export.py','joint.py','reference.py','bid_premise.py')]+[ENGINE/'judgement/cfr/toy.py',ENGINE/'judgement/cfr/heuristics.py']
def params(path):
    d=load(path);return [d[f'p{i}'] for i in range(len(d))]
def guard(sources):check();assert all(sha(Path(p))==v for p,v in sources.items()),'Source drift; stop'
def fit(tag,config,loss,seed,rs,receipt,sources,starttime):
    ci=TAGS.index(tag);ai=0 if config=='summary_linear' else 1;isrich=config=='rich_mlp';folder=O/tag/'models'/f'{config}_{loss}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    done=folder/'frozen.json'
    if done.exists():
        r=json.loads(done.read_text());assert r['sources']==sources and all(sha(folder/f'selected{b}.npz')==r['models'][str(b)]['sha256'] for b in (20,80));return
    tr=[r for r in rs if r.split==0];va=[r for r in rs if r.split==1];meanreach=np.mean([r.count for r in tr]);p=initialize(ai,[BASE,DOMAIN,ci,ai,seed,0]);opt=Adam(p);first=1;best={20:None,80:None};epochs={20:None,80:None};pointer=folder/'state.json'
    if pointer.exists():
        meta=json.loads(pointer.read_text());assert sha(folder/meta['state'])==meta['sha256'];z=load(folder/meta['state']);p=[z[f'p{i}'] for i in range(len(p))];opt=Adam(p);opt.m=[z[f'm{i}'] for i in range(len(p))];opt.v=[z[f'v{i}'] for i in range(len(p))];opt.t=int(z['step']);first=meta['epoch']+1;best={int(k):v for k,v in meta['best'].items()};epochs={int(k):v for k,v in meta['epochs'].items()}
        for budget in (20,80):
            if epochs[budget] is not None:npz(folder/f'selected{budget}.npz',**load(folder/f'candidate_{epochs[budget]}.npz'))
    for epoch in range(first,81):
        guard(sources);tick=time.perf_counter();order=np.random.default_rng([BASE,DOMAIN,ci,ai,seed,1,epoch]).permutation(len(tr))
        for begin in range(0,len(order),128):
            check();batch=[tr[j] for j in order[begin:begin+128]];_,gr=loss_gradient(batch,p,loss,isrich,float(meanreach));opt.step(p,gr)
        val=None
        if epoch in EPOCHS:
            val=evaluate(va,p,isrich)[0];npz(folder/f'candidate_{epoch}.npz',**{f'p{i}':v for i,v in enumerate(p)})
            for budget in (20,80):
                if epoch<=budget and (best[budget] is None or val<best[budget]):best[budget]=val;epochs[budget]=epoch;npz(folder/f'selected{budget}.npz',**{f'p{i}':v for i,v in enumerate(p)})
        state=folder/f'epoch_{epoch}.npz';npz(state,step=opt.t,**{f'p{i}':v for i,v in enumerate(p)},**{f'm{i}':v for i,v in enumerate(opt.m)},**{f'v{i}':v for i,v in enumerate(opt.v)})
        save(pointer,{'epoch':epoch,'state':state.name,'sha256':sha(state),'best':best,'epochs':epochs})
        with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,'val_joint_KL':val,'seconds':time.perf_counter()-tick,'utc':stamp()})+'\n')
        save(O/'progress.json',{'tag':tag,'config':config,'loss':loss,'seed':seed,'epoch':epoch,'utc':stamp()})
        gate=O/'resource_gate.json'
        if not gate.exists():
            assert (tag,config,loss,seed,epoch)==('S3R4n3_t-1','rich_mlp','joint',0,1)
            seconds=time.perf_counter()-tick;projected=1.5*seconds*120*80+900;remaining=float(os.environ['JUDGEMENT_JOB_DEADLINE'])-time.time()
            save(gate,{'utc':stamp(),'first_largeNT_rich_joint_epoch_seconds':seconds,'projected_remaining_seconds':projected,'remaining_window_seconds':remaining,'fits_in_window':projected<remaining,'scope':'timing-only conservative worst-condition extrapolation, not guaranteed; actual epoch1 retained'})
            if projected>remaining:raise RuntimeError('Resource-only forecast failed; preserve epoch1/cache, no science retuning')
    save(done,{'tag':tag,'config':config,'loss':loss,'seed':seed,'base':BASE,'domain':DOMAIN,'sources':sources,'data_sha256':receipt['data_sha256'],'models':{str(b):{'selected_epoch':epochs[b],'validation_joint_KL':best[b],'sha256':sha(folder/f'selected{b}.npz')} for b in (20,80)},'utc':stamp()});log(H,'Frozen '+folder.name+'/'+tag)
def main():
    O.mkdir(exist_ok=True);assert np.__version__=='2.5.4';sources={str(p):sha(p) for p in SOURCES};guard(sources);fp=O/'sources.json'
    if fp.exists():assert json.loads(fp.read_text())==sources
    else:save(fp,sources)
    start=time.perf_counter();datasets={tag:build(tag,sources) for tag in TAGS}
    jobs=[(t,c,l,s) for t in TAGS for c in CONFIGS for l in LOSSES for s in range(5)];pilot=('S3R4n3_t-1','rich_mlp','joint',0);jobs.remove(pilot);jobs.insert(0,pilot)
    for tag,c,l,s in jobs:fit(tag,c,l,s,*datasets[tag],sources,start)
    receipts=list(O.glob('*/models/*/frozen.json'));assert len(receipts)==120
    save(O/'training_complete.json',{'utc':stamp(),'n_trajectories':120,'n_selected_models':240,'receipts':{str(p.relative_to(O)):sha(p) for p in receipts},'sources':sources})
    sys.path.insert(0,str(H))
    from evaluate import main as evaluate_all
    evaluate_all()
if __name__=='__main__':main()
