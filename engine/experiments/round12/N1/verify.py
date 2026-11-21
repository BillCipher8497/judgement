"""Separate N1 raw-label/selection/prediction/log-loss checks after completion."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,math,statistics,sys
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;sys.path[:0]=[str(H),str(H.parent)]
import run as experiment
from common import sha,save,stamp
def forward(x,mask,prop,params):
    out=np.empty_like(prop)
    for start in range(0,len(x),4096):
        xx=x[start:start+4096].reshape(-1,35);hidden=xx@params[0]+params[1];hidden[hidden<0]=0
        score=(hidden@params[2]+params[3]).reshape(-1,4)
        lp=np.where(mask[start:start+4096],np.log(np.maximum(prop[start:start+4096],1e-300)),-np.inf)
        score+=lp;score-=score.max(1,keepdims=True);prob=np.exp(score);prob/=prob.sum(1,keepdims=True);out[start:start+len(prob)]=prob
    return out
def loss(p,y,offset):
    ll=-np.log(p[np.arange(len(y)),y]);return np.array([math.fsum(ll[a:b])/int(b-a) for a,b in zip(offset[:-1],offset[1:])])
def main():
    O=experiment.O;experiment.check_sources();freeze=json.loads((O/'source_freeze.json').read_text())
    failure=json.loads((O/'scope_failure.json').read_text()) if (O/'scope_failure.json').exists() else None
    if failure and failure['reason']=='resource-only pilot gate':
        assert not (O/'confirmation').exists() and not (O/'frozen_models.json').exists()
        experiment.data_manifest('pilot')
        save(O/'separate_verification.json',{'status':'declared resource scope failure; ready for Claude','failure':failure,'no_confirmation_used':True,'utc':stamp()});return
    models=json.loads((O/'frozen_models.json').read_text())
    assert models['source_freeze_sha256']==sha(O/'source_freeze.json')
    for rel,h in models['model_sha256'].items():assert sha(O/rel)==h
    for stage,h in models['data_manifests'].items():assert experiment.data_manifest(stage)==h
    counts={}
    for stage in (('train','validation','confirmation') if models['confirmation_allowed'] else ('train','validation')):
        data,records=experiment.load(stage);assert [q['g'] for q in records]==list(range(experiment.STAGES[stage][2]))
        for index,q in enumerate(records):
            a,b=data['offset'][index:index+2];o=q['observation'];assert q['hero']==index%4 and o['seat']==index%4
            seen=o['hand']
            for _,c in o['history']:seen|=1<<c
            cards=[c for c in range(52) if not (seen>>c)&1];assert cards==q['cards'] and len(cards)==b-a
            hands=q['initial_hands'][:]
            for p,c in o['history']:
                assert hands[p]&(1<<c);hands[p]^=1<<c
            assert hands[q['hero']]==o['hand']
            xx,mm,pp,cc=experiment.encode(o,q['table'])
            np.testing.assert_array_equal(xx,data['X'][a:b]);np.testing.assert_array_equal(mm,data['M'][a:b]);np.testing.assert_allclose(pp,data['P'][a:b],rtol=0,atol=1e-14)
            opps=[(q['hero']+k)%4 for k in (1,2,3)];truth=[]
            for c in cards:
                holders=[j for j,p in enumerate(opps) if hands[p]&(1<<c)];assert len(holders)<=1
                truth.append(holders[0] if holders else 3)
            np.testing.assert_array_equal(truth,data['Y'][a:b]);assert data['M'][np.arange(a,b),data['Y'][a:b]].all()
            assert np.allclose(data['P'][a:b].sum(1),1)
            if o['n']==13:assert not data['M'][a:b,3].any() and not np.any(data['Y'][a:b]==3)
        counts[stage]=len(records)
    val,_=experiment.load('validation');computed={}
    for variant,model in models['models'].items():
        x=val['X'].copy()
        if variant=='bids':x[:,:,[18,19,20,21,22,26,27,28,30]]=0
        selected=[]
        for seed in range(5):
            folder=O/f'{variant}_s{seed}';receipt=json.loads((folder/'frozen.json').read_text());choices=[]
            trace=[json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()];assert [q['epoch'] for q in trace]==list(range(1,51))
            for epoch in (10,25,50):
                saved=dict(np.load(folder/f'candidate_{epoch}.npz'));p=[saved[f'p{i}'] for i in range(4)];pred=forward(x,val['M'],val['P'],p)
                mean=float(loss(.999*pred+.001*val['P'],val['Y'],val['offset']).mean());recorded=next(q['validation_loss'] for q in trace if q['epoch']==epoch)
                assert abs(mean-recorded)<1e-11;choices.append((recorded,epoch))
                if epoch==receipt['selected_epoch']:
                    selected.append(pred);saved_selected=dict(np.load(folder/'selected.npz'))
                    for i,parameter in enumerate(p):np.testing.assert_array_equal(parameter,saved_selected[f'p{i}'])
            assert min(choices)==(receipt['validation_loss'],receipt['selected_epoch'])
        ensemble=np.mean(selected,axis=0);choices=[]
        for alpha in (0.,.25,.5,1.):
            value=float(loss((1-.999*alpha)*val['P']+.999*alpha*ensemble,val['Y'],val['offset']).mean());choices.append((value,alpha))
        minimum=min(v[0] for v in choices);assert next(v for v in choices if v[1]==model['alpha'])[0]<=minimum+1e-11
        computed[variant]=True
        if variant=='full':
            chosen=(1-.999*model['alpha'])*val['P']+.999*model['alpha']*ensemble
            difference=loss(chosen,val['Y'],val['offset'])-loss(val['P'],val['Y'],val['offset'])
            parts=[difference[np.arange(len(difference))%176==s] for s in range(176)]
            point=statistics.mean([statistics.mean(v) for v in parts]);half=1.96*math.sqrt(math.fsum(statistics.variance(v)/len(v) for v in parts))/176
            assert abs(point-models['selection_gate']['mean'])<1e-11
            assert models['confirmation_allowed']==(model['alpha']>0 and models['selection_gate']['ci95'][1]<=0)
    if not models['confirmation_allowed']:
        assert not (O/'confirmation').exists()
        save(O/'separate_verification.json',{'status':'declared selection scope failure; ready for Claude','raw_label_rounds':counts,'selection_verified':computed,'no_confirmation_used':True,'utc':stamp()});return
    test,records=experiment.load('confirmation');stored=dict(np.load(O/'confirmation_predictions.npz'));summary=json.loads((O/'summary.json').read_text());losses={}
    for name,p in stored.items():
        assert np.allclose(p.sum(1),1) and np.all(p[~test['M']]==0);losses[name]=loss(p,test['Y'],test['offset'])
        np.testing.assert_allclose(losses[name],summary['raw_decision_losses'][name],rtol=0,atol=1e-11)
    for variant,model in models['models'].items():
        x=test['X'].copy()
        if variant=='bids':x[:,:,[18,19,20,21,22,26,27,28,30]]=0
        probs=[]
        for rel in model['selected_models']:
            a=dict(np.load(O/rel));probs.append(forward(x,test['M'],test['P'],[a[f'p{i}'] for i in range(4)]))
        pred=(1-.999*model['alpha'])*test['P']+.999*model['alpha']*np.mean(probs,axis=0)
        np.testing.assert_allclose(pred,stored['net_'+variant],rtol=0,atol=1e-11)
    delta=losses['net_full']-losses['bi'];groups=[delta[np.arange(len(delta))%176==s] for s in range(176)]
    mean=statistics.mean([statistics.mean(v) for v in groups]);half=1.96*math.sqrt(math.fsum(statistics.variance(v)/len(v) for v in groups))/176
    np.testing.assert_allclose([mean-half,mean+half],summary['primary_net_minus_BI']['ci95'],rtol=0,atol=1e-11)
    save(O/'separate_verification.json',{'status':'ready-for-Claude-external','raw_label_rounds':counts,'both_model_selections':computed,
        'primary_recomputed':{'mean':mean,'ci95':[mean-half,mean+half],'n_decisions':len(delta)},'utc':stamp(),
        'limitations':'Independent numerical forward/label reconstruction; validated core supplies original histories. ECE bootstrap and fresh sampler/game-prefix replay still need external checking.'})
if __name__=='__main__':main()
