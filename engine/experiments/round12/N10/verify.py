"""Separate full prefix/private-transition/selection/gate/CI audit, no refitting."""
from n10lib import *
from reference import prefix,rollout
import math
def fresh_ci(values,strata):
    values=list(map(float,values));n=len(values);mean=math.fsum(values)/n;var=0
    for s in sorted(set(map(int,strata))):
        a=[v for v,k in zip(values,strata) if k==s];m=math.fsum(a)/len(a);var+=(len(a)/n)**2*math.fsum((v-m)**2 for v in a)/(len(a)*(len(a)-1))
    h=1.959963984540054*math.sqrt(var);return mean,[mean-h,mean+h]
def compare(c,values,strata):
    m,b=fresh_ci(values,strata);assert c['n']==len(values) and abs(m-c['mean'])<1e-12 and np.allclose(c['ci95'],b,rtol=0,atol=1e-12)
def forward_fresh(x,p):
    a=x
    for k in range(0,len(p),2):
        a=np.einsum('ij,jk->ik',a,p[k])+p[k+1]
        if k+2<len(p):a=np.tanh(a)
    return a
def average(data,models):return data['base']+sum(m['alpha']*forward_fresh(data['x'],m['params']) for m in models)/5
def audit_data(name):
    folder=O/'data'/name;paths=sorted(folder.glob('*.npz'));assert paths;chunks=[];count=0;handset=set()
    for path in paths:
        check();meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['data_sha256'];a=load(path);assert meta['first_g']==count;assert meta['n']==len(a['g']);chunks.append(a)
        for j,g0 in enumerate(a['g']):
            check();g=int(g0);attempt=int(a['attempt'][j]);assert g==count;accepted=None
            for prior in range(attempt+1):
                candidate=prefix(name,g,prior)
                if prior<attempt:assert candidate is None,('Skippedearlier eligible context',name,g,prior)
                else:accepted=candidate
            assert accepted is not None;obs=meta['observations'][j]
            assert json.loads(json.dumps(accepted['observation']))==obs and int(a['original_hand'][j])==accepted['original_hand'];np.testing.assert_array_equal(accepted['x'],a['x'][j]);np.testing.assert_array_equal(accepted['base'],a['base'][j]);np.testing.assert_array_equal(accepted['cards'],a['cards'][j]);assert accepted['hit']==a['hit'][j] and accepted['round_index']==a['round_index'][j] and g%96==a['stratum'][j]
            handset.add(int(a['original_hand'][j]))
            if name!='fixtures':
                info=OBS.restore(obs);hero=obs['seat'];base=BASE;wa=[base,DOMAIN,STAGES[name],g,attempt,5];rng=np.random.default_rng(wa);sampler=HandSampler(info)
                np.testing.assert_array_equal(accepted['true_worlds'],a['true_worlds'][j]);truth={p:int(a['true_worlds'][j,p]) for p in range(4) if p!=hero};ta=[base,DOMAIN,STAGES[name],g,attempt,7]
                for k,card in enumerate(a['cards'][j]):assert rollout(obs,truth,int(card),hero,ta)==a['q'][j,0,k]*a['hit'][j]
                for w in range(16):
                    hidden=sampler.sample(rng);np.testing.assert_array_equal([hidden.get(p,info.hand) for p in range(4)],a['worlds'][j,w]);address=[base,DOMAIN,STAGES[name],g,attempt,w,6]
                    for k,card in enumerate(a['cards'][j]):
                        score=rollout(obs,hidden,int(card),hero,address);assert score==a['proxy_q'][j,w,k]*a['hit'][j]
            count+=1
        save(O/'verification_progress.json',{'stage':'data','name':name,'checked':count,'target':COUNTS[name],'utc':stamp()})
    assert count==COUNTS[name];data={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]};return data,handset
def main():
    sources=json.loads((O/'sources.json').read_text());assert all(sha(Path(p))==v for p,v in sources.items());stages={};hands={}
    names=('pilot',) if not(O/'models/frozen_models.json').exists() else tuple(COUNTS)
    for name in names:stages[name],hands[name]=audit_data(name)
    if names==('pilot',):
        failure=json.loads((O/'scope_failure.json').read_text());gate=json.loads((O/'resource_gate.json').read_text());assert not gate['accepted'] and failure['scientific_games']==0
        save(O/'verification.json',{'utc':stamp(),'scope':'resourcefailure pilot96fully independentlychecked, no fits/games','source_sha256':sha(Path(__file__)),'pending_external':True});return
    assert not(hands['train']&hands['validation'] or hands['train']&hands['held'] or hands['validation']&hands['held']);train=stages['train'];val=stages['validation'];held=stages['held'];fixtures=stages['fixtures'];mean=val['q'].mean(1)*val['hit'][:,None];rows=np.arange(len(mean));models={a:[] for a in ARMS}
    for arm in ARMS:
        for seed in range(5):
            check();folder=O/'models'/f'{arm}_s{seed}';f=json.loads((folder/'frozen.json').read_text());trials=[]
            for epoch in EPOCHS:
                z=load(folder/f'candidate_{epoch}.npz');p=[z[f'p{i}'] for i in range(len(z))];pred=forward_fresh(val['x'],p)
                for alpha in ALPHAS:
                    action=(val['base']+alpha*pred).argmax(1);score=float(mean[rows,action].mean());trials.append((score,epoch,alpha))
            chosen=min(trials,key=lambda r:(-r[0],r[1],r[2]));assert (f['selected_epoch'],f['alpha'])==chosen[1:] and abs(f['validation_rollout_score']-chosen[0])<1e-12
            z=load(folder/'selected.npz');p=[z[f'p{i}'] for i in range(len(z))];candidate=load(folder/f"candidate_{f['selected_epoch']}.npz");assert all(np.array_equal(v,candidate[k]) for k,v in z.items()) and sha(folder/'selected.npz')==f['model_sha256'];models[arm].append({'params':p,'alpha':f['alpha']})
    saved_gates=json.loads((O/'gates.json').read_text());passed=True
    for arm in ARMS:
        z=average(fixtures,models[arm]);saved=load(O/f'fixture_scores_{arm}.npz');np.testing.assert_allclose(z,saved['scores'],rtol=0,atol=2e-9);a=z.argmax(1);b=fixtures['base'].argmax(1);changed=a!=b;np.testing.assert_array_equal(changed,saved['changed']);np.testing.assert_array_equal(a,saved['actions']);marg=np.abs(z[:,0]-z[:,1]);g=saved_gates['arms'][arm];n=int(changed.sum());nh=len(set(fixtures['original_hand'][changed].tolist()));seats=set((fixtures['g'][changed]%4).tolist());active=n>=16 and nh>=8 and seats==set(range(4)) and bool(np.all(marg[changed]>=1e-6));assert n==g['changed_fixture_actions'] and active==g['activation_gate']
        radius=math.sqrt(math.log(40)/(2*len(a)));assert np.allclose(g['diagnostic_activation_rate']['ci95'],[max(0,n/len(a)-radius),min(1,n/len(a)+radius)],rtol=0,atol=1e-14)
        diff=mean[rows,average(val,models[arm]).argmax(1)]-mean[rows,val['base'].argmax(1)];bound=identity_bound(models[arm]);assert bound==g['global_identity_bound']
        if bound['global_baseline_identity_by_tanh_margin']:assert not np.any(diff) and g['validation_rollout_minus_baseline']['ci95'] is None;ni=True
        else:compare(g['validation_rollout_minus_baseline'],diff,val['stratum']);ni=fresh_ci(diff,val['stratum'])[1][0]>=-.05
        assert ni==g['selection_reused_validation_gate'] and g['pass']==(active and ni);passed=passed and active and ni
    assert passed==saved_gates['both_pass'];hmean=held['q'].mean(1)*held['hit'][:,None];hbest=hmean.max(1);ix=np.arange(len(hmean));report=json.loads((O/'held_report.json').read_text());regret={}
    regret['baseline']=hbest-hmean[ix,held['base'].argmax(1)]
    for arm in ARMS:
        regret[arm]=hbest-hmean[ix,average(held,models[arm]).argmax(1)];label=np.eye(2)[held['q'].argmax(2)].mean(1) if arm=='hard' else (held['q']==held['q'].max(2,keepdims=True)).mean(1);regret['label_'+arm]=hbest-hmean[ix,label.argmax(1)]
    for k,v in regret.items():compare(report[k],v,held['stratum'])
    for name,arm in (('proxy_hard','hard'),('proxy_set','set')):
        q=held['proxy_q'];target=np.eye(2)[q.argmax(2)].mean(1) if arm=='hard' else (q==q.max(2,keepdims=True)).mean(1);a=target.argmax(1)
        compare(report[name+'_actual_hindsight_regret'],hbest-hmean[ix,a],held['stratum']);compare(report[name+'_actual_score_minus_baseline'],hmean[ix,a]-hmean[ix,held['base'].argmax(1)],held['stratum'])
    if report['set_minus_hard_regret']['ci95'] is None:assert not np.any(regret['set']-regret['hard'])
    else:compare(report['set_minus_hard_regret'],regret['set']-regret['hard'],held['stratum'])
    nreplays=0
    if passed:
        from evaluate import Student
        for stage,base in (('screen',2200000000),('confirmation',2210000000)):
            folder=O/'games'/stage;raw=[json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))];assert len(raw)==1000
            for i,r in enumerate(raw):
                assert r['g']==i and r['seed']==base+i and r['hero']==i%4
                for arm,a in r['arms'].items():
                    assert len(a['rounds'])==25 and a['sources']==sources;totals=[sum(rr['scores'][p] for rr in a['rounds']) for p in range(4)];assert totals==a['totals'] and a['hero_score']==totals[i%4] and a['ties_as_win']==int(totals[i%4]==max(totals))
                    for rr in a['rounds']:
                        assert sum(rr['tricks'])==rr['n'] and sum(rr['bids'])!=rr['n'] and rr['scores']==[10+b if b==t else 0 for b,t in zip(rr['bids'],rr['tricks'])]
            for g0 in [4*k+s for k in (0,35,71,106,142,177,213,249) for s in range(4)]:
                for arm in ('baseline','hard','set'):
                    agents=OBS.agents();agents[g0%4]=Agent(ModelBidder(MODEL),OBS.baseline() if arm=='baseline' else Student(models[arm]));fresh=play_game(RULES,agents,seed=base+g0,record_trace=False);assert fresh.totals==raw[g0]['arms'][arm]['totals'];assert [list(r.scores) for r in fresh.rounds]==[r['scores'] for r in raw[g0]['arms'][arm]['rounds']];nreplays+=1
            summary=json.loads((O/f'{stage}_summary.json').read_text())
            for name,a,b in (('PRIMARY_set_minus_hard','set','hard'),('set_minus_baseline','set','baseline'),('hard_minus_baseline','hard','baseline')):
                for metric in ('hero_score','ties_as_win'):compare(summary['contrasts'][name][metric],[r['arms'][a][metric]-r['arms'][b][metric] for r in raw],np.arange(1000)%4)
    else:
        assert not(O/'games').exists();failure=json.loads((O/'scope_failure.json').read_text());assert failure['scientific_games']==0
    save(O/'verification.json',{'utc':stamp(),'all_new_data_prefixes_rejection_firstness_and_private_rollouts_replayed':True,'n_data_contexts':sum(COUNTS.values()),'all10fitselections_andbothgates_forward_CIs_match':True,'gate_pass':passed,'scientific_game_blocks_checked':2000 if passed else 0,'n_complete_game_replays':nreplays,'source_sha256':sha(Path(__file__)),'scope':'separate Codex source/chance/replay/math check; same fixed immutable TApolicy definitions, not external validation or new-game estimate; pending Claude'})
    log(H,'N10separate audit passed for allfixeddata/models/gates/heldmetrics'+(' andgames' if passed else '; prespecifiedgatefailure, no scientificgames')+'; pending external/local.')
if __name__=='__main__':main()
