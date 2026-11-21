"""Separate raw/label/model/CI checks, explicitly scoped rather than external."""
from common import *
from observables import restore,encode,baseline
from judgement.sampling import HandSampler
from reference import prefix,move_scores,generated_context,full_game
from data import load,partition
from fit import CANDIDATES,ALPHAS,read_params
import math
def independent_forward(x,p):
    y=x
    for k in range(0,len(p),2):
        y=np.add(np.matmul(y,p[k]),p[k+1])
        if k+2<len(p):y=np.tanh(y)
    return y
def independent_interval(values,strata):
    # Use Python sums/residuals, not the production np.var path.
    n=len(values);mean=math.fsum(map(float,values))/n;variance=0.
    for s in sorted(set(map(int,strata))):
        v=[float(x) for x,t in zip(values,strata) if t==s];m=math.fsum(v)/len(v);variance+=(len(v)/n)**2*math.fsum((x-m)**2 for x in v)/(len(v)*(len(v)-1))
    radius=1.959963984540054*math.sqrt(variance);return [mean-radius,mean+radius]
def labels(deadline):
    result={};handsets={}
    for name in COUNTS:
        data=load(name);assert np.isin(data['q'],(0,1)).all();handsets[name]=set(map(int,data['original_hand']));direct=0;total=0
        selected=set(range(len(data['g'])))
        for p in sorted((O/'data'/name).glob('*.npz')):
            check_time(deadline)
            with np.load(p) as source:a={k:source[k] for k in source.files}
            meta=json.loads(p.with_suffix('.json').read_text())
            for row,o in enumerate(meta['observations']):
                g=int(a['g'][row]);hero=g%4;assert o['seat']==hero and a['stratum'][row]==g%96 and o['hand']==int(a['worlds'][row,0,hero]);assert partition(int(a['original_hand'][row]))==(0 if name in ('pilot','train') else 1 if name=='validation' else 2)
                own_played=sum(1<<c for seat,c in o['history'] if seat==hero);assert (own_played|o['hand'])==int(a['original_hand'][row]);obs=restore(o);legal=o['hand'];base=baseline().play(obs,legal,np.random.default_rng(42));x,b,cards=encode(obs,legal,base);np.testing.assert_array_equal(x,a['x'][row]);np.testing.assert_array_equal(b,a['base'][row]);assert cards==list(a['cards'][row]);assert 10+dict(o['bids'])[hero]==a['hit'][row]
                for w in a['worlds'][row]:assert prefix(o,w)
                if g in selected:
                    original=generated_context([BASE,DOMAIN,STAGES[name],g,int(a['attempt'][row]),3],int(a['round_index'][row]),o);assert original==int(a['original_hand'][row])
                    address=[BASE,DOMAIN,STAGES[name],g,int(a['attempt'][row]),5];rng=np.random.default_rng(np.random.SeedSequence(address));sampler=HandSampler(obs)
                    for k in range(16):
                        check_time(deadline);hidden=sampler.sample(rng);world=[hidden.get(q,o['hand']) for q in range(4)];np.testing.assert_array_equal(world,a['worlds'][row,k]);scores=move_scores(o,world);got=np.array([scores[c] for c in cards])/a['hit'][row];np.testing.assert_array_equal(got,a['q'][row,k]);direct+=1
                total+=1
            save(O/'label_verification_progress.json',{'stage':name,'raw_contexts':total,'independent_MaxN_worlds':direct,'utc':utc()})
        result[name]={'all_raw_features_and_sample_prefixes':total,'independent_manual_MaxN_worlds':direct,'sampler_replay_scope':'Every saved context/seed prefix and all16worlds; same defined structural sampler, independent solver/rules'}
    assert not (handsets['train']&handsets['validation'] or handsets['train']&handsets['held'] or handsets['validation']&handsets['held'])
    return result
def models(deadline):
    data=load('validation');q=data['q'].mean(1)*data['hit'][:,None];frozen=json.loads((O/'models/frozen_models.json').read_text());means=[]
    for path,h in {**frozen['candidate_hashes'],**frozen['data_chunks_sha256']}.items():assert digest(O/path)==h
    for arm,packet in enumerate(frozen['arms']):
        chosen=[]
        for seed,record in enumerate(packet['models']):
            d=O/'models'/ARMS[arm]/f's{seed}';candidates=[]
            for epoch in CANDIDATES:
                check_time(deadline);p=read_params(d/f'epoch{epoch:02d}.npz');z=independent_forward(data['x'],p)
                for alpha in ALPHAS:
                    score=(1-alpha)*data['base']+alpha*z if arm==7 else data['base']+alpha*z;action=score.argmax(1);value=math.fsum(float(q[i,a]) for i,a in enumerate(action))/len(action);candidates.append((value,epoch,alpha))
            best=min(candidates,key=lambda c:(-c[0],c[1],c[2]));assert best[1:]==(record['epoch'],record['alpha']);assert abs(best[0]-record['teacher_score'])<1e-11
            p=read_params(O/record['path']);assert digest(O/record['path'])==record['sha256'];z=independent_forward(data['x'],p);alpha=record['alpha'];chosen.append((1-alpha)*data['base']+alpha*z if arm==7 else data['base']+alpha*z)
        action=np.mean(chosen,axis=0).argmax(1);value=math.fsum(float(q[i,a]) for i,a in enumerate(action))/len(action);means.append(value);assert abs(value-packet['validation_teacher_score'])<1e-11
    assert max(range(8),key=lambda a:means[a])==frozen['best_arm_index'];return {'models':40,'candidates_per_fit':16,'best_student_checked':True}
def games(deadline):
    result={};frozen=json.loads((O/'models/frozen_models.json').read_text());best=frozen['best_arm_index'];parameters=[[read_params(O/r['path']) for r in a['models']] for a in frozen['arms']]
    for stage,base in (('screen',2130000000),('confirmation',2140000000)):
        summary=json.loads((O/f'{stage}_summary.json').read_text());raw=[json.loads(p.read_text()) for p in sorted((O/'games'/stage).glob('*.json'))];assert len(raw)==2000
        scores=[];wins=[];table=[]
        for g,block in enumerate(raw):
            check_time(deadline);assert block['g']==g and block['hero']==g%4 and block['seed']==base+g and len(block['arms'])==9
            for a,row in enumerate(block['arms']):
                assert row['arm']==a and row['hero']==g%4 and row['seed']==base+g and len(row['rounds'])==25
                total=[0]*4
                for r in row['rounds']:
                    assert sum(r['tricks'])==r['n'] and sum(r['bids'])!=r['n']
                    assert sum(r['scores'])<=30+r['n']
                    for p in range(4):
                        value=10+r['bids'][p] if r['bids'][p]==r['tricks'][p] else 0;assert value==r['scores'][p];total[p]+=value
                assert total==row['totals'] and total[g%4]==row['hero_score'] and int(total[g%4]==max(total))==row['hero_ties_as_win']
            scores.append([a['hero_score'] for a in block['arms']]);wins.append([a['hero_ties_as_win'] for a in block['arms']]);table.append([sum(a['totals']) for a in block['arms']])
        scores=np.asarray(scores);wins=np.asarray(wins);table=np.asarray(table);strata=np.arange(2000)%4
        contrasts=[('PRIMARY_set-minus-hard',2,0),('PRIMARY_best-minus-baseline',best,8)]+[(name+'-minus-baseline',i,8) for i,name in enumerate(ARMS)]+[('uniform-minus-hard',1,0),('set-minus-uniform',2,1),('set-minus-linear-set',2,6),('hard-minus-linear-hard',0,5),('set-minus-no-offset-set',2,7)]
        for name,a,b in contrasts:
            for metric,rawarray in (('score',scores),('ties_as_win',wins),('table_score',table),('opponents_mean_score',(table-scores)/3)):
                got=independent_interval(rawarray[:,a]-rawarray[:,b],strata);np.testing.assert_allclose(got,summary['contrasts'][name][metric]['ci95'],rtol=0,atol=1e-11)
        interaction=(scores[:,2]-scores[:,0])-(scores[:,6]-scores[:,5]);np.testing.assert_allclose(independent_interval(interaction,strata),summary['contrasts']['exploratory_label_gain_interaction_MLP-minus-linear']['score']['ci95'],rtol=0,atol=1e-11)
        replays=0
        for g in (*range(4),*range(1996,2000)):
            for arm in range(9):
                check_time(deadline)
                def choose(info,legal,rng):
                    base=baseline().play(info,legal,rng)
                    if arm==8:return base
                    from observables import active
                    if not active(info,legal):return base
                    x,b,cards=encode(info,legal,base);pred=[]
                    for seed,p in enumerate(parameters[arm]):
                        f=independent_forward(x[None,:],p)[0];alpha=frozen['arms'][arm]['models'][seed]['alpha'];pred.append((1-alpha)*b+alpha*f if arm==7 else b+alpha*f)
                    return cards[int(np.mean(pred,axis=0).argmax())]
                replay=full_game(base+g,g%4,choose);assert replay['totals']==raw[g]['arms'][arm]['totals'] and replay['rounds']==raw[g]['arms'][arm]['rounds'];replays+=1
        result[stage]={'n_complete_blocks':2000,'all_round_scores_hook_ties_and_paired_CIs_checked':True,'independent_deal_rules_and_forward_full_game_replays':replays}
    return result
def held_report(deadline):
    data=load('held');frozen=json.loads((O/'models/frozen_models.json').read_text());report=json.loads((O/'held_label_report.json').read_text());q=data['q'];mean=q.mean(1)*data['hit'][:,None];best=mean.max(1);rows=np.arange(len(q));strata=data['g']%96
    hardfirst=(q[:,:,0]>=q[:,:,1]).mean(1);uniformfirst=((q[:,:,0]>q[:,:,1])+.5*(q[:,:,0]==q[:,:,1])).mean(1);membership=np.column_stack([(q[:,:,0]>=q[:,:,1]).mean(1),(q[:,:,1]>=q[:,:,0]).mean(1)]);decoders={'hard':np.column_stack([hardfirst,1-hardfirst]),'uniform':np.column_stack([uniformfirst,1-uniformfirst]),'set':membership,'baseline':data['base']};regrets={}
    for arm,name in enumerate(ARMS):
        check_time(deadline);pred=[]
        for r in frozen['arms'][arm]['models']:
            f=independent_forward(data['x'],read_params(O/r['path']));alpha=r['alpha'];pred.append((1-alpha)*data['base']+alpha*f if arm==7 else data['base']+alpha*f)
        decoders['student-'+name]=np.mean(pred,axis=0)
    for name,z in decoders.items():regrets[name]=best-mean[rows,z.argmax(1)]
    for a,b in (('set','hard'),('set','uniform'),('student-set','student-hard')):regrets[a+'-minus-'+b]=regrets[a]-regrets[b]
    for name,values in regrets.items():
        assert report[name]['n']==len(values) and abs(report[name]['mean']-math.fsum(map(float,values))/len(values))<1e-11;np.testing.assert_allclose(independent_interval(values,strata),report[name]['ci95'],rtol=0,atol=1e-11)
    return {'n_contexts':len(q),'all_label_and_student_surrogate_costs_and_paired_CIs_checked':True,'scope':'Same finite-K surrogate, not actual TA continuation or true posterior'}
def main():
    deadline=float(os.environ['N3_AUDIT_DEADLINE']);freeze=json.loads((O/'source_freeze.json').read_text());verify_sources(freeze)
    result={'labels':labels(deadline),'models':models(deadline),'held_report':held_report(deadline),'games':games(deadline),'status':'self-checked(Codex); pending external audit','limitations':'Every saved context and sampled teacher root checked by manual solver/rules/seed-prefix path; this does not enumerate the full52card chance posterior. Sampler and fixed baseline policy functions reused as definitions. All game totals/CIs and144manual game/deal/forward replays checked; not every game independently re-simulated.','utc':utc()};save(O/'separate_verification.json',result);print(json.dumps(result))
if __name__=='__main__':main()
