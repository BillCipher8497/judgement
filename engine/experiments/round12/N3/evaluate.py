"""Held-label diagnostics and incremental full-ladder matched game blocks."""
from common import *
from observables import MODEL,RULES,agents,baseline,active,encode,detach
from network import policy_scores,targets
from judgement.bidding.model import Agent,ModelBidder
from judgement.engine import play_game
class Student:
    def __init__(self,models,arm):self.base=baseline();self.models=models;self.arm=arm;self.active_count=0
    def play(self,info,legal,rng):
        if not active(info,legal):return self.base.play(info,legal,rng)
        detached=detach(info);base_card=self.base.play(detached,legal,rng);x,b,cards=encode(detached,legal,base_card);z=policy_scores(x[None,:],b[None,:],self.models,self.arm);self.active_count+=1;return cards[int(z.argmax(1)[0])]
def label_report(data,models,frozen):
    q=data['q'];mean=q.mean(1)*data['hit'][:,None];best=mean.max(1);rows=np.arange(len(q));report={};regrets={}
    for name,arm in (('hard',0),('uniform',1),('set',2)):
        label=targets(q,arm).argmax(1);regret=best-mean[rows,label];regrets[name]=regret;report[name]=interval(regret,data['stratum'])
    # In this binary, constant-hit-payoff surrogate, membership ordering and
    # mean teacher score ordering coincide exactly; ties do not cost value.
    membership=targets(q,2);np.testing.assert_array_equal(np.sign(membership[:,0]-membership[:,1]),np.sign(mean[:,0]-mean[:,1]));assert report['set']['mean']==0.
    for arm,name in enumerate(ARMS):
        action=policy_scores(data['x'],data['base'],models[arm],arm).argmax(1);regrets['student-'+name]=best-mean[rows,action];report['student-'+name]=interval(regrets['student-'+name],data['stratum'])
    for a,b in (('set','hard'),('set','uniform'),('student-set','student-hard')):report[a+'-minus-'+b]=interval(regrets[a]-regrets[b],data['stratum'])
    report['baseline']=interval(best-mean[rows,data['base'].argmax(1)],data['stratum'])
    report['meta']={'n_contexts':len(q),'n_strata':96,'n_worlds_per_context':16,'distinct_original_hands':len(np.unique(data['original_hand'])),'fixed_ensemble_uncertainty':'context draws only, not refitting or true posterior uncertainty','frozen_models_sha256':digest(O/'models/frozen_models.json'),'scope':'Finite-K PI-continuation surrogate, not actual TA continuation or true hidden value'}
    save(O/'held_label_report.json',report);return report
def game_block(seed,g,models,deadline,path,freeze):
    path=Path(path);path.mkdir(parents=True,exist_ok=True);results=[]
    for arm in range(9):
        check_time(deadline);verify_sources(freeze);p=path/f'arm{arm}.json'
        if p.exists():
            r=json.loads(p.read_text());assert r['seed']==seed and r['hero']==g%4 and r['arm']==arm and r['source_freeze_sha256']==digest(O/'source_freeze.json');results.append(r);continue
        policies=agents();hero=g%4;student=None
        if arm==8:policies[hero]=Agent(ModelBidder(MODEL),baseline())
        else:student=Student(models[arm],arm);policies[hero]=Agent(ModelBidder(MODEL),student)
        started=time.monotonic();result=play_game(RULES,policies,seed=seed,record_trace=False);elapsed=time.monotonic()-started
        r={'seed':seed,'g':g,'hero':hero,'arm':arm,'totals':result.totals,'hero_score':result.totals[hero],'hero_ties_as_win':int(result.totals[hero]==max(result.totals)),'active_decisions':student.active_count if student else None,'game_wall_seconds':elapsed,'rounds':[{'n':v.n_cards,'trump':v.trump,'dealer':v.dealer,'bids':v.bids,'tricks':v.tricks,'scores':v.scores} for v in result.rounds],'source_freeze_sha256':digest(O/'source_freeze.json')};save(p,r);results.append(r)
    return results
def games(stage,base,models,frozen,registry,freeze,deadline):
    folder=O/'games'/stage;folder.mkdir(parents=True,exist_ok=True)
    for g in range(2000):
        registry.add([base+g]);p=folder/f'{g:05d}.json'
        if p.exists():
            block=json.loads(p.read_text());assert block['frozen_models_sha256']==digest(O/'models/frozen_models.json');continue
        rows=game_block(base+g,g,models,deadline,folder/'partial'/f'{g:05d}',freeze)
        save(p,{'g':g,'hero':g%4,'seed':base+g,'arms':rows,'frozen_models_sha256':digest(O/'models/frozen_models.json'),'source_freeze_sha256':digest(O/'source_freeze.json')});status(f'N3game {stage} {g+1}/2000 complete nine-arm blocks')
    raw=[json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))];assert len(raw)==2000
    scores=np.array([[a['hero_score'] for a in r['arms']] for r in raw]);wins=np.array([[a['hero_ties_as_win'] for a in r['arms']] for r in raw]);table=np.array([[sum(a['totals']) for a in r['arms']] for r in raw]);seats=np.arange(2000)%4;best=frozen['best_arm_index'];out={}
    contrasts=[('PRIMARY_set-minus-hard',2,0),('PRIMARY_best-minus-baseline',best,8)]+[(name+'-minus-baseline',i,8) for i,name in enumerate(ARMS)]+[('uniform-minus-hard',1,0),('set-minus-uniform',2,1),('set-minus-linear-set',2,6),('hard-minus-linear-hard',0,5),('set-minus-no-offset-set',2,7)]
    for name,a,b in contrasts:out[name]={'score':interval(scores[:,a]-scores[:,b],seats),'ties_as_win':interval(wins[:,a]-wins[:,b],seats),'table_score':interval(table[:,a]-table[:,b],seats),'opponents_mean_score':interval(((table[:,a]-scores[:,a])-(table[:,b]-scores[:,b]))/3,seats)}
    out['exploratory_label_gain_interaction_MLP-minus-linear']={'score':interval((scores[:,2]-scores[:,0])-(scores[:,6]-scores[:,5]),seats),'scope':'Selected-pipeline difference-in-differences, not an isolated nonlinearity effect'}
    report={'stage':stage,'base':base,'contrasts':out,'absolute_scores':{name:interval(scores[:,i],seats) for i,name in enumerate((*ARMS,'baseline'))},'absolute_ties_as_win':{name:interval(wins[:,i],seats) for i,name in enumerate((*ARMS,'baseline'))},'n_complete_blocks':2000,'frozen_models_sha256':digest(O/'models/frozen_models.json'),'limitations':'Two fixed primary tests; all secondary controls exploratory without familywise correction. Same selected pipelines, not isolated architecture effects.'};save(O/f'{stage}_summary.json',report);return report
