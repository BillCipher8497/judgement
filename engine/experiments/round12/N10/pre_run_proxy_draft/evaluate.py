from n10lib import *
def label_report(held,models):
    mean=held['q'].mean(1)*held['hit'][:,None];best=mean.max(1);rows=np.arange(len(mean));regrets={};report={}
    base=held['base'].argmax(1);regrets['baseline']=best-mean[rows,base]
    for arm in ARMS:
        action=ensemble(held,models[arm]).argmax(1);regrets[arm]=best-mean[rows,action]
        label=targets(held['q'],arm).argmax(1);regrets['label_'+arm]=best-mean[rows,label]
    assert not np.any(regrets['label_set'])
    for k,v in regrets.items():report[k]=interval(v,held['stratum'])
    if all(identity_bound(models[a])['global_baseline_identity_by_tanh_margin'] for a in ARMS):
        assert np.array_equal(regrets['set'],regrets['hard']) and np.array_equal(regrets['set'],regrets['baseline'])
        report['set_minus_hard_regret']={'n_contexts':len(mean),'value':0.,'ci95':None,'kind':'exact baseline-policy identity under margin bound; not a sampling interval or equivalence test'}
    else:report['set_minus_hard_regret']=interval(regrets['set']-regrets['hard'],held['stratum'])
    report['scope']='n1536independent conditional activecontexts,96strata,K16structuraluniform/privateTArollout teacher,fixedensemble; not true posterior/privateBR/fullgame values'
    npz(O/'held_predictions.npz',mean_scores=mean,**{'regret_'+k:v for k,v in regrets.items()});save(O/'held_report.json',report)
def gates(fixtures,val,models):
    outputs={};passed=True;values=val['q'].mean(1)*val['hit'][:,None];rows=np.arange(len(values));b=val['base'].argmax(1)
    for arm in ARMS:
        z=ensemble(fixtures,models[arm]);a=z.argmax(1);baseline=fixtures['base'].argmax(1);changed=a!=baseline;count=int(changed.sum());hands=len(np.unique(fixtures['original_hand'][changed]));seats=set((fixtures['g'][changed]%4).tolist());margin=np.abs(z[:,0]-z[:,1]);activation=count>=16 and hands>=8 and seats==set(range(4)) and bool(np.all(margin[changed]>=1e-6))
        a_val=ensemble(val,models[arm]).argmax(1);bound=identity_bound(models[arm])
        if bound['global_baseline_identity_by_tanh_margin']:
            assert not count and np.array_equal(a_val,b);contrast={'n_contexts':len(val['x']),'value':0.,'ci95':None,'kind':'exact baseline-policy identity under margin bound, not a sampling interval'};ni=True
        else:contrast=interval(values[rows,a_val]-values[rows,b],val['stratum']);ni=contrast['ci95'][0]>=-.05
        rate=count/len(a);half=np.sqrt(np.log(40)/(2*len(a)))
        outputs[arm]={'changed_fixture_actions':count,'n_fixtures':len(a),'n_distinct_changed_hands':hands,'changed_seats':sorted(seats),'min_changed_score_margin':float(margin[changed].min()) if count else None,'activation_gate':activation,'global_identity_bound':bound,'diagnostic_activation_rate':{'n':len(a),'mean':rate,'ci95':[max(0.,float(rate-half)),min(1.,float(rate+half))],'method':'conservative Hoeffding95%for independent bounded conditional fixture draws; not natural game frequency'},'validation_rollout_minus_baseline':contrast,'selection_reused_validation_gate':ni,'pass':activation and ni}
        npz(O/f'fixture_scores_{arm}.npz',scores=z,actions=a,baseline=baseline,changed=changed,margins=margin)
        passed=passed and activation and ni
    save(O/'gates.json',{'utc':stamp(),'arms':outputs,'both_pass':passed,'scientific_games_authorized':passed,'scope':'Prespecified finite activation proof and selection-reused conditional validation safeguard, not independent noninferiority/generalization guarantee; failedgate=>0games,notzeroeffect'})
    return passed
class Student:
    def __init__(self,models):self.models=models;self.base=OBS.baseline();self.active=0;self.changed=0
    def play(self,info,legal,rng):
        private=OBS.detach(info);bc=self.base.play(private,legal,rng)
        if not OBS.active(private,legal):return bc
        x,b,cards=OBS.encode(private,legal,bc);a=int(ensemble({'x':x[None,:],'base':b[None,:]/8},self.models).argmax(1)[0]);self.active+=1;choice=cards[a];self.changed+=choice!=bc;return choice
def games(stage,base,models,sources,registry):
    folder=O/'games'/stage;folder.mkdir(parents=True,exist_ok=True);arms=('baseline','hard','set')
    for g in range(1000):
        check();registry.add([base+g]);block=folder/f'{g:04d}.json'
        if block.exists():
            old=json.loads(block.read_text());assert old['model_freeze_sha256']==sha(O/'models/frozen_models.json');continue
        part=folder/'partial'/f'{g:04d}';part.mkdir(parents=True,exist_ok=True);done={}
        for arm in arms[g%3:]+arms[:g%3]:
            check();assert all(sha(Path(k))==v for k,v in sources.items());path=part/f'{arm}.json'
            if path.exists():
                old=json.loads(path.read_text());assert old['sources']==sources and old['model_freeze_sha256']==sha(O/'models/frozen_models.json') and old['seed']==base+g and old['hero']==g%4 and old['arm']==arm;done[arm]=old;continue
            agents=OBS.agents();hero=g%4;student=None
            if arm=='baseline':policy=OBS.baseline()
            else:student=Student(models[arm]);policy=student
            agents[hero]=Agent(ModelBidder(MODEL),policy);tick=time.perf_counter();res=play_game(RULES,agents,seed=base+g,record_trace=False)
            row={'arm':arm,'g':g,'seed':base+g,'hero':hero,'totals':res.totals,'hero_score':res.totals[hero],'ties_as_win':int(res.totals[hero]==max(res.totals)),'active':student.active if student else None,'changed':student.changed if student else None,'rounds':[{'bids':r.bids,'tricks':r.tricks,'scores':r.scores,'n':r.n_cards,'trump':r.trump,'dealer':r.dealer} for r in res.rounds],'sources':sources,'model_freeze_sha256':sha(O/'models/frozen_models.json'),'seconds':time.perf_counter()-tick};save(path,row);done[arm]=row
        save(block,{'g':g,'seed':base+g,'hero':g%4,'arms':done,'model_freeze_sha256':sha(O/'models/frozen_models.json')});save(O/'progress.json',{'stage':stage,'blocks':g+1,'target':1000,'utc':stamp()})
    raw=[json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))];assert len(raw)==1000;out={}
    for name,a,b in (('PRIMARY_set_minus_hard','set','hard'),('set_minus_baseline','set','baseline'),('hard_minus_baseline','hard','baseline')):
        out[name]={metric:interval([r['arms'][a][metric]-r['arms'][b][metric] for r in raw],np.arange(1000)%4) for metric in ('hero_score','ties_as_win')}
    save(O/f'{stage}_summary.json',{'utc':stamp(),'base':base,'n_complete_blocks':1000,'contrasts':out,'scope':'paired4seat normal95%,n1000/stage; fixedconditional-gatedpipelines vs3TA, not universal/neural-architecture causality'})
