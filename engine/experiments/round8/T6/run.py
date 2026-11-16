"""Frozen one-candidate study: train -> screen -> one fresh validation."""
import hashlib,itertools,json,os,sys,threading,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from judgement import Rules,RoundState,play_game
from judgement.engine import deal,run_round
from judgement.bidding.model import Agent,BidModel,ModelBidder
from judgement.policies.target_aware import TargetAwarePolicy
from judgement.policies.human_guide import GuideAgent,HumanBidder,HumanPlayPolicy,_round_half_down
from judgement.policies.human_guide_v2 import HumanBidderV2,shape_features
HERE=Path(__file__).resolve().parent;RES=HERE/'results'
TRAIN=1_250_000_000;SCREEN=1_300_000_000;VALIDATION=1_400_000_000
NTRAIN=60000;NSCREEN=2000;NVALIDATION=4000

def log(message):
    with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:
        f.write(datetime.now(timezone.utc).isoformat()+' '+message+'\n')

def save_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def load_rows(path):
    if not path.exists():return []
    data=path.read_bytes().splitlines(keepends=True)
    if data:
        try:json.loads(data[-1])
        except json.JSONDecodeError:
            path.with_suffix('.incomplete_tail').write_bytes(data.pop())
            temp=path.with_suffix('.tmp');temp.write_bytes(b''.join(data));temp.replace(path)
    return [json.loads(x) for x in data]

def ci(x):
    x=np.asarray(x,float)
    assert len(x)>1 and np.isfinite(x).all()
    return {'mean':float(x.mean()),'half95':float(1.96*x.std(ddof=1)/np.sqrt(len(x))),'n':len(x)}

def predicted_bids(points,features,adjust,n,legal,weights):
    p=np.maximum(0,points+features@np.asarray(weights,int))
    b=np.clip(p//2-adjust,0,n).astype(int)
    ix=np.arange(len(b));bad=~legal[ix,b]
    up=(p%2==1)&(b<n)
    result=b.copy();result[bad&up]=b[bad&up]+1
    down=bad&~up&(b>0);result[down]=b[down]-1
    result[bad&~up&(b==0)]=1
    assert legal[ix,result].all()
    return result

def train(rules,model):
    path=RES/'train.jsonl';rows=load_rows(path)
    bidder=HumanBidder();mb=ModelBidder(model)
    for i in range(len(rows),NTRAIN):
        rng=np.random.default_rng([TRAIN,i]);r=i%rules.n_rounds
        n=rules.ladder()[r];trump=rules.trump_for_round(r);dealer=rules.dealer_for_round(r)
        pos=int(rng.integers(4));hero=(dealer+1+pos)%4
        st=RoundState(rules,n,trump,dealer,deal(rules,n,rng))
        while st.to_act!=hero:
            st.apply_bid(mb.bid(st.info_state(st.to_act),st.legal_bids(),rng))
        info=st.info_state(hero);legal=st.legal_bids()
        base_points=bidder.half_points(info.hand,trump,n)
        feat=shape_features(info.hand,trump,n,pos)
        adjust=_round_half_down((sum(b for _,b in info.bids)-pos*n/4)/4)
        base_bid=bidder.bid(info,legal,rng)
        players=[Agent(mb,TargetAwarePolicy()) for _ in range(4)]
        players[hero]=GuideAgent(bidder,HumanPlayPolicy())
        cf=[None]*14
        for b in legal:
            world=st.copy();world.apply_bid(b);run_round(world,players,[rng]*4,check=False)
            cf[b]=float(world.scores()[hero])
        row={'i':i,'seed_base':TRAIN,'n':n,'trump':trump,'dealer':dealer,'hero':hero,'pos':pos,
             'hand':int(info.hand),'prior_bids':list(info.bids),'points':base_points,'features':feat,
             'adjust':adjust,'legal':legal,'base_bid':base_bid,'cf':cf}
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        rows.append(row)
        if (i+1)%500==0:log(f'Train: {i+1}/{NTRAIN} complete counterfactual situations.')
    points=np.array([r['points'] for r in rows],int);F=np.array([r['features'] for r in rows],int)
    adjust=np.array([r['adjust'] for r in rows],int);n=np.array([r['n'] for r in rows],int)
    legal=np.zeros((len(rows),14),bool)
    for i,row in enumerate(rows):legal[i,row['legal']]=True
    cf=np.array([[np.nan if v is None else v for v in r['cf']] for r in rows])
    base=np.array([r['base_bid'] for r in rows]);ix=np.arange(len(rows))
    np.testing.assert_array_equal(predicted_bids(points,F,adjust,n,legal,[0]*4),base)
    baseline=cf[ix,base];sweep=[];best_key=None
    for w in itertools.product([-1,0,1],[-1,0,1],[-1,0,1,2],[-1,0,1,2]):
        bids=predicted_bids(points,F,adjust,n,legal,w);score=cf[ix,bids]
        entry={'weights':w,'score':ci(score),'delta_vs_current':ci(score-baseline)};sweep.append(entry)
        key=(float(score.sum()),-sum(x!=0 for x in w),-sum(abs(x) for x in w),tuple(-abs(x) for x in w))
        if best_key is None or key>best_key:best_key=key;best=entry
    config={'weights':best['weights'],'training_seed':TRAIN,'training_n':NTRAIN,
            'policy_source_sha256':hashlib.sha256((ROOT/'judgement/policies/human_guide_v2.py').read_bytes()).hexdigest(),
            'base_policy_sha256':hashlib.sha256((ROOT/'judgement/policies/human_guide.py').read_bytes()).hexdigest()}
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    result={'config':config,'frozen_sha256':digest,'selected_training':best,
            'qualification':'Selected-training CI is descriptive and selection-biased; use independent game validation.'}
    save_json(RES/'fit_sweep.json',sweep);save_json(RES/'fit.json',result)
    log('FROZEN before screening: '+json.dumps(result));return result

def games(stage,base,count,fit,rules,model):
    path=RES/f'{stage}.jsonl';rows=load_rows(path);weights=fit['config']['weights'];digest=fit['frozen_sha256']
    assert all(r['frozen_sha256']==digest and r['seed_base']==base for r in rows)
    if stage=='validation':
        marker=RES/'validation_started.json'
        if marker.exists():assert json.loads(marker.read_text())['frozen_sha256']==digest
        else:save_json(marker,{'frozen_sha256':digest,'seed_base':base,'target_games':count,'started_utc':datetime.now(timezone.utc).isoformat()})
    for g in range(len(rows),count):
        seat=g%4;out=[]
        for candidate in [False,True]:
            players=[Agent(ModelBidder(model),TargetAwarePolicy()) for _ in range(4)]
            players[seat]=GuideAgent(HumanBidderV2(weights) if candidate else HumanBidder(),HumanPlayPolicy())
            game=play_game(rules,players,seed=base+g,record_trace=False)
            out.append({'totals':list(map(float,game.totals)),
                        'bids':[list(map(int,r.bids)) for r in game.rounds],
                        'tricks':[list(map(int,r.tricks)) for r in game.rounds],
                        'scores':[list(map(float,r.scores)) for r in game.rounds]})
        row={'g':g,'seat':seat,'seed_base':base,'frozen_sha256':digest,'arms':out}
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        rows.append(row)
        if (g+1)%100==0:log(f'{stage}: {g+1}/{count} complete paired games.')
    delta=[];wins=[];scores=[[],[]];winrates=[[],[]]
    for r in rows:
        s=r['seat'];a,b=r['arms'];va,vb=a['totals'][s],b['totals'][s]
        wa=int(va==max(a['totals']));wb=int(vb==max(b['totals']))
        delta.append(vb-va);wins.append(wb-wa)
        scores[0].append(va);scores[1].append(vb);winrates[0].append(wa);winrates[1].append(wb)
    result={'seed_base':base,'frozen_sha256':digest,'weights':weights,'score_delta_v2_minus_current':ci(delta),
            'ties_win_delta':ci(wins),'current_score':ci(scores[0]),'v2_score':ci(scores[1]),
            'current_ties_win':ci(winrates[0]),'v2_ties_win':ci(winrates[1])}
    save_json(RES/f'{stage}_summary.json',result);log(stage+' COMPLETE: '+json.dumps(result));return result

def main():
    RES.mkdir(parents=True,exist_ok=True)
    stop=datetime(2026,11,16,12,30,tzinfo=timezone.utc).timestamp()
    if time.time()>=stop:return
    timer=threading.Timer(stop-time.time(),lambda:os._exit(0));timer.daemon=True;timer.start()
    rules=Rules();model=BidModel.load(ROOT/'experiments/phase1/results/model_ta_final.json')
    if (RES/'fit.json').exists():fit=json.loads((RES/'fit.json').read_text())
    else:fit=train(rules,model)
    assert fit['config']['policy_source_sha256']==hashlib.sha256((ROOT/'judgement/policies/human_guide_v2.py').read_bytes()).hexdigest()
    assert fit['config']['base_policy_sha256']==hashlib.sha256((ROOT/'judgement/policies/human_guide.py').read_bytes()).hexdigest()
    screen=games('screen',SCREEN,NSCREEN,fit,rules,model)
    validation=games('validation',VALIDATION,NVALIDATION,fit,rules,model)
    timer.cancel()
    log('READY FOR REVIEW\nFiles changed: judgement/policies/human_guide_v2.py; tests/test_round8_human_v2.py; T6/run.py, STATUS and results/*.jsonl/*.json.\nExact command: cd engine/experiments/round8/T6; python run.py.\nSeed bases and n: train1250M, 60000 situations; screen1300M, 2000 paired games; validation1400M, 4000 paired games; challenger g%4.\nResults (95% game-level CIs): '+json.dumps({'fit':fit,'screen':screen,'validation':validation})+'\nLimitations: train uses true-deal counterfactual guide/TA play and selection-biased training estimates. Screen and validation use one unchanged frozen candidate; no tuning after screening. Four integer feature additions only; final guide defaults are unchanged and no human playtest occurred.')
    board=HERE.parent/'BOARD.md';lines=board.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| T6 |'):
            cells=line.split('|');cells[-3]=' ready-for-review ';cells[-2]=' Train1250M n60k; screen1300M n2000, validation1400M n4000. One frozen rule, no tuning after screen; current guide unchanged. ';lines[i]='|'.join(cells)
    board.write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
