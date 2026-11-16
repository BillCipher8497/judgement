"""Independent D2 raw-game review; no runner/analyzer imports."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;D2=HERE.parent/'D2';ENGINE=HERE.parents[2]
sys.path.insert(0,str(ENGINE));OUT=HERE/'results'
NS=np.array([*range(13,0,-1),*range(2,14)])
NT=np.arange(25)%5==4;DEALER=np.arange(25)%4
BASES=[1500000000,1510000000];report={};fresh={}

def est(y,seats,counts=None):
    y=np.asarray(y,float);point=float(y.mean())
    if counts is not None:
        point=float(y.sum()/counts.sum());y=(y-point*counts)/counts.mean()
    # Variance of equal-weight independent seat-stratum means.
    variances=[]
    for seat in range(4):
        z=y[seats==seat];variances.append(float(((z-z.mean())**2).sum()/(len(z)-1)/len(z)))
    half=1.96*np.sqrt(sum(variances))/4
    return {'point':point,'half95':float(half),'ci95':[point-half,point+half],'n_games':len(y)}

for base in BASES:
    path=D2/'results'/f'raw_{base}.jsonl';raw=path.read_bytes();rows=[json.loads(line) for line in raw.splitlines()]
    assert len(rows)==4000
    delta=[];errors=[];wins=[];seats=[]
    for g,row in enumerate(rows):
        seat=g%4;assert row['g']==g and row['base']==base and row['seat']==seat;seats.append(seat)
        rng=np.random.default_rng(np.random.SeedSequence(base+g).spawn(5)[0]);a,b=row['arms']
        assert len(a['hero_bid_contexts'])==len(b['hero_bid_contexts'])==25
        for r,n in enumerate(NS):
            perm=rng.permutation(52);hand=sum(1<<int(c) for c in perm[seat*n:(seat+1)*n])
            ca,cb=a['hero_bid_contexts'][r],b['hero_bid_contexts'][r]
            assert ca['hand']==cb['hand']==hand and ca['prior']==cb['prior'] and ca['legal']==cb['legal']
            position=(seat-int(DEALER[r])-1)%4
            assert [pc[0] for pc in ca['prior']]==[(int(DEALER[r])+1+k)%4 for k in range(position)]
            expected=[v for v in range(int(n)+1) if position<3 or v+sum(pc[1] for pc in ca['prior'])!=n]
            assert ca['legal']==expected
            for arm,context in [(a,ca),(b,cb)]:
                assert context['bid']==arm['bids'][r][seat] and context['bid'] in expected
                assert context['prior']==[[p,arm['bids'][r][p]] for p,v in context['prior']]
        for arm in [a,b]:
            bids=np.asarray(arm['bids']);taken=np.asarray(arm['tricks']);score=np.asarray(arm['scores'])
            assert bids.shape==taken.shape==score.shape==(25,4)
            assert np.all((bids>=0)&(bids<=NS[:,None])) and np.all(taken>=0)
            np.testing.assert_array_equal(taken.sum(1),NS);assert np.all(bids.sum(1)!=NS)
            np.testing.assert_array_equal(score,np.where(bids==taken,10+bids,0))
            np.testing.assert_array_equal(score.sum(0),arm['totals'])
        delta.append(np.asarray(b['scores'])[:,seat]-np.asarray(a['scores'])[:,seat])
        errors.append(np.asarray(a['bids'])[:,seat]-np.asarray(b['bids'])[:,seat])
        wins.append(int(b['totals'][seat]==max(b['totals']))-int(a['totals'][seat]==max(a['totals'])))
    delta=np.asarray(delta);errors=np.asarray(errors);seats=np.asarray(seats)
    positions=(seats[:,None]-DEALER-1)%4
    def cell(mask):
        mask=np.broadcast_to(mask,delta.shape);count=mask.sum(1)
        if not count.sum():return {'n_rounds':0,'structurally_absent':True}
        loss=(delta*mask).sum(1);dist={}
        for error in np.unique(errors[mask]):
            num=((errors==error)&mask).sum(1)
            dist[str(error)]={'count':int(num.sum()),'fraction':est(num,seats,count)}
        return {'n_rounds':int(count.sum()),'games_with_rounds':int(sum(count>0)),
                'loss_contribution_per_game':est(loss,seats),'loss_per_round':est(loss,seats,count),
                'mean_bid_error_guide_minus_model':est((errors*mask).sum(1),seats,count),
                'bid_error_distribution':dist}
    joint={f'n{n}|{kind}|pos{p}':cell((NS[None,:]==n)&(NT[None,:]==nt)&(positions==p))
           for n in range(1,14) for nt,kind in [(False,'trump'),(True,'NT')] for p in range(4)}
    primary=est(delta.sum(1),seats)
    assert abs(sum(x['loss_contribution_per_game']['point'] for x in joint.values() if x['n_rounds'])-primary['point'])<1e-10
    item={'n_games':4000,'raw_sha256':hashlib.sha256(raw).hexdigest(),
          'primary_model_minus_guide_points':primary,'ties_as_win_model_minus_guide':est(wins,seats),
          'by_cards':{str(n):cell(NS==n) for n in range(1,14)},
          'by_trump':{kind:cell(NT==nt) for nt,kind in [(False,'trump'),(True,'NT')]},
          'by_position':{str(p):cell(positions==p) for p in range(4)},'joint_cells':joint}
    fresh[str(base)]=item
    if base==BASES[0]:
        selected=sorted((k for k,v in joint.items() if v['n_rounds'] and v['loss_contribution_per_game']['point']>0),
                        key=lambda k:(-joint[k]['loss_contribution_per_game']['point'],k))[:3]
        frozen=json.loads((D2/'results/frozen_top3.json').read_text())
        assert selected==frozen['cells'] and item['raw_sha256']==frozen['raw_sha256']
    report[str(base)]={'primary':primary,'wins':item['ties_as_win_model_minus_guide'],
                       'raw_sha256':item['raw_sha256'],'frozen_top3':{k:joint[k] for k in selected}}
    (OUT/'D2_raw_audit.json').write_text(json.dumps({'stages':report,'all_cells':fresh},indent=2),encoding='utf-8')

# Only after saving fresh estimates compare every original numerical cell.
old=json.loads((D2/'results/summary.json').read_text())
def compare(x,y):
    if isinstance(x,dict):
        for k,v in x.items():
            if k in ['raw_sha256','ci95']:continue
            compare(v,y[k])
    elif isinstance(x,(int,float)):assert abs(x-y)<1e-10,(x,y)
    else:assert x==y
for base in BASES:compare(fresh[str(base)],old[str(base)])

# Original policy full-game replay on an a priori spread of seeds.
from judgement import Rules,play_game
from judgement.bidding.model import Agent,BidModel,ModelBidder
from judgement.policies.target_aware import TargetAwarePolicy
from judgement.policies.human_guide import GuideAgent,HumanBidder,HumanPlayPolicy
model=BidModel.load(ENGINE/'experiments/phase1/results/model_ta_final.json');replays=0
for base in BASES:
    rows=[json.loads(line) for line in (D2/'results'/f'raw_{base}.jsonl').read_text().splitlines()]
    for g in [0,997,1994,2991,3988]:
        for i,bidder in enumerate([HumanBidder(),ModelBidder(model)]):
            agents=[Agent(ModelBidder(model),TargetAwarePolicy()) for p in range(4)]
            agents[g%4]=GuideAgent(bidder,HumanPlayPolicy())
            game=play_game(Rules(),agents,seed=base+g,record_trace=False);arm=rows[g]['arms'][i]
            assert game.totals==arm['totals']
            assert [r.bids for r in game.rounds]==arm['bids'] and [r.tricks for r in game.rounds]==arm['tricks']
            replays+=1
report['checks']={'all_numeric_cells_match_to':1e-10,'full_game_reference_replays':replays,
    'seed_bases':BASES,'n_per_base':4000,'method':'independent full-game cluster normal CI within rotated-seat strata; ratio influence',
    'limits':'Selected discovery cells remain descriptive; fresh replication has three tests, unadjusted95% intervals. No rule tuning or isolated direct bid EV claim.'}
(OUT/'D2_raw_audit.json').write_text(json.dumps({'stages':report,'all_cells':fresh},indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
