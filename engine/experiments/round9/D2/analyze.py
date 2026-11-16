import hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE.parents[2]),str(HERE.parent)]
from common import save,log,status
from judgement import Rules
rules=Rules();ns=np.array(rules.ladder());tr=np.array([rules.trump_for_round(r)==-1 for r in range(25)])
dealer=np.array([rules.dealer_for_round(r) for r in range(25)]);out={}
def estimate(x,seats,counts=None):
    x=np.asarray(x,float);value=float(x.mean())
    if counts is not None:
        countmean=counts.mean()
        if countmean==0:return None
        value=float(x.mean()/countmean);x=(x-value*counts)/countmean
    variance=sum(np.var(x[seats==s],ddof=1)/sum(seats==s) for s in range(4))/16
    return {'point':value,'half95':float(1.96*np.sqrt(variance)),'n_games':len(x)}

for base in [1500000000,1510000000]:
    path=HERE/'results'/f'raw_{base}.jsonl'
    if not path.exists():continue
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    if len(rows)<8:continue
    seats=np.array([r['seat'] for r in rows]);G=len(rows)
    assert np.array_equal(seats,np.arange(G)%4)
    # Analyze balanced complete 4-game blocks only, preserving the raw remainder.
    balanced=G//4*4;rows=rows[:balanced];seats=seats[:balanced];G=balanced
    delta=np.zeros((G,25));errors=np.zeros((G,25),int);win=[]
    for g,row in enumerate(rows):
        a,b=row['arms'];hero=row['seat']
        for arm in [a,b]:
            bids,taken,scores=[np.asarray(arm[k]) for k in ['bids','tricks','scores']]
            np.testing.assert_array_equal(taken.sum(1),ns);assert np.all(bids.sum(1)!=ns)
            np.testing.assert_array_equal(scores,np.where(bids==taken,10+bids,0))
            np.testing.assert_array_equal(scores.sum(0),arm['totals'])
        delta[g]=np.array(b['scores'])[:,hero]-np.array(a['scores'])[:,hero]
        errors[g]=np.array(a['bids'])[:,hero]-np.array(b['bids'])[:,hero]
        win.append(int(b['totals'][hero]==max(b['totals']))-int(a['totals'][hero]==max(a['totals'])))
        assert delta[g].sum()==b['totals'][hero]-a['totals'][hero]
        for ca,cb in zip(a['hero_bid_contexts'],b['hero_bid_contexts']):
            assert ca['hand']==cb['hand'] and ca['prior']==cb['prior'] and ca['legal']==cb['legal']
    positions=(seats[:,None]-dealer[None,:]-1)%4
    def cell(mask):
        mask=np.broadcast_to(mask,delta.shape);counts=mask.sum(1);loss=(delta*mask).sum(1)
        if not counts.sum():return {'n_rounds':0,'structurally_absent':True}
        distribution={}
        for error in range(-13,14):
            hits=((errors==error)&mask).sum(1)
            if hits.sum():distribution[str(error)]={'count':int(hits.sum()),'fraction':estimate(hits,seats,counts)}
        return {'n_rounds':int(counts.sum()),'games_with_rounds':int(sum(counts>0)),
                'loss_contribution_per_game':estimate(loss,seats),
                'loss_per_round':estimate(loss,seats,counts),
                'mean_bid_error_guide_minus_model':estimate((errors*mask).sum(1),seats,counts),
                'bid_error_distribution':distribution}
    joint={f'n{n}|{kind}|pos{p}':cell((ns[None,:]==n)&(tr[None,:]==nt)&(positions==p))
           for n in range(1,14) for nt,kind in [(False,'trump'),(True,'NT')] for p in range(4)}
    summed=sum(d['loss_contribution_per_game']['point'] for d in joint.values() if d.get('n_rounds'))
    primary=estimate(delta.sum(1),seats);assert abs(summed-primary['point'])<1e-10
    item={'n_games':G,'source_complete_games':len(path.read_text().splitlines()),'primary_model_minus_guide_points':primary,
          'ties_as_win_model_minus_guide':estimate(win,seats),
          'by_cards':{str(n):cell(ns[None,:]==n) for n in range(1,14)},
          'by_trump':{kind:cell(tr[None,:]==nt) for nt,kind in [(False,'trump'),(True,'NT')]},
          'by_position':{str(p):cell(positions==p) for p in range(4)},'joint_cells':joint,
          'CI_method':'full-game cluster normal CI, stratified by rotated hero seat; ratios use cluster influence'}
    out[str(base)]=item
    if base==1500000000 and G==4000:
        selected=sorted((k for k,v in joint.items() if v.get('n_rounds') and v['loss_contribution_per_game']['point']>0),key=lambda k:(-joint[k]['loss_contribution_per_game']['point'],k))[:3]
        frozen=HERE/'results/frozen_top3.json'
        record={'discovery_base':base,'n_discovery_games':G,'cells':selected,'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if frozen.exists():assert json.loads(frozen.read_text())==record
        else:save(frozen,record);log('D2','Top3 frozen before replication: '+json.dumps(record))
    if base==1510000000 and (HERE/'results/frozen_top3.json').exists():
        item['replication_of_frozen_top3']={k:joint[k] for k in json.loads((HERE/'results/frozen_top3.json').read_text())['cells']}
save(HERE/'results/summary.json',out)
print(json.dumps({k:{name:v[name] for name in ['n_games','primary_model_minus_guide_points','ties_as_win_model_minus_guide']} for k,v in out.items()},indent=2))
if all(out.get(str(b),{}).get('n_games',0)==4000 for b in [1500000000,1510000000]):
    packet={str(b):{k:v for k,v in out[str(b)].items() if k in ['n_games','primary_model_minus_guide_points','ties_as_win_model_minus_guide','replication_of_frozen_top3']} for b in [1500000000,1510000000]}
    log('D2','READY FOR REVIEW\nFiles: D2/run.py, analyze.py, STATUS.txt, results/*.jsonl/*.json.\nExact command: from engine python experiments/round9/D2/run.py; analysis standalone python experiments/round9/D2/analyze.py.\nSeed bases1500M discovery,1510M fresh replication, n4000 paired games each, hero g%4. Stratified game-cluster95% CIs: '+json.dumps(packet)+'\nLimitations: whole bidder intervention includes subsequent target-aware responses, not isolated direct bid error. Joint cells selected on discovery only, fixed before replication; structurally absent ladder cells not estimated. No new rule, human playtest or validation-seed tuning. All raw scores/hands/prior contexts saved; all prior contexts identical. Full cell distributions and CIs in summary.json.')
    status('D2','ready-for-review','1500M/1510M fresh replication, 4000 paired games/base; three cells frozen before replication; guide play unchanged.')
