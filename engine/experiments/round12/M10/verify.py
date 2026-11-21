"""Separate model-selection, raw-game, paired-CI and fresh replay audit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,sys,importlib.util
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent; R=H.parent; ROOT=R.parents[2]
sys.path[:0]=[str(R/'M6'),str(R/'M4'),str(R),str(ROOT/'engine')]
from verify_nt import independent_probabilities,paired_ci,design,MODEL,WIDTH
from common import sha,save,stamp
spec=importlib.util.spec_from_file_location('matched_frozen',H/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
O=H/'results'
def main():
    freeze=json.loads((O/'source_freeze.json').read_text());packet=json.loads((O/'frozen_ensembles.json').read_text())
    for rel,h in freeze['source_sha256'].items():assert sha(ROOT/rel)==h
    assert packet['freeze_sha256']==sha(O/'source_freeze.json')
    for rel,h in packet['model_sha256'].items():assert sha(O/rel)==h
    for rel,h in freeze['training_raw_sha256'].items():assert sha(m.OLD/rel)==h
    rows=[json.loads(s) for s in (m.OLD/'selection_contexts.jsonl').read_text().splitlines()];n=len(rows)
    x=np.zeros((n,14,WIDTH));off=np.zeros((n,14));base=np.zeros_like(off);mask=np.zeros_like(off,bool);score=np.zeros_like(off)
    for i,row in enumerate(rows):
        bids=row['context']['legal'];xx,oo,pp=design(row['context'],MODEL)
        x[i,bids]=xx;off[i,bids]=oo;base[i,bids]=pp;mask[i,bids]=True;score[i,bids]=[o['hero_score'] for o in row['outcomes']]
    norm=json.loads((m.OLD/'training_normalization.json').read_text());x=(x-np.array(norm['mu']))/np.array(norm['sd'])
    def total(p):
        a=np.argmax(np.where(mask,p*(10+np.arange(14)),-np.inf),axis=1);return int(score[np.arange(n),a].sum())
    checked=0
    for arm in ('L','R','N'):
        probabilities=[]
        for seed in range(5):
            f=O/f'{arm}_s{seed}';r=json.loads((f/'frozen.json').read_text());trace=[json.loads(s) for s in (f/'epochs.jsonl').read_text().splitlines()]
            assert [v['epoch'] for v in trace]==list(range(1,51));choices=[]
            for epoch in (10,25,50):
                p=m.parameters(f/f'candidate_{epoch}.npz');pred=independent_probabilities(x,off if arm!='N' else np.zeros_like(off),p)
                value=total(pred);assert value==next(v['selection_score_sum'] for v in trace if v['epoch']==epoch);choices.append((-value,epoch))
                if epoch==r['selected_epoch']:
                    probabilities.append(pred)
                    for a,b in zip(p,m.parameters(f/'selected.npz')):np.testing.assert_array_equal(a,b)
            assert min(choices)==(-r['selection_score_sum'],r['selected_epoch']);checked+=1
        mean=np.mean(probabilities,axis=0);best=min((-total((1-a)*base+a*mean),a) for a in (0.,.25,.5,1.))
        assert best[1]==packet['models'][arm]['alpha']
    stages={};models={a:m.Ensemble(p) for a,p in packet['models'].items()};fresh=0
    for stage,seedbase in [('screen',2040000000),('confirmation',2050000000)]:
        values=[];seats=[]
        for g in range(2000):
            row=json.loads((O/stage/f'{g:04d}.json').read_text());assert row['g']==g and row['base']==seedbase and row['models_freeze_sha256']==sha(O/'frozen_ensembles.json')
            m.validate(row);values.append(row);seats.append(g%4)
        contrasts={}
        for a,b in [('R','L'),('R','N'),('R','canonical'),('L','canonical'),('N','canonical')]:
            dif=[];wins=[]
            for row in values:
                h=row['hero'];aa=row['arms'][a]['totals'];bb=row['arms'][b]['totals'];dif.append(aa[h]-bb[h]);wins.append(int(aa[h]==max(aa))-int(bb[h]==max(bb)))
            contrasts[a+'-'+b]={'score':paired_ci(dif,seats),'ties_as_win':paired_ci(wins,seats)}
        addresses=[4*k+p for k in (0,1,250,498,499) for p in range(4)]
        for g in addresses:
            for arm in ('L','R','N','canonical'):
                agents=[m.Agent(m.ModelBidder(m.BASE),m.ta_plus_v2()) for _ in range(4)]
                if arm!='canonical':agents[g%4]=m.Agent(m.NTResidualBidder(models[arm],m.ModelBidder(m.BASE)),m.ta_plus_v2())
                result=m.play_game(m.RULES,agents,seedbase+g,record_trace=False)
                rebuilt={'totals':result.totals,'rounds':[{'n':v.n_cards,'trump':v.trump,'bids':v.bids,'tricks':v.tricks,'scores':v.scores} for v in result.rounds]}
                assert rebuilt==values[g]['arms'][arm];fresh+=1
        stages[stage]={'base':seedbase,'n_quadruplets':2000,'contrasts':contrasts,'n_fresh_replay_games':80}
        save(O/'verification_progress.json',{'stages':stages,'utc':stamp()})
    save(O/'verified_summary.json',{'status':'ready-for-Claude-external-review','models_checked':checked,'fresh_games_replayed':fresh,'stages':stages,
         'primary_supported':all(s['contrasts']['R-L']['score']['ci95'][0]>0 for s in stages.values()),'verifier_sha256':sha(__file__),'utc':stamp(),
         'limitations':'Reused selection; fixed families/ensembles; marginal normal95%CIs, no general architecture or causal feature claim.'})
if __name__=='__main__':main()
