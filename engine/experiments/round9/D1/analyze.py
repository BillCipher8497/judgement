import json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from common import save,log,status
arms=['self','own','hero','all','omni'];out={}
for base in [1600000000,1610000000]:
    rows=[json.loads(p.read_text()) for p in sorted((HERE/'results').glob(f'{base}_*.json'))]
    if len(rows)<2:continue
    x=np.array([[r['arms'][a]['scores'][0] for a in arms] for r in rows],float)
    groups=[np.flatnonzero(np.array([r['trump'] for r in rows])==t) for t in [0,-1]]
    if not all(len(v)>0 for v in groups):continue
    for r in rows:
        assert r['arms']['all']==r['arms']['omni']
        for a in arms:
            b,t,s=[np.asarray(r['arms'][a][k]) for k in ['bids','tricks','scores']]
            np.testing.assert_array_equal(b,r['fixed_bids']);assert sum(t)==5 and sum(b)!=5
            np.testing.assert_array_equal(s,np.where(b==t,10+b,0))
    # Equal mixture of S/NT even if deadline cuts the last block unevenly.
    mean=np.mean([x[g].mean(0) for g in groups],axis=0)
    rng=np.random.default_rng([base,9001]);draw=[]
    for _ in range(100):
        boot=sum(x[g[rng.integers(len(g),size=(100,len(g)))]].mean(1) for g in groups)/2
        draw.append(boot)
    boot=np.concatenate(draw)
    share=(mean[0]-mean)/mean[0];bs=(boot[:,[0]]-boot)/boot[:,[0]]
    def interval(p,sample):return {'point':float(p),'ci95':np.quantile(sample,[.025,.975]).tolist(),'n':len(rows)}
    item={'n':len(rows),'trump_counts':[len(g) for g in groups],
          'mean_hero_score':{a:interval(mean[i],boot[:,i]) for i,a in enumerate(arms)},
          'surplus_share':{a:interval(share[i],bs[:,i]) for i,a in enumerate(arms) if a!='self'},
          'primary_hero_minus_own_share':interval(share[2]-share[1],bs[:,2]-bs[:,1]),
          'share_ratio_to_omniscient':{a:interval(share[i]/share[4],bs[:,i]/bs[:,4]) for i,a in enumerate(arms) if a not in ['self','omni']},
          'all_equals_omni_every_deal':True,'bootstrap_replicates':10000,'CI_unit':'paired deal, stratified S/NT'}
    out[str(base)]=item
save(HERE/'results/summary.json',out);print(json.dumps(out,indent=2))
complete=all(out.get(str(b),{}).get('n',0)==1000 for b in [1600000000,1610000000])
log('D1','READY FOR REVIEW\nFiles: D1/{intervention,run,analyze}.py, STATUS, results/*.json; tests/test_round9_information.py.\nCommands: from engine python -m pytest -q -p no:cacheprovider tests/test_round9_information.py (2 passed); python experiments/round9/D1/run.py; python experiments/round9/D1/analyze.py.\nSeeds1600M/1610M, target n1000 complete paired five-arm deals/base; K16 all information arms. Results with paired stratified deal-bootstrap95% CIs: '+json.dumps(out)+'\nLimitations: fixed heuristic hero and fixed bids, PIMC strategy fusion and paranoid hero model; only information varies among intervention arms. This identifies the effect of revelation on this algorithm, not optimal coalition value. Partial counts are explicitly reported if deadline cut. '+('Targets complete.' if complete else 'Target shortfall: task incomplete at hard cut.'))
status('D1','ready-for-review','Information intervention saved with discovery/replication CIs; see STATUS for complete counts and scope.' if complete else 'Hard-cut partial result; target shortfall and achieved n documented for review.')
