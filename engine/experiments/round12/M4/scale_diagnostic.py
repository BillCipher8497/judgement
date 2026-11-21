"""Descriptive baseline scale and paired delta-method ratio; no new draws."""
import json,math,statistics
from pathlib import Path
H=Path(__file__).resolve().parent
result={}
for stage in ('screen','confirmation'):
    records=[json.loads(p.read_text()) for p in sorted((H/'results'/stage).glob('*.json'))]
    b=[r['arms']['canonical']['totals'][r['hero']] for r in records]
    d=[r['arms']['bce']['totals'][r['hero']]-bb for r,bb in zip(records,b)]
    mu=statistics.mean(b);delta=statistics.mean(d)
    def se(values):return math.sqrt(sum(statistics.variance([v for i,v in enumerate(values) if i%4==s])/(len(values)//4) for s in range(4))/16)
    half=1.96*se(b);ratio=delta/mu
    influence=[dd/mu-delta*bb/(mu*mu) for dd,bb in zip(d,b)];rh=1.96*se(influence)
    result[stage]={'n_games':len(b),'canonical_mean':mu,'canonical_ci95':[mu-half,mu+half],
        'gain_fraction':ratio,'gain_fraction_ci95':[ratio-rh,ratio+rh],
        'scope':'Descriptive equal-seat normal mean CI; paired delta-method ratio CI, fixed ensembles, post hoc scale only.'}
(H/'results/scale_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
