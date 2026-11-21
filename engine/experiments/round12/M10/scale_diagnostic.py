import json,math,statistics
from pathlib import Path
from datetime import datetime,timezone
H=Path(__file__).resolve().parent;rows={}
def interval(d,seats):
 parts=[[float(x) for x,s in zip(d,seats) if s==p] for p in range(4)];m=statistics.mean([statistics.mean(p) for p in parts]);h=1.96*math.sqrt(sum(statistics.variance(p)/len(p) for p in parts))/4
 return {'n_games':len(d),'mean':m,'ci95':[m-h,m+h],'halfwidth':h}
for stage in ('screen','confirmation'):
 raw=[json.loads(p.read_text()) for p in sorted((H/'results'/stage).glob('*.json'))];assert len(raw)==2000
 seats=[r['g']%4 for r in raw];base=[r['arms']['canonical']['totals'][r['hero']] for r in raw];rd=[r['arms']['R']['totals'][r['hero']]-b for r,b in zip(raw,base)]
 nl=[r['arms']['N']['totals'][r['hero']]-r['arms']['L']['totals'][r['hero']] for r in raw]
 mean=statistics.mean(base);delta=statistics.mean(rd);ratio=delta/mean
 # Paired delta-method ratio, equal seat means.
 influence=[(a-ratio*b)/mean for a,b in zip(rd,base)];ratio_ci=interval(influence,seats);half=ratio_ci['halfwidth']
 rows[stage]={'canonical':interval(base,seats),'N-L':interval(nl,seats),'R-L_ratio_of_canonical_mean':None,'R-canonical_fraction':{'n_games':2000,'mean':ratio,'ci95':[ratio-half,ratio+half],'method':'paired delta-method equal-seat95%CI; posthoc scale diagnostic'}}
out={'stages':rows,'utc':datetime.now(timezone.utc).isoformat(),'scope':'Deterministic saved-raw diagnostic; no new games or seeds. Rounded ratios descriptive.'}
(H/'results'/'scale_diagnostic.json').write_text(json.dumps(out,indent=2))
with (H/'RESULTS.md').open('a',encoding='utf-8') as f:
 f.write('\n### Raw-derived N-L and magnitude diagnostic\n\nAll rows n=2000 games/stage, paired/equal-seat normal95%CIs; ratios use a paired delta method. These are post-hoc descriptive scales.\n\n|stage|canonical mean [95%CI]|N-L [95%CI]|R-canonical / canonical mean [95%CI]|\n|---|---:|---:|---:|\n')
 for name,v in rows.items():
  fmt=lambda z:f"{z['mean']:.6f} [{z['ci95'][0]:.6f}, {z['ci95'][1]:.6f}]"
  f.write(f"|{name}|{fmt(v['canonical'])}|{fmt(v['N-L'])}|{fmt(v['R-canonical_fraction'])}|\n")
print(json.dumps(out))
