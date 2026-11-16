import json
from io import BytesIO
from pathlib import Path
import numpy as np
from run import metrics
HERE=Path(__file__).resolve().parent
out={}
for base in [1100000000,1110000000]:
 for n in [5,8]:
  for pos in [1,2,3]:
   rows=[]
   paths=sorted((HERE/'results').glob(f'{base}_{n}_{pos}_*.npz'))
   for p in paths:
    with np.load(BytesIO(p.read_bytes())) as z:
     h=metrics(z['cards'],z['cm'],z['tm'],n,z['legal'])
     if h is not None:rows.append(h)
   if len(rows)<2:continue
   d={'raw_hands':len(paths),'n_hands':len(rows)}
   for key in rows[0]:
    x=np.array([h[key] for h in rows if h[key] is not None])
    if len(x)>1:d[key]=[float(x.mean()),float(1.96*x.std(ddof=1)/np.sqrt(len(x))),len(x)]
   # Paired-hand bootstrap for ratio of match improvements to PB-to-floor gap.
   rng=np.random.default_rng([base,n,pos,81]);a=np.array([[h[k] for k in ['PB_match','floor_match','LC0_match','TPB_match']] for h in rows])
   boot=a[rng.integers(len(a),size=(4000,len(a)))].mean(1);m=a.mean(0)
   for k,name in [(2,'LC0'),(3,'TPB')]:
    denom=boot[:,1]-boot[:,0];valid=np.abs(denom)>1e-9
    ratio=(boot[valid,k]-boot[valid,0])/denom[valid]
    d[name+'_gap_closed']={'point':float((m[k]-m[0])/(m[1]-m[0])) if abs(m[1]-m[0])>1e-9 else None,
                         'ci95':np.quantile(ratio,[.025,.975]).tolist() if len(ratio) else None,
                         'valid_bootstraps':int(valid.sum()),'warning':'unstable if PB-to-floor denominator approaches zero'}
   out[f'{base}|n{n}|pos{pos}']=d
   temp=HERE/'results/summary.tmp'
   temp.write_text(json.dumps(out,indent=2,allow_nan=False));temp.replace(HERE/'results/summary.json')
   with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:
    f.write(f'Analysis checkpoint: base={base}, n={n}, pos={pos}, raw={len(paths)}, nondegenerate={len(rows)}.\n')
(HERE/'results/summary.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps(out,indent=2))
if len(out)==12 and all(v['raw_hands']==500 for v in out.values()):
 with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:
  f.write('READY FOR REVIEW\nFiles changed: T4/run.py, T4/analyze.py, T4/worker.py, results/*.npz and summary.json; engine/tests/test_round8_lead.py.\nExact command: cd engine/experiments/round8/T4; python worker.py (focused tests, compute and analysis in the freed slot); equivalent python run.py then python analyze.py.\nSeed bases: 1100000000 discovery, 1110000000 replication; 500 hands x 300 rollouts per n/position/base; n=5,8 and hero positions1,2,3. Completed nondegenerate hand counts and 95% intervals:\n'+json.dumps(out)+'\nLimitations: uniform hidden worlds ignore bid-likelihood weighting; all bidding/play TA, prior public bids and own bid fixed before sampling. Cross-suit component is normalized by PB variance, matching M5. Lead persistence excludes trick1. Bid-match choices respect the bid-time legal set, including dealer hook. Representational fits, not deployable bidder. Bootstrap gap ratios can be unstable near a zero PB-to-floor denominator.\n')
 board=HERE.parent/'BOARD.md';lines=board.read_text(encoding='utf-8').splitlines()
 for i,line in enumerate(lines):
  if line.startswith('| T4 |'):
   cells=line.split('|');cells[-3]=' ready-for-review ';cells[-2]=' 1100M discovery and 1110M replication complete; raw masks, paired bootstrap CIs and hook-legal diagnostics saved. See STATUS limitations. ';lines[i]='|'.join(cells)
 board.write_text('\n'.join(lines)+'\n',encoding='utf-8')
