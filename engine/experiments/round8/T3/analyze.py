import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def wilson(x):
    n=len(x);p=float(np.mean(x));z=1.96;c=(p+z*z/(2*n))/(1+z*z/n)
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {'n':n,'p':p,'low95':float(c-h),'high95':float(c+h)}
out={}
for path in (HERE/'results').glob('raw_*.jsonl'):
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    if not rows:continue
    out[path.stem]={}
    for pol in ['greedy','ta','ta_plus']:
        records=[r['policies'][pol] for r in rows]
        out[path.stem][pol]={k:wilson([v['b']==v[k] for v in records]) for k in ['mode','floor','ceil','round']}
        out[path.stem][pol]['floor_or_ceil']=wilson([v['b'] in [v['floor'],v['ceil']] for v in records])
    if path.stem=='raw_4242':
        original=json.loads((HERE.parents[1]/'phase1/results/e3_greedy_grid.json').read_text())
        out[path.stem]['original_Greedy_distribution_reproduction']=wilson([
            np.array_equal(r['policies']['greedy']['laws'][str(r['legal'][0])],old['dist'])
            for r,old in zip(rows,original)])
    out[path.stem]['MC_pairing']=('original discovery: same own-hand/prior grid; independently indexed policy MC draws'
        if path.stem=='raw_4242' else 'confirmatory replication: same hidden worlds for Greedy, TA and TA+ and all bid candidates')
(HERE/'results/summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
complete=all(sum(1 for _ in (HERE/'results'/f'raw_{seed}.jsonl').open())==4160 for seed in [4242,4243] if (HERE/'results'/f'raw_{seed}.jsonl').exists()) and len(out)==2
if complete:
    with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:
        f.write('READY FOR REVIEW\nFiles changed: T3/run.py, T3/analyze.py, results/raw_{4242,4243}.jsonl, results/summary.json; diagnostic-only unpaired_prefix_4243.jsonl excluded.\nExact commands: python experiments/round8/run_tasks.py (from engine; sequential second-slot worker); equivalent python T3/run.py then python T3/analyze.py.\nSeed bases: original grid 4242, fresh replication 4243; n=4160 situations per seed, 300 worlds per bid. Wilson 95% CIs:\n'+json.dumps(out)+'\nLimitations: Greedy prior bids preserve the original situation design. Discovery pairs own-hand/prior contexts but its hidden MC draws differ by policy; confirmatory replication strictly pairs hidden worlds across all policies/bids. No paired TA-vs-TA+ inference is claimed from discovery. TA and TA+ laws depend on the chosen bid; mode/mean are evaluated in the law at the oracle-selected bid. Monte Carlo selection noise and overlapping mode ties remain. Wilson intervals treat situations as units, not individual rollout trials. No universal PB claim for these dependent laws.\n')
    board=HERE.parent/'BOARD.md';lines=board.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| T3 |'):
            cells=line.split('|');cells[-3]=' ready-for-review ';cells[-2]=' Original 4242 and strict-CRN replication 4243 complete; all laws and Wilson CIs saved. Bid-conditional interpretation explicitly limited. ';lines[i]='|'.join(cells)
    board.write_text('\n'.join(lines)+'\n',encoding='utf-8')
