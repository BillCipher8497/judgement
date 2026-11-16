"""Post-review multiplicity sensitivity; does not replace primary analysis."""
import json
from pathlib import Path
from statistics import NormalDist
HERE=Path(__file__).resolve().parent
audit=json.loads((HERE/'results/D2_raw_audit.json').read_text())
z=NormalDist().inv_cdf(1-.05/(2*3));out={}
for cell,row in audit['stages']['1510000000']['frozen_top3'].items():
    d=row['loss_contribution_per_game'];half=z*d['half95']/1.96
    out[cell]={'n_games':d['n_games'],'n_rounds':row['n_rounds'],'point':d['point'],
        'familywise95_Bonferroni_interval':[d['point']-half,d['point']+half],
        'individual_coverage':1-.05/3}
(HERE/'results/D2_multiplicity.json').write_text(json.dumps({'qualification':'Secondary review sensitivity: three frozen cells, normal cluster SEs, Bonferroni familywise95% coverage. Discovery selection not corrected by its CIs.','cells':out},indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
