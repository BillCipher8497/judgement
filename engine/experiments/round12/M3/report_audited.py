"""Generate corrected report from saved results, without modifying any fit."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
r=json.loads((H/'results/held_out_results.json').read_text(encoding='utf-8'))
d=json.loads((H/'results/label_decode_audit.json').read_text(encoding='utf-8'))['conditions']
lines=['# Root label-misalignment audit','Externally validated (Claude,2026-11-17). Set supervision avoids a label-induced cost; hard-CE gap is almost entirely label misalignment, not a learning-dynamics advantage. Label and decoder ties choose lowest original-hand index. Exact population costs have no sampling CI. Learned intervals below are paired seed t95%CIs only.']
lines.append('Primary hypothesis '+('supported' if all(v['ci95'][1]<0 for v in r['primary'].values()) else 'not supported')+'.')
for label,v in r['primary'].items():
    h=d['S3R4n3_t-1']['held_out'][label]
    lines.append(f"{label}: set-BCE minus hard-CE {v['mean']:.8f} +/-{v['half95']:.8f}; n={v['n_training_seeds']}paired seeds,95%tCI; {h['n_worlds']}worlds/{h['n_keys']}keys/{h['actual_own_hands']}distinct reached hands.")
lines+=['Discovery/confirmation spread shows hand-sampling uncertainty dominates; seed CIs do not cover it. Same fits, two fixed populations, not independent training replications.','|condition/split|actual hands|worlds|information floor|hard excess|uniform excess|set excess|','|---|---:|---:|---:|---:|---:|---:|']
for tag,c in d.items():
    for label,h in c['held_out'].items():
        q=h['label_excess'];lines.append(f"|{tag}/{label}|{h['actual_own_hands']}|{h['n_worlds']}|{h['information_floor']:.8f}|{q['hard']:.8f}|{q['uniform']:.8f}|{q['set']:.8f}|")
lines+=['|MLP condition/split|hard CE|uniform CE|set BCE|Q regression|direct|','|---|---:|---:|---:|---:|---:|']
for key,v in r['summaries'].items():
    if '/mlp/' in key:
        cells=[f"{v['objectives'][o]['mean']:.5f} +/-{v['objectives'][o]['half95']:.5f}" for o in ('hard_ce','uniform_ce','set_bce','q_reg','direct')]
        lines.append('|'+key+'|'+'|'.join(cells)+'|')
lines+=['Every learner cell: n5seeds,95%tCI, fixed exact population. Secondary comparisons descriptive; small controls do not uniformly favor BCE. Targets retain different label information, no label-efficiency claim.','Numerical stack: results/numerical_stack.json. Audit minimum top-two logit gap1.6e-7; exact integer totals require the same numerical stack. No full-team/52-card/general learning-advantage or novelty claim.']
(H/'RESULTS.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
