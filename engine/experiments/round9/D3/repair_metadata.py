"""Correct ambiguous derived iteration labels; never alters values/profiles."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=HERE/'results/summary_final.json';data=json.loads(p.read_text())
for key,row in data.items():
    main=json.loads((HERE/'results'/f'trump_{key}.json').read_text())
    for candidate in [row['selected_policy'],*row['profile_replays']]:
        if 'iteration' not in candidate:continue
        candidate['archive_iteration']=candidate.pop('iteration')
        name=candidate['name']
        if name=='second_slot_CFR':
            extra=json.loads((HERE/'results/trump_0_optimization.json').read_text())
            candidate['best_iteration']=extra['best_iteration']
            candidate['updates_completed']=extra['iteration']
        elif name=='TA_baseline':candidate.update(best_iteration=0,updates_completed=0)
        else:candidate.update(best_iteration=main['best_feasible']['iteration'],updates_completed=None)
p.write_text(json.dumps(data,indent=2),encoding='utf-8')
print('Corrected iteration metadata only; numerical results and saved profiles unchanged.')
