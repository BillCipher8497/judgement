"""Replay saved feasible profiles and combine completed D3 outputs; no training."""
import gc,hashlib,json,pickle,sys
from fractions import Fraction
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ENGINE=HERE.parents[2]
sys.path[:0]=[str(ENGINE),str(HERE),str(HERE.parent)]
from common import save,log,status
from compact_tree import CompactTree
from judgement.cfr.solve import profile_value
from root_bound import root_information_bound

def read(path):return json.loads(path.read_text(encoding='utf-8'))
out={}
for trump in [0,-1]:
    raw=HERE/'results'/f'trump_{trump}.json'
    if not raw.exists():continue
    base=read(raw);metadata=read(HERE/'results'/f'tree_cache_{trump}.json')
    fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in [HERE/'compact_tree.py',HERE/'bounded_ta.py',ENGINE/'judgement/cfr/toy.py',ENGINE/'judgement/cfr/heuristics.py'])).hexdigest()
    assert fingerprint==metadata['source_sha256']
    with Path(metadata['derived_cache']).open('rb') as f:t=pickle.load(f)
    assert t.n_deals==369600 and (t.g.N,t.g.S,t.g.R,t.g.n,t.g.trump)==(4,3,4,3,trump)
    bound=root_information_bound(t)
    assert bound==base['root_information_relaxation']
    assert bound['PI_fraction']==base['PI_fraction']
    candidates=[]
    with np.load(HERE/'results'/f'trump_{trump}_profile.npz') as z:
        candidates.append({'name':'main_best','file':f'trump_{trump}_profile.npz','array':'sigma','sigma':z['sigma'].copy(),
                           'expected':base['best_feasible']['hero_score'],'best_iteration':base['best_feasible']['iteration'],
                           'archive_iteration':int(z['iteration']),'updates_completed':None})
        candidates.append({'name':'TA_baseline','file':f'trump_{trump}_profile.npz','array':'fixed','sigma':z['fixed'].copy(),
                           'expected':base['baseline_TA'],'best_iteration':0,'archive_iteration':0,'updates_completed':0})
    extra=HERE/'results/trump_0_optimization.json'
    if trump==0 and extra.exists():
        extension=read(extra)
        with np.load(HERE/'results/trump_0_optimized_profile.npz') as z:
            candidates.append({'name':'second_slot_CFR','file':'trump_0_optimized_profile.npz','array':'sigma','sigma':z['sigma'].copy(),
                               'expected':extension['best_feasible_score'],'best_iteration':extension['best_iteration'],
                               'archive_iteration':int(z['iteration']),'updates_completed':extension['iteration']})
    replay=[]
    for candidate in candidates:
        sigma=candidate['sigma']
        assert sigma.shape==t.is_valid.shape and np.isfinite(sigma).all()
        assert np.all(sigma>=0) and np.all(sigma<=1) and np.all(sigma[~t.is_valid]==0)
        np.testing.assert_allclose(sigma.sum(1),1,atol=1e-10,rtol=0)
        value=float(profile_value(t,sigma)[0]);assert abs(value-candidate['expected'])<1e-7,(trump,candidate['name'],value,candidate['expected'])
        candidate['score']=value
        replay.append({k:v for k,v in candidate.items() if k!='sigma'})
    selected=min(candidates,key=lambda c:c['score']);achieved=selected['score']
    pi=float(Fraction(bound['PI_fraction']));lower=float(Fraction(bound['first_decision_hidden_lower_fraction']))
    assert pi<=lower+1e-12 and lower<=achieved+1e-7
    baseline=base['baseline_TA'];den=baseline-pi;assert den>0
    item={'trump':trump,'n_exact_deals':t.n_deals,'baseline_TA':baseline,
          'independent_BR_hero_score':base['independent_BR_to_TA']['hero_score'],
          'PI_fraction':bound['PI_fraction'],'root_hidden_lower_fraction':bound['first_decision_hidden_lower_fraction'],
          'certified_information_cost_lower_fraction':bound['lower_bound_gain_fraction'],
          'achieved_score':achieved,'primary_PI_achieved_interval':[pi,achieved],
          'tighter_hidden_optimum_interval':[lower,achieved],
          'optimal_information_cost_interval':[lower-pi,achieved-pi],
          'achieved_punishment_ratio':(baseline-achieved)/den,
          'optimal_hidden_PI_punishment_ratio_bounds':[(baseline-achieved)/den,(baseline-lower)/den],
          'selected_policy':{k:v for k,v in selected.items() if k!='sigma'},'profile_replays':replay,
          'TA_profile_support':base['TA_profile_support'],
          'uncertainty':'Complete chance enumeration; no sampling CI. Lower bounds are exact rationals; feasible expectations are floating replay values. No global team optimum claimed.'}
    out[str(trump)]=item
    save(HERE/'results/summary_final.json',out)
    log('D3',f'Saved-profile replay and bound consolidation complete for trump{trump}.')
    del t,candidates,selected,sigma
    gc.collect()
print(json.dumps(out,indent=2))
log('D3','READY FOR REVIEW — combined final packet\nFiles: D3/{frozen_tree,compact_tree,bounded_ta,reachable_ta,root_bound,run,optimize_saved,consolidate}.py, README.md, ROOT_BOUND.md, STATUS and complete results/*.json/*.npz; tests/test_round9_frozen_toy.py; round9/common.py and watch.ps1 deadline/IO helpers.\nCommands from engine: python -m pytest -q -p no:cacheprovider tests/test_round9_frozen_toy.py (1 passed in1.64s after support/root changes); python experiments/round9/D3/run.py; python experiments/round9/D3/optimize_saved.py (second slot inside original box); python experiments/round9/D3/consolidate.py (saved-output validation only, no optimizer steps).\nSeed1620M resource pilot only; full population n369600 ordered deals/regime, no sampling95% CI applies. Exact rational lower bounds and replayed feasible values: '+json.dumps(out)+'\nLimitations: original3h wall box retained through infrastructure pauses/retries; fewer than1000 CFR updates can be completed. All bids/TA hero frozen; private infosets retained. Opponent baseline completion is exact only on baseline/unilateral-deviation support; off-support defaults do not restrict free-team CFR. PI/achieved and J1/achieved are optimization bounds, not statistical intervals. First-decision relaxation proof is a new derivation awaiting Claude audit; not a full-team solve. No proof from unilateral residuals or CFR convergence, and no52-card extrapolation. Derived local caches are not deliverable results. All candidate saved profiles replay with valid probabilities/legal masks and matching recorded values. Root bounds reproduced from complete cached trees. Await Claude verdict before commit or paper/AGENTS promotion.')
status('D3','ready-for-review','Both regimes exact369600 deals, saved profiles replayed; combined CFR/PI and first-decision bounds in summary_final.json; new proof awaits audit.')
