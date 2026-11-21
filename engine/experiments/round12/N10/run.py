from n10lib import *
import argparse
from fit import fit
from evaluate import label_report,gates,games
FILES=('n10lib.py','run.py','fit.py','evaluate.py','reference.py','self_check.py','PROTOCOL.md')
def packet():
    imported={Path(m.__file__).resolve() for name,m in list(sys.modules.items()) if name.startswith('judgement') and getattr(m,'__file__',None) and str(m.__file__).endswith('.py')}
    paths=[H/p for p in FILES]+[R/'learning.py',R/'common.py',R/'N3/observables.py',ENGINE/'experiments/phase1/results/model_ta_final.json']+sorted(imported)
    return {str(p):sha(p) for p in paths}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args();O.mkdir(exist_ok=True);sources=packet();registry=Registry()
    for s in range(5):
        registry.add([FITBASE,DOMAIN,20,s])
        for e in range(1,51):registry.add([FITBASE,DOMAIN,21,s,e])
    registry.add([FITBASE,DOMAIN,99])
    for b in (2200000000,2210000000):
        for g in range(1000):registry.add(b+g)
    if args.preflight:
        save(H/'preflight.json',{'utc':stamp(),'sources':sources,'seed_receipt':registry.receipt(),'scope':'metadata only; noN10chance data/models/games yet'});print('N10namespace preflight passed',registry.receipt(),flush=True);return
    check();assert np.__version__=='2.5.4';fp=O/'sources.json'
    if fp.exists():assert json.loads(fp.read_text())==sources
    else:save(fp,sources)
    pilot=generate('pilot',registry,sources);elapsed=sum(json.loads(p.read_text())['generation_wall_seconds'] for p in (O/'data/pilot').glob('*.json'))
    oldgate=json.loads((R/'N3/results/resource_gate.json').read_text());oldpergame=oldgate['games_seconds']/(oldgate['pilot_nine_arm_game_blocks']*9);estimate=1.5*(elapsed/96*(6144+1536+1536+768)+oldpergame*6000)+900
    gp=O/'resource_gate.json'
    if gp.exists():gate=json.loads(gp.read_text())
    else:
        gate={'utc':stamp(),'pilot_contexts':96,'pilot_elapsed_seconds':elapsed,'old_N3_per_game_seconds':oldpergame,'projection_seconds':estimate,'max_two_window_seconds':21600,'accepted':estimate<21600,'scope':'timing-only oldgame extrapolation not equal-compute claim/guarantee; no models/counts changed if fails'};save(gp,gate)
    if not gate['accepted']:
        save(O/'scope_failure.json',{'reason':'prespecified resource forecast','gate':gate,'scientific_games':0});log(H,'N10resource scope failure; pilot retained, no nonpilot data/fits/games.');return
    train=generate('train',registry,sources);val=generate('validation',registry,sources)
    datafreeze={'sources':sources,'chunks':{str(p.relative_to(O)):sha(p) for name in ('train','validation') for p in (O/'data'/name).glob('*.npz')}}
    df=O/'data_frozen.json'
    if df.exists():assert json.loads(df.read_text())==datafreeze
    else:save(df,datafreeze)
    models=fit(train,val,sources,registry)
    fixtures=generate('fixtures',registry,sources);passed=gates(fixtures,val,models)
    held=generate('held',registry,sources);label_report(held,models);save(O/'seed_receipt.json',registry.receipt())
    if not passed:
        save(O/'scope_failure.json',{'utc':stamp(),'reason':'Prespecifiedactivation/selectiongatefailed','scientific_games':0,'gates_sha256':sha(O/'gates.json'),'models_sha256':sha(O/'models/frozen_models.json'),'scope':'Not a zero effect/CI; no replacementmodels/thresholds/n/epochs/alpha retuning'});log(H,'N10fixed models andheldreport complete; gate failed, ZERO scientific games, no effect promoted.')
    else:
        games('screen',2200000000,models,sources,registry);games('confirmation',2210000000,models,sources,registry);save(O/'seed_receipt.json',registry.receipt())
    assert all(sha(Path(k))==v for k,v in sources.items());save(O/'complete.json',{'utc':stamp(),'gates_passed':passed,'scientific_game_blocks':2000 if passed else 0,'source_sha256':sha(fp),'models_sha256':sha(O/'models/frozen_models.json'),'held_report_sha256':sha(O/'held_report.json'),'status':'fixedtargetsfinished or prespecifiedgatefailure; separate audit/external review pending'});log(H,'N10primary pipeline complete; separate independent checks next, no push.')
if __name__=='__main__':main()
