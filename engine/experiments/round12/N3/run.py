"""Fixed N3 experiment, two preallocated bounded windows, no effect-based changes."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from common import *
import argparse,threading
from seeds import Registry
from data import generate
from network import init
from fit import fit,load_models
from evaluate import game_block,label_report,games
def preflight():
    # Metadata-only check and stream reservation; no game/model draw.
    registry=Registry();registry.fit()
    for base in (2130000000,2140000000):
        for g in range(2000):registry.add([base+g])
    for g in range(16):registry.add([BASE,DOMAIN,90,g])
    p=source_packet();save(H/'preflight.json',{'sources':p,'seed_collision_receipt':registry.receipt(),'scope':'Metadata only; N3data generation not launched','utc':utc()});print(p['source_fingerprint'])
def pilot(registry,freeze,deadline):
    receipt=O/'resource_gate.json'
    if receipt.exists():return json.loads(receipt.read_text())
    generate('pilot',registry,freeze,deadline);registry.fit();models=[[{'params':init(arm,seed),'alpha':1.} for seed in range(5)] for arm in range(8)]
    for g in range(16):
        address=[BASE,DOMAIN,90,g];registry.add(address);game_block(address,g,models,deadline,O/'timing_games'/f'{g:03d}',freeze);status(f'N3timing-only pilot {g+1}/16nine-arm blocks')
    data_seconds=sum(json.loads(p.read_text())['generation_wall_seconds'] for p in (O/'data/pilot').glob('*.json'));game_seconds=sum(json.loads(p.read_text())['game_wall_seconds'] for p in (O/'timing_games').glob('*/arm*.json'));projection=1.5*(data_seconds/192*14400+game_seconds/16*4000)+900
    gate={'pilot_contexts':192,'pilot_nine_arm_game_blocks':16,'data_seconds':data_seconds,'games_seconds':game_seconds,'overhead_and_timing_margin_multiplier':1.5,'projection_data_fit_games_seconds':projection,'allocated_additional_seconds':21600-data_seconds-game_seconds,'fitting_reserve_seconds':900,'accepted':projection<21600-data_seconds-game_seconds,'scope':'Timing-only scheduling forecast, not an accuracy result or guarantee; at most two3hscientific windows, fixed counts even if null.','utc':utc()};save(receipt,gate);return gate
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.preflight:preflight();return
    assert not (O/'summary.json').exists(),'Targets already complete; do not restart writers'
    freeze=freeze_sources();deadline=float(os.environ['N3_DEADLINE']);remaining=deadline-time.time();assert 0<remaining<=10801
    jobs=sorted((O/'windows').glob('*.json'));assert len(jobs)<2,'Predeclared maximum of two scientific windows exhausted'
    job=len(jobs)+1;save(O/'windows'/f'{job}.json',{'window':job,'startedUTC':utc(),'hard_stop_epoch':deadline,'pre_run_source_commit':freeze['pre_run_source_commit'],'source_freeze_sha256':digest(O/'source_freeze.json')});status(f'WINDOW{job} primary=set-minus-hard and validation-best-minus-baseline full-game score; absolute cutoffepoch={deadline},checkpointper64contexts/epoch/completegameblock; no outcome-based retuning')
    timer=threading.Timer(remaining,lambda:os._exit(124));timer.daemon=True;timer.start()
    try:
        registry=Registry();gate=pilot(registry,freeze,deadline)
        if not gate['accepted']:
            save(O/'scope_failure.json',{'reason':'Predeclared timing forecast exceeds two3hscientific windows; no nonpilottraining/screen/confirmation generation.','resource_gate_sha256':digest(O/'resource_gate.json'),'utc':utc()});status('N3RESOURCE SCOPE FAILURE; retain pilot, do not bend fixed experiment');return
        train=generate('train',registry,freeze,deadline);val=generate('validation',registry,freeze,deadline);frozen=fit(train,val,freeze,registry,deadline);models=load_models(frozen);held=generate('held',registry,freeze,deadline);labels=label_report(held,models,frozen);screen=games('screen',2130000000,models,frozen,registry,freeze,deadline);confirmation=games('confirmation',2140000000,models,frozen,registry,freeze,deadline)
        verify_sources(freeze);save(O/'seed_collision_receipt.json',registry.receipt());save(O/'summary.json',{'status':'All fixed targets complete, separate verification and external audit pending','source_freeze_sha256':digest(O/'source_freeze.json'),'frozen_models_sha256':digest(O/'models/frozen_models.json'),'labels':labels,'screen':screen,'confirmation':confirmation,'utc':utc()});status('N3FIXED TARGETS COMPLETE; no publication, separate audit next')
    finally:timer.cancel()
if __name__=='__main__':main()
