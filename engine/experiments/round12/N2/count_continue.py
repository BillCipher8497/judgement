"""Resource-only N2 count continuation; immutable scientific producers reused."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib,itertools,subprocess,sys,time,threading
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H))
import run as science
from export import digest
from fit import save,check_deadline
ROOT=H.parents[3];O=science.O
OWN_SOURCES=[H/'count_continue.py',H/'verify_continuation.py',H/'RESOURCE_CONTINUATION.md']

def own_packet():
    raw={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in OWN_SOURCES};canonical={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in OWN_SOURCES};fp=hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()
    return {'raw_sources':raw,'canonical_sources':canonical,'scheduling_source_fingerprint':fp}

def initial_premises():
    science.source_check();original=json.loads((O/'source_freeze.json').read_text());gate=json.loads((O/'resource_gate.json').read_text())
    actual=science.packet()
    assert actual['source_fingerprint']==original['source_fingerprint']
    assert actual['raw_sources']==original['raw_sources'] and actual['canonical_sources']==original['canonical_sources']
    assert original['source_fingerprint']=='7c7ba88609951b1b6a4be672984bf7adc3dc1bc7ababaeff7afa5d9ab10173b7' and gate['accepted'] is False
    assert original['pre_run_commit']=='1c0c7192f65464b79afa7a525e121f76543218c1'
    return {'original_scientific_freeze_sha256':digest(O/'source_freeze.json'),'original_failed_resource_gate_sha256':digest(O/'resource_gate.json'),'original_scope_failure_sha256':digest(O/'scope_failure.json'),'original_pre_run_commit':original['pre_run_commit'],'original_scientific_fingerprint':original['source_fingerprint']}

def check_all_sources(packet,origin):
    science.source_check()
    for rel,h in packet['raw_sources'].items():assert digest(ROOT/rel)==h,('Scheduling source drift',rel)
    assert initial_premises()==origin

def event(path,kind,**data):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':kind,**data})+'\n')

def completed_condition(folder,freeze,total,n):
    path=folder/'joint_summary.json'
    if not path.exists():return None
    result=json.loads(path.read_text());assert result['source_freeze_sha256']==freeze
    data=json.loads((folder/'posterior_manifest.json').read_text());assert data['n_exact_chance_deals']==total and data['n_recorded_decision_worlds']==total*4*n
    assert digest(folder/'posterior_supports.npz')==data['posterior_sha256']
    assert data['raw_hashes']=={p.name:digest(p) for p in sorted((folder/'raw').glob('*.npz'))}
    assert result['raw_chunks_sha256']=={p.name:digest(p) for p in sorted((folder/'joint_KL').glob('*.npz'))}
    models=json.loads((folder/'models/frozen_models.json').read_text());assert models['source_freeze_sha256']==freeze and models['posterior_sha256']==data['posterior_sha256']
    for seed,r in enumerate(models['models']):assert digest(folder/f'models/s{seed}/selected.npz')==r['selected_model_sha256']
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args();packet=own_packet();origin=initial_premises()
    if args.preflight:
        assert not list(O.glob('*/models/*')) and not list(O.glob('*/joint_summary.json'))
        save(H/'continuation_preflight.json',{**packet,**origin,'scope':'Resource amendment source only; original failed gate preserved. No model/test outcome existed at allocation.'});print(packet['scheduling_source_fingerprint']);return
    review=H/'CONTINUATION_REVIEW.md';assert review.exists() and 'N2_RESOURCE_CODEX_REVIEWED '+packet['scheduling_source_fingerprint'] in review.read_text(encoding='utf-8')
    commit=os.environ['N2_CONTINUATION_COMMIT']
    for rel,h in packet['canonical_sources'].items():
        blob=subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT);assert hashlib.sha256(blob.replace(b'\r\n',b'\n')).hexdigest()==h
    deadline=min(float(os.environ['JUDGEMENT_N2_CONTINUE_DEADLINE']),time.time()+10800);check_deadline(deadline)
    assert not (O/'summary.json').exists(),'Fixed targets already complete; use verifier instead of restarting writers'
    folder=O/'continuation';folder.mkdir(exist_ok=True);manifest={**packet,**origin,'scheduling_pre_run_commit':commit,'source_review_authority':'self-reviewed(Codex), pending external implementation/result audit; independent owner authorization','scope':'Resource scheduling only; scientific target/producers/hypotheses/counts/seeds unchanged; original gate remains a failure.'}
    path=folder/'source_manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:save(path,manifest)
    jobs=sorted(folder.glob('job_*.json'));assert len(jobs)<2,'Resource amendment allows at most two additional windows'
    jobid=len(jobs)+1;jobpath=folder/f'job_{jobid}.json';save(jobpath,{'job':jobid,'startedUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'absolute_cutoff_epoch':deadline,'source_manifest_sha256':digest(path),'status':'started; see append-only event log for completion'})
    journal=folder/'events.jsonl';event(journal,'window_started',job=jobid,deadline=deadline,source_manifest_sha256=digest(path));timer=threading.Timer(deadline-time.time(),lambda:os._exit(124));timer.daemon=True;timer.start()
    def source_check():check_all_sources(packet,origin)
    def status(text):science.status('COUNT_CONTINUATION '+text);event(journal,'checkpoint',job=jobid,message=text)
    try:
        results=[]
        for S,n,tr,total in science.CONFIGS:
            check_deadline(deadline);source_check();g=science.ToyGame(N=4,S=S,R=4,n=n,trump=tr);f=science.folder(S,n,tr)
            done=completed_condition(f,origin['original_scientific_freeze_sha256'],total,n)
            if done is not None:
                results.append(done);science.clear_mid();status(f'Completedconditionhash-verifiedwithoutrewriting:S{S}n{n}trump{tr}');continue
            gen=science.deals(tuple(g.deck),n);list(itertools.islice(gen,256));start=256
            while start<total:
                check_deadline(deadline);source_check();hands=list(itertools.islice(gen,min(1024,total-start)));assert hands
                before=start;science.write_chunk(g,f/'raw',start,hands,science.myopic_bid,science.ta_card,origin['original_scientific_freeze_sha256'],deadline);start+=len(hands);status(f'Exacttrajectorysaved{start}/{total}deals:S{S}n{n}trump{tr}');event(journal,'raw_chunk_verified',job=jobid,condition=[S,n,tr],start=before,end=start)
            manifest_data=science.merge(g,f/'raw',total,deadline);assert manifest_data['source_freeze_sha256']==origin['original_scientific_freeze_sha256'];status(f'Exactposteriorcomplete:S{S}n{n}trump{tr};{manifest_data["n_private_play_infosets"]}keys')
            corpus=science.Corpus(g,f);science.fit_models(corpus,f/'models',origin['original_scientific_freeze_sha256'],deadline,source_check,status);result=science.evaluate(corpus,f/'models',f/'joint_KL',origin['original_scientific_freeze_sha256'],deadline,source_check,status,science.myopic_bid);results.append(result);science.clear_mid()
            save(folder/'completed_conditions.json',{'conditions':[v['condition'] for v in results],'scheduling_manifest_sha256':digest(path),'original_failed_gate_preserved':True})
        summary={'conditions':results,'source_freeze_sha256':origin['original_scientific_freeze_sha256'],'scheduling_manifest_sha256':digest(path),'original_resource_gate_failed':True,'status':'Fixed scientific targets complete via explicit resource-only continuation; separate verification/external result audit pending','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        save(O/'summary.json',summary);event(journal,'all_targets_complete',job=jobid);status('AllN2fixedtargetscomplete; original resource failure retained and resource extension explicit, no publication.')
    finally:timer.cancel()
if __name__=='__main__':main()
