"""Check N2resource-extension provenance, then unchanged science verification."""
import os,sys,json,hashlib,time
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H))
import count_continue as continuation
from export import digest

def main():
    O=continuation.O;folder=O/'continuation';record=json.loads((folder/'source_manifest.json').read_text());origin=continuation.initial_premises()
    for k,v in origin.items():assert record[k]==v
    for rel,h in record['raw_sources'].items():assert digest(continuation.ROOT/rel)==h
    summary=json.loads((O/'summary.json').read_text());assert summary['original_resource_gate_failed'] is True and summary['scheduling_manifest_sha256']==digest(folder/'source_manifest.json')
    jobs=sorted(folder.glob('job_*.json'));assert 1<=len(jobs)<=2
    events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()];assert any(e['event']=='all_targets_complete' for e in events)
    import verify
    verify.main()
    report={'original_failed_gate_unchanged':True,'immutable_scientific_source_freeze_checked':True,'resource_only_schedule_source_checked':True,'additional_windows':len(jobs),'scientific_verifier_receipt_sha256':digest(O/'separate_verification.json'),'status':'self-verified(Codex), resource amendment/result external audit pending','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (O/'continuation_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
