"""Separate provenance checks for N2's explicitly recorded Windows I/O adapter."""
import json,hashlib,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent/'N2';ROOT=H.parents[3];O=H/'results'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    m=json.loads((O/'windows_io/runtime_manifest.json').read_text());p=H/'windows_io_resume.py';assert sha(p)==m['adapter_raw_sha256'];rel=str(p.relative_to(ROOT)).replace('\\','/');blob=subprocess.check_output(['git','show',f'{m["adapter_commit"]}:{rel}'],cwd=ROOT)
    canon=lambda b:hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest();assert canon(blob)==canon(p.read_bytes());test=json.loads((H/'windows_io_self_check.json').read_text());assert test['status']=='pass' and test['source_sha256']==sha(p)
    summary=json.loads((O/'summary.json').read_text());assert summary['original_resource_gate_failed'] is True
    scientific=json.loads((O/'source_freeze.json').read_text())
    for rel,digest in scientific['raw_sources'].items():assert sha(ROOT/rel)==digest
    extension=json.loads((O/'continuation/source_manifest.json').read_text())
    for rel,digest in extension['raw_sources'].items():assert sha(ROOT/rel)==digest
    jobs=[json.loads(p.read_text()) for p in sorted((O/'continuation').glob('job_*.json'))];assert len(jobs)==2 and [j['job'] for j in jobs]==[1,2]
    events=[json.loads(s) for s in (O/'windows_io/events.jsonl').read_text().splitlines()];reads=[v for v in events if v['event']=='memory_backed_checkpoint_read'];assert reads and reads[0]['sha256']=='58e573725f37a3762bc55f868d62c8910708f6fc899135b5decaf9fd279b913f'
    assert (O/'continuation_verification.json').exists()
    result={'status':'self-checked(Codex), external implementation/results review pending','adapter_commit':m['adapter_commit'],'adapter_sha256':sha(p),'source_hashes_match_original_and_all_extensions':True,'windows_used':2,'memory_backed_checkpoint_reads':len(reads),'bounded_replace_retries':sum(v['event']=='transient_replace_retry' for v in events),'scope':'I/O-only adapter source/provenance and synthetic array/API equivalence; scientific raw/model/CI checks are the separate original verifier receipt.'};(O/'windows_io_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
