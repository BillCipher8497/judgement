"""Serial N2 verification driver; one bounded worker, no new experiments."""
import os,runpy,time,json,hashlib,gc
from pathlib import Path
H=Path(__file__).resolve().parent;O=H/'N2/results'
def main():
    deadline=float(os.environ['JUDGEMENT_N2_AUDIT_DEADLINE']);assert time.time()<deadline and (O/'summary.json').exists()
    scripts=[H/'N2/verify_continuation.py',H/'n2_training_replay.py',H/'n2_windows_check.py',H/'n2_report.py'];manifest={'deadline_epoch':deadline,'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scripts_sha256':{str(p.relative_to(H)):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},'startedUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'Post-outcome independent verification only, no new scientific seed/sample/selection. Exact commands are runpy versions of the listed standalone scripts.'};(O/'audit_driver_manifest.json').write_text(json.dumps(manifest,indent=2))
    for script in scripts:
        assert time.time()<deadline
        print('AUDIT START',script.name,flush=True);runpy.run_path(str(script),run_name='__main__');gc.collect();print('AUDIT PASS',script.name,flush=True)
    (O/'audit_driver_complete.json').write_text(json.dumps({'status':'All serial audit scripts passed, self-checked(Codex), pending external review','completedUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2))
if __name__=='__main__':main()
