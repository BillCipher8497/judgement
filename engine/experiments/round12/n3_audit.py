"""Serial N3 separate verification/report driver; no new scientific samples."""
import os,sys,runpy,time,json,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;O=H/'N3/results'
def main():
    deadline=float(os.environ['N3_AUDIT_DEADLINE']);assert time.time()<deadline and (O/'summary.json').exists();sys.path.insert(0,str(H/'N3'))
    scripts=[H/'N3/verify.py',H/'n3_report.py'];manifest={'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scripts_sha256':{str(p.relative_to(H)):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},'deadline_epoch':deadline,'scope':'Separate checks on original frozen dataset/models/seeded games; no new experiment, samples or tuning','startedUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};(O/'audit_driver_manifest.json').write_text(json.dumps(manifest,indent=2))
    for script in scripts:
        assert time.time()<deadline;print('AUDIT START',script.name,flush=True);runpy.run_path(str(script),run_name='__main__');print('AUDIT PASS',script.name,flush=True)
    (O/'audit_driver_complete.json').write_text(json.dumps({'status':'All N3separate checks passed, self-checked(Codex), external audit pending','completedUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2))
if __name__=='__main__':main()
