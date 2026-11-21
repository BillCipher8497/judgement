"""Run one script in this process with a required absolute UTC epoch cutoff."""
import os,sys,time,threading,runpy
from pathlib import Path
cut=float(os.environ['JUDGEMENT_JOB_DEADLINE'])
if time.time()>=cut:raise SystemExit('Absolute job cutoff already expired')
script=Path(sys.argv[1]).resolve();sys.argv=sys.argv[1:];sys.path.insert(0,str(script.parent))
timer=threading.Timer(cut-time.time(),lambda:os._exit(124));timer.daemon=True;timer.start()
try:runpy.run_path(str(script),run_name='__main__')
finally:timer.cancel()
