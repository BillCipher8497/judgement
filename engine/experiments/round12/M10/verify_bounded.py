"""Serial read-only science audit and derived diagnostic checks, absolute stop."""
import os,time,threading,runpy,json,itertools
from pathlib import Path
from datetime import datetime,timezone
H=Path(__file__).resolve().parent
stop=datetime(2026,11,17,23,45,tzinfo=timezone.utc).timestamp()
assert time.time()<stop
timer=threading.Timer(stop-time.time(),lambda:os._exit(124));timer.daemon=True;timer.start()
try:
    runpy.run_path(str(H/'verify.py'),run_name='__main__')
    runpy.run_path(str(H.parent/'M2/diagnose.py'),run_name='__main__')
    allows=[{0,1},{1,2},{2,0}]
    feasible=[p for p in itertools.permutations(range(3)) if all(c in a for c,a in zip(p,allows))]
    assert set(feasible)=={(0,1,2),(1,2,0)}
    edges=[]
    for p in feasible:
        for i,j in itertools.combinations(range(3),2):
            q=list(p);q[i],q[j]=q[j],q[i]
            if tuple(q) in feasible:edges.append((p,q))
    assert not edges
    (H.parent/'AUDIT_FIXES/verification.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
        'counterexamples_computed':True,'D5_disconnected_feasible_worlds':feasible,'D5_nonnull_edges':edges,
        'M2_decoder_ranges_recomputed':True,'status':'ready-for-Claude-review'},indent=2)+'\n',encoding='utf-8')
finally:timer.cancel()
