"""N3 paths, immutable source receipts and atomic checkpoints."""
import hashlib,json,os,sys,time,subprocess,platform
from pathlib import Path
H=Path(__file__).resolve().parent
ENGINE=H.parents[2];ROOT=ENGINE.parent;R12=H.parent
for p in (str(ENGINE),str(R12),str(H)):
    if p not in sys.path:sys.path.insert(0,p)
import numpy as np
BASE=2120000000;DOMAIN=0x4E33;O=H/'results'
ARMS=('hard','uniform','set','q','direct','linear-hard','linear-set','no-offset-set')
OBJECTIVES=('hard_ce','uniform_ce','set_bce','q_reg','direct','hard_ce','set_bce','set_bce')
COUNTS={'pilot':192,'train':9600,'validation':2400,'held':2400}
STAGES={'pilot':0,'train':1,'validation':2,'held':3}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def replace(source,target):
    # Result-directory sync/indexing can briefly deny Windows replacement.
    # Retry I/O only, bounded5s; the scientific absolute timer still applies.
    until=time.monotonic()+5
    while True:
        try:return source.replace(target)
        except PermissionError:
            if time.monotonic()>=until:raise
            time.sleep(.1)
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8');replace(tmp,path)
def arrays(path,**data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    if tmp.exists():tmp.rename(tmp.with_name(tmp.name+'.incomplete_'+str(time.time_ns())))
    with tmp.open('wb') as f:np.savez_compressed(f,**data)
    if path.exists():path.rename(path.with_name(path.name+'.previous_'+str(time.time_ns())))
    replace(tmp,path)
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def status(message):
    with (H/'STATUS.txt').open('a',encoding='utf-8') as f:f.write(utc()+' '+message+'\n')
    print(message,flush=True)
def check_time(deadline):
    if time.time()>=deadline:raise TimeoutError('N3 absolute UTC hard stop')
def source_packet():
    paths=sorted(list(H.glob('*.py'))+[H/'PROTOCOL.md']+list((ENGINE/'judgement').rglob('*.py'))+[R12/'learning.py',ENGINE/'experiments/phase1/results/model_ta_final.json'])
    raw={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in paths}
    canonical={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths}
    return {'raw_sources':raw,'canonical_sources':canonical,'source_fingerprint':hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()}
def verify_sources(freeze):
    for p,h in freeze['raw_sources'].items():assert digest(ROOT/p)==h,('N3 source drift',p)
def freeze_sources():
    packet=source_packet();review=(H/'SOURCE_REVIEW.md').read_text(encoding='utf-8')
    assert 'N3_CODEX_SOURCE_REVIEWED '+packet['source_fingerprint'] in review
    commit=os.environ['N3_SOURCE_COMMIT']
    for rel,h in packet['canonical_sources'].items():
        data=subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT)
        assert hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()==h
    import scipy
    result={**packet,'pre_run_source_commit':commit,'source_review':'self-reviewed(Codex), pending external implementation/result audit','numerical_stack':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'thread_limits':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}}
    path=O/'source_freeze.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:save(path,result)
    return result
def interval(values,strata=None):
    values=np.asarray(values,float);n=len(values);mean=float(values.mean());assert n>1
    if strata is None:variance=float(values.var(ddof=1)/n)
    else:
        strata=np.asarray(strata);variance=0.
        for s in np.unique(strata):
            v=values[strata==s];assert len(v)>1
            variance+=(len(v)/n)**2*float(v.var(ddof=1)/len(v))
    radius=1.959963984540054*np.sqrt(variance)
    return {'n':n,'mean':mean,'ci95':[mean-radius,mean+radius],'method':'paired normal; strata fixed' if strata is not None else 'normal'}
