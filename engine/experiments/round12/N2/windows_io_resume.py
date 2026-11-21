"""Recorded Windows I/O-only runtime adapter; frozen science files untouched.

The original resume reader retains an NpzFile for checkpoint.npz while later
replacing that same filename, which can retain a Windows sharing lock. Read
those bytes into a memory-backed NpzFile instead. Arrays and NPZ API remain
unchanged. Separately retry transient PermissionError on result-file replace.
No numerical, seed, model, sample, selection, policy or stopping rule changes.
"""
import os,sys,io,time,json,hashlib,subprocess,tempfile
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[3];O=H/'results'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def memory_load(path,*args,loader=np.load,**kwargs):
    return loader(io.BytesIO(Path(path).read_bytes()),*args,**kwargs)
def retry_replace(source,target,replace,log,deadline_seconds=5.):
    stop=time.monotonic()+deadline_seconds;attempt=0
    while True:
        try:return replace(source,target)
        except PermissionError:
            attempt+=1;log(attempt)
            if time.monotonic()>=stop:raise
            time.sleep(.1)
def self_check():
    with tempfile.TemporaryDirectory(prefix='judgement_n2_io_fixture_') as folder:
        p=Path(folder)/'checkpoint.npz';values={'epoch':np.array(1),'p0':np.arange(64,dtype=float).reshape(8,8),'trace_row':np.array('fixture')};np.savez_compressed(p,**values)
        a=memory_load(p)
        for k,v in values.items():np.testing.assert_array_equal(a[k],v)
        # Keep memory NpzFile open while replacing actual disk file on Windows.
        other=p.with_suffix('.tmp')
        with other.open('wb') as f:np.savez_compressed(f,**values)
        other.replace(p)
        for k,v in values.items():np.testing.assert_array_equal(a[k],v)
        a.close();calls=[]
        def fixture(s,t):
            calls.append(1)
            if len(calls)<3:raise PermissionError('injected transient fixture')
            return 'done'
        assert retry_replace('source','target',fixture,lambda n:None)=='done' and len(calls)==3
    result={'status':'pass','array_and_NPZ_API_equivalence':True,'open_memory_NPZ_does_not_lock_disk_replacement':True,'bounded_transient_retry_fixture':True,'source_sha256':sha(__file__),'scope':'Synthetic I/O test only; no frozen scientific artifact modified','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (H/'windows_io_self_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
def main():
    if '--self-check' in sys.argv:self_check();return
    commit=os.environ['N2_WINDOWS_IO_COMMIT'];rel=str(Path(__file__).resolve().relative_to(ROOT)).replace('\\','/');blob=subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT);canonical=lambda b:hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest();assert canonical(blob)==canonical(Path(__file__).read_bytes())
    assert json.loads((H/'windows_io_self_check.json').read_text())['source_sha256']==sha(__file__)
    assert len(list((O/'continuation').glob('job_*.json')))==1 and not (O/'summary.json').exists()
    folder=O/'windows_io';folder.mkdir(exist_ok=True);receipt={'adapter_commit':commit,'adapter_raw_sha256':sha(__file__),'scientific_and_schedule_files':'unchanged; original source manifests and continuation source checks still enforced','reason':'WinError5 first continuation epoch2; original stderr retained. Resume source also leaves np.load(checkpoint.npz) open; memory-backed load prevents self-lock. Original first-error cause not independently established.','semantics':'Only checkpoint disk-handle lifetime and bounded result-file PermissionError retries; identical NPZ arrays/API. No changes to math, counts, seeds, selected completed conditions or targets.','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};(folder/'runtime_manifest.json').write_text(json.dumps(receipt,indent=2))
    original_load=np.load;original_replace=Path.replace
    def log(kind,**data):
        with (folder/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':kind,**data})+'\n')
    def scoped(p):
        try:return Path(p).resolve().is_relative_to(O.resolve())
        except TypeError:return False
    def load(p,*args,**kwargs):
        if scoped(p) and Path(p).name=='checkpoint.npz':
            log('memory_backed_checkpoint_read',path=str(p),sha256=sha(p));return memory_load(p,*args,loader=original_load,**kwargs)
        return original_load(p,*args,**kwargs)
    def replace(p,target):
        if scoped(p) and scoped(target):return retry_replace(p,target,original_replace,lambda n:log('transient_replace_retry',source=str(p),target=str(target),attempt=n))
        return original_replace(p,target)
    np.load=load;Path.replace=replace
    try:
        sys.path.insert(0,str(H));sys.argv=[str(H/'count_continue.py')];import count_continue;count_continue.main()
    finally:np.load=original_load;Path.replace=original_replace
if __name__=='__main__':main()
