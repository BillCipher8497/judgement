"""Verification-only eager NPZ read adapter, not a scientific producer change.

Frozen verify.py repeatedly accesses lazy compressed NPZ members per record.
Read each immutable raw-chunk member once into a read-only mapping with NPZ
context API. No change to arrays, original verifier checks or raw file bytes.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib,runpy,tempfile
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;O=H/'N2/results'
class EagerNPZ(dict):
    @property
    def files(self):return list(self)
    def close(self):pass
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
def read(path,*args,loader=np.load,**kwargs):
    with loader(path,*args,**kwargs) as source:
        result=EagerNPZ({k:source[k] for k in source.files})
    for value in result.values():value.flags.writeable=False
    return result
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def self_check():
    with tempfile.TemporaryDirectory(prefix='judgement_n2_cache_fixture_') as folder:
        p=Path(folder)/'raw.npz';values={'keys':np.array([b'\x01\x04\x00\x00\x02\x00',b'abcd'],dtype='S20'),'holder_bits':np.array([0,2**24-1],np.uint32),'chosen_card':np.array([0,11],np.uint8),'deal_index':np.array([0,369599],np.uint32),'empty':np.empty((0,4))};np.savez_compressed(p,**values);before=sha(p)
        with np.load(p) as source:
            first=source['holder_bits'];second=source['holder_bits'];assert not np.shares_memory(first,second)
            with read(p) as eager:
                assert source.files==eager.files
                for k in values:np.testing.assert_array_equal(eager[k],source[k]);assert eager[k].dtype==source[k].dtype and not eager[k].flags.writeable
                assert eager['holder_bits'] is eager['holder_bits']
        assert sha(p)==before
    result={'status':'pass','arrays_dtypes_keys_empty_and_NPZ_context_API_match':True,'immutable_source_bytes_unchanged':True,'repeated_member_reads_cached':True,'original_NPZ_members_are_reloaded_on_repeated_getitem_observed':True,'source_sha256':sha(__file__),'scope':'Verification I/O fixture only, no scientific data modified','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};(H/'N2/cache_audit_self_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
def main():
    if '--self-check' in sys.argv:self_check();return
    test=json.loads((H/'N2/cache_audit_self_check.json').read_text());assert test['status']=='pass' and test['source_sha256']==sha(__file__)
    assert (O/'summary.json').exists();original=np.load;counts={};manifest={'adapter_source_sha256':sha(__file__),'scope':'Verification-only raw NPZ eager member cache; all original scientific sources/data/verifier checks unchanged','fixture':test,'startedUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};(O/'cache_audit_runtime_manifest.json').write_text(json.dumps(manifest,indent=2))
    def load(path,*args,**kwargs):
        try:p=Path(path).resolve();scoped=p.is_relative_to(O.resolve()) and p.parent.name=='raw' and p.suffix=='.npz'
        except TypeError:scoped=False
        if scoped:
            name=str(p.relative_to(O));counts[name]=counts.get(name,0)+1;return read(p,*args,loader=original,**kwargs)
        return original(path,*args,**kwargs)
    np.load=load
    try:
        runpy.run_path(str(H/'n2_audit.py'),run_name='__main__')
        # Additional action-definition replay strengthens the frozen verifier's
        # all-record legality/consistency test; same data, no new experiment.
        runpy.run_path(str(H/'n2_policy_actions.py'),run_name='__main__')
        (O/'cache_audit_verification.json').write_text(json.dumps({'status':'All original audit checks plus all private policy choices passed; self-checked(Codex), pending external audit','adapter_source_sha256':sha(__file__),'n_raw_chunk_snapshots':sum(counts.values()),'raw_chunk_snapshot_counts':counts,'extra_policy_checker_sha256':sha(H/'n2_policy_actions.py'),'scope':'Cache changes reads only; immutable array snapshots/dtypes checked in fixture. Includes original verifier, training trajectory, Windows-I/O provenance and extra policy action replay.','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2))
    finally:np.load=original
if __name__=='__main__':main()
