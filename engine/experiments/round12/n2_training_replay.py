"""Recompute NT seed0 optimization to cross-check the I/O-interrupted trajectory.

Reuses frozen neural gradient/Adam definitions, but gathers batches and weights
by vectorized offsets. No new fit seed, tuning, data or selected model output.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
from pathlib import Path
H=Path(__file__).resolve().parent/'N2';sys.path.insert(0,str(H))
import run as science
import numpy as np
from neural import initialize,loss_gradient
from learning import Adam
from corpus import BASE,DOMAIN
from export import digest
def main():
    deadline=float(os.environ['JUDGEMENT_N2_AUDIT_DEADLINE']);folder=science.O/'S3R4n3_t-1';g=science.ToyGame(N=4,S=3,R=4,n=3,trump=-1);c=science.Corpus(g,folder);ids=c.selected(0,30000);pack=c.pack(ids);mean=float(pack['count'].mean());p=initialize(np.random.SeedSequence([BASE,DOMAIN,20,0]));adam=Adam(p);old=folder/'models/s0';orphan=[]
    for path in old.glob('checkpoint.tmp.orphan_*'):
        with np.load(path) as a:
            if int(a['epoch'])==2:orphan.append(path)
    assert len(orphan)==1
    matches=[]
    for epoch in range(1,21):
        assert time.time()<deadline;order=np.random.default_rng([BASE,DOMAIN,21,0,epoch]).permutation(len(ids))
        for start in range(0,len(order),128):
            assert time.time()<deadline;b=order[start:start+128];lo=pack['offset'][b];hi=pack['offset'][b+1];length=hi-lo;index=np.concatenate([np.arange(a,z,dtype=np.int64) for a,z in zip(lo,hi)]);per=np.divide(pack['count'][b],mean*len(b)*length,out=np.zeros(len(b),float),where=length>0);weights=np.repeat(per,length)
            _,grads=loss_gradient(pack['X'][index],pack['M'][index],pack['P'][index],pack['T'][index],weights,p);adam.step(p,grads)
        if epoch in (2,20):
            target=orphan[0] if epoch==2 else old/'checkpoint.npz'
            with np.load(target) as a:
                assert int(a['epoch'])==epoch and int(a['step'])==adam.t
                for name,arrays in (('p',p),('m',adam.m),('v',adam.v)):
                    for k,value in enumerate(arrays):np.testing.assert_array_equal(value,a[f'{name}{k}'])
            matches.append({'epoch':epoch,'target_sha256':digest(target),'parameters_and_Adam_state_bit_exact':True})
    result={'status':'pass, self-checked(Codex); external audit pending','condition':'S3R4n3_t-1','fit_seed_index':0,'n_replayed_epochs':20,'checkpoints':matches,'scope':'Original gradient/optimizer definitions reused; vectorized batch gathering independently reconstructs interrupted epoch2and finalepoch20state. This is a numerical replay, not another stochastic experiment or new estimate.','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};(science.O/'training_trajectory_replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
