"""Second numerical path for predeclared N1 smoothing and calibration reports."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,math,hashlib,time
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;O=H/'results'

def intervals(delta):
    matrix=np.asarray(delta).reshape(20,176);mean=float(matrix.mean());half=1.96*math.sqrt(float(matrix.var(axis=0,ddof=1).sum())/20)/176
    return [mean-half,mean+half]

def main():
    prop=[];truth=[];u=[];b=[];offset=[0]
    for path in sorted((O/'confirmation').glob('*.npz')):
        a=np.load(path);prop.append(a['P']);truth.append(a['Y']);u.append(a['uniform']);b.append(a['bi']);offset.extend((a['offset'][1:]+offset[-1]).tolist())
    prop=np.concatenate(prop);truth=np.concatenate(truth);samplers={'uniform':np.concatenate(u),'bi':np.concatenate(b)};offset=np.array(offset);pred=np.load(O/'confirmation_predictions.npz');report=json.loads((O/'sparsity_diagnostics.json').read_text());n=np.diff(offset)
    def losses(p):return np.add.reduceat(-np.log(p[np.arange(len(truth)),truth]),offset[:-1])/n
    net=losses(pred['net_full']);checks=0
    for name,raw in samplers.items():
        assert np.all(raw*256==np.rint(raw*256))
        for label,a in [('declared_floor',256*.001/.999),('1',1),('4',4),('16',16)]:
            smooth=(256/(256+a))*raw+(a/(256+a))*prop;ci=intervals(net-losses(smooth));np.testing.assert_allclose(ci,report['sensitivity'][name][label]['net_minus_sampler']['ci95'],rtol=0,atol=1e-11);checks+=1
        zero=raw[np.arange(len(truth)),truth]==0;fraction=np.add.reduceat(zero.astype(int),offset[:-1])/n;np.testing.assert_allclose(intervals(fraction),report['zero_count'][name]['round_equal_fraction']['ci95'],rtol=0,atol=1e-12);assert int(zero.sum())==report['zero_count'][name]['n_zero_card_labels']
    original=json.loads((O/'summary.json').read_text());calibration_checks=0
    for name,p in pred.items():
        bins=np.zeros((3520,3,10));counts=np.zeros(10,np.int64)
        for i,(a,b) in enumerate(zip(offset[:-1],offset[1:])):
            confidence=np.max(p[a:b],axis=1);correct=np.argmax(p[a:b],axis=1)==truth[a:b];which=np.minimum((confidence*10).astype(int),9)
            for k in range(10):
                take=which==k;counts[k]+=int(take.sum());bins[i,:,k]=[take.mean(),float(confidence[take].sum()/n[i]),float(correct[take].sum()/n[i])]
        aggregate=np.mean(bins,axis=0);entries=report['reliability_with_CIs'][name]['bins']
        for k,entry in enumerate(entries):
            assert entry['n_raw_cards']==int(counts[k]) and entry['n_rounds_with_cards_in_bin']==int((bins[:,0,k]>0).sum())
            if aggregate[0,k]:
                assert abs(entry['predicted_confidence']['mean']-aggregate[1,k]/aggregate[0,k])<1e-12
                assert abs(entry['observed_accuracy']['mean']-aggregate[2,k]/aggregate[0,k])<1e-12
        rng=np.random.default_rng([2110000000,0x4E31,90]);strata=[np.arange(s,3520,176) for s in range(176)];multiplicity=[]
        for _ in range(2000):multiplicity.append(np.bincount(np.concatenate([rng.choice(s,20,replace=True) for s in strata]),minlength=3520))
        frequency=np.array(multiplicity,dtype=float)/3520;boot=(frequency@bins.reshape(3520,30)).reshape(2000,3,10);ece=np.sum(abs(boot[:,2]-boot[:,1]),axis=1)
        np.testing.assert_allclose(np.quantile(ece,[.025,.975]),original['calibration'][name]['ci95'],rtol=0,atol=1e-12)
        for k,entry in enumerate(entries):
            valid=boot[:,0,k]>0
            for axis,label in ((1,'predicted_confidence'),(2,'observed_accuracy')):
                if valid.any():np.testing.assert_allclose(np.quantile(boot[valid,axis,k]/boot[valid,0,k],[.025,.975]),entry[label]['ci95'],rtol=0,atol=1e-12)
        calibration_checks+=1
    result={'status':'self-reviewed(Codex),pendingexternalresultaudit','smoothing_CIs_recomputed_independently':checks,'zero_count_CIs_recomputed':2,'all_calibration_bin_counts_means_and_bootstrap_CIs_recomputed':calibration_checks,'n_decisions':3520,
        'methods':'reshape/variance CI, algebraically equivalent pseudocount law, reduceat loss aggregation, bootstrap multiplicity-matrix path independent of index-mean path','checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (O/'secondary_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
