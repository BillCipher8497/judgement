"""Separate reader for the all-candidate posthoc diagnostic, no reselection."""
from n10lib import *
from verify import forward_fresh,fresh_ci
import math
def main():
    check();parts=[load(p) for p in sorted((O/'data/validation').glob('*.npz'))];d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};values=d['q'].mean(1)*d['hit'][:,None];ix=np.arange(len(values));b=d['base'].argmax(1);receipt=json.loads((O/'candidate_diagnostics.json').read_text());assert receipt['n_candidates']==240;rows=receipt['records'];checked=0;summary={}
    for arm in ARMS:
        aa=[r for r in rows if r['arm']==arm];summary[arm]={'n_candidates':len(aa),'n_with_changed_validation_actions':sum(r['n_changed']>0 for r in aa),'max_changed_contexts':max(r['n_changed'] for r in aa),'max_validation_score_delta':max(r['actual_score_minus_baseline']['mean'] for r in aa),'scope':'posthoc on1536selection-reused contexts, no deployment or independent effect claim'}
        for seed in range(5):
            for epoch in EPOCHS:
                check();z=load(O/'models'/f'{arm}_s{seed}'/f'candidate_{epoch}.npz');p=[z[f'p{k}'] for k in range(len(z))];f=forward_fresh(d['x'],p)
                for alpha in ALPHAS:
                    r=next(a for a in aa if (a['seed'],a['epoch'],a['alpha'])==(seed,epoch,alpha));a=(d['base']+alpha*f).argmax(1);n=int((a!=b).sum());assert n==r['n_changed'];radius=math.sqrt(math.log(40)/(2*len(a)));assert np.allclose(r['changed_rate']['ci95'],[max(0,n/len(a)-radius),min(1,n/len(a)+radius)],rtol=0,atol=1e-12);v=values[ix,a]-values[ix,b];m,ci=fresh_ci(v,d['stratum']);c=r['actual_score_minus_baseline'];assert c['n']==1536 and abs(c['mean']-m)<1e-12 and np.allclose(c['ci95'],ci,rtol=0,atol=1e-12);checked+=1
    assert checked==240;save(O/'candidate_diagnostics_check.json',{'utc':stamp(),'n_candidates_independently_checked':checked,'independent_forwards_counts_CIs_match':True,'all_candidate_summary':summary,'diagnostic_sha256':sha(O/'candidate_diagnostics.json'),'source_sha256':sha(Path(__file__)),'pending_Claude_external':True});log(H,'All240posthoc candidate diagnostic independent forwards/counts/intervals pass; no selection changed.')
if __name__=='__main__':main()
