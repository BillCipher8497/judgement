"""Oracle free-potential KL values and exact-moment entropy witnesses."""
from lib13 import *
from scipy.optimize import linprog,minimize
from scipy.special import logsumexp
def main():
    O.mkdir(exist_ok=True);paths=[H/p for p in ('lib13.py','run.py','verify.py','self_check.py','PROTOCOL.md','BOUND.md')]+[R/'N9/results/verification.json',R/'N9/n9lib.py']+[R/'N9/results'/tag/'data.npz' for tag in TAGS];sources={str(p):sha(p) for p in paths};fp=O/'sources.json'
    if fp.exists():assert json.loads(fp.read_text())==sources
    else:save(fp,sources)
    chosen={tag:selection(tag,B.rows(B.load(R/'N9/results'/tag/'data.npz'))) for tag in TAGS};sel={tag:[r.kid for r in rs] for tag,rs in chosen.items()};p=O/'selection.json'
    if p.exists():assert json.loads(p.read_text())==sel
    else:save(p,sel)
    summaries={}
    for tag,rs in chosen.items():
        folder=O/tag;folder.mkdir(exist_ok=True);rows=[]
        for j,r in enumerate(rs):
            check();path=folder/f'{r.kid:07d}.json'
            if path.exists():rows.append(json.loads(path.read_text()));continue
            A,counts=matrix(r);mu=counts[1:]/r.count;feat=A[1:].T
            def objective(theta):
                s=feat@theta;lz=logsumexp(s);p=np.exp(s-lz);return float(lz-theta@mu-np.log(r.count)),feat.T@p-mu
            opt=minimize(objective,np.zeros(len(mu)),jac=True,method='L-BFGS-B',options={'maxiter':1000,'gtol':1e-10,'ftol':1e-14});value,grad=objective(opt.x)
            outside=np.ones(len(r.F));outside[r.posterior]=0;lp=linprog(-outside,A_eq=A,b_eq=counts/r.count,bounds=(0,None),method='highs',options={'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            P={int(i):F(1,r.count) for i in r.posterior};proposal={i:F(float(v)).limit_denominator(10**9) for i,v in enumerate(lp.x) if v>0} if lp.success else {};ok=bool(lp.success) and feasible(proposal,A,counts,r.count);q=proposal if ok else P
            best=None;eps=F(0);mixed=P
            for power in [0]+list(range(1,41)):
                e=F(0) if power==0 else F(1,2**power);w={i:(1-e)*P.get(i,F(0))+e*q.get(i,F(0)) for i in P.keys()|q.keys()};w={i:v for i,v in w.items() if v};h=-math.fsum(float(v)*math.log(float(v)) for v in w.values())
                if best is None or h>best:best=h;eps=e;mixed=w
            assert feasible(mixed,A,counts,r.count);lower,upper=entropy_gain(mixed,r.count);cert=max(F(0),lower);assert float(cert)<=value+1e-7
            row={'tag':tag,'key_id':r.kid,'reach':r.count,'n_structural_worlds':len(r.F),'n_posterior_worlds':r.count,'lp_success':bool(lp.success),'lp_status':str(lp.message),'lp_proposal_exact_feasible':ok,'lp_proposal_weights':{str(k):str(v) for k,v in proposal.items()},'fallback_P_if_infeasible':not ok,'epsilon':str(eps),'mixture_weights':{str(k):str(v) for k,v in mixed.items()},'entropy_gain_enclosure_exact_nats':[str(lower),str(upper)],'certified_KL_lower_nats':str(cert),'numeric_free_potential_KL':value,'numeric_gradient_linf':float(np.max(np.abs(grad))),'numeric_optimizer_success':bool(opt.success),'numeric_optimizer_message':str(opt.message),'numeric_iterations':int(opt.nit),'theta':opt.x.tolist(),'scope':'Exact rational moment/entropy certificate under pending external lemma; numeric potential objective not a rigorous optimum/certificate. Known deterministic toy keys, no samplingCI.'};save(path,row);rows.append(row);save(O/'progress.json',{'tag':tag,'complete':j+1,'target':len(rs),'utc':stamp()})
        reach=sum(a['reach'] for a in rows);lower=sum((F(a['certified_KL_lower_nats'])*a['reach'] for a in rows),F(0))/reach;summary={'n_keys':len(rows),'reach':reach,'n_strict_positive_certificates':sum(F(a['certified_KL_lower_nats'])>0 for a in rows),'n_exact_LP_reconstruction_failures':sum(not a['lp_proposal_exact_feasible'] for a in rows),'n_numerical_optimizer_failures':sum(not a['numeric_optimizer_success'] for a in rows),'reach_weighted_certified_lower_exact_nats':str(lower),'reach_weighted_numeric_potential_KL':sum(a['reach']*a['numeric_free_potential_KL'] for a in rows)/reach,'sampling_CI':None};summaries[tag]=summary;save(folder/'summary.json',summary)
    assert all(sha(Path(p))==v for p,v in sources.items());save(O/'complete.json',{'utc':stamp(),'summaries':summaries,'primary_largeNT':summaries['S3R4n3_t-1'],'sources_sha256':sha(fp),'status':'complete, separate arithmetic audit/Claude external pending'});log(H,'N13fixed projection/witness diagnostic complete; separate arithmetic audit next, pendingClaude/local.')
if __name__=='__main__':main()
