"""Separate exact rational moment/log certificate and finite-potential reader."""
from lib13 import *
@lru_cache(None)
def fresh_log(x):
    # Alternate40-term enclosure and explicit integer dyadic rounding.
    x=F(x);assert x>0;y=x;e=0
    while y<1:y*=2;e-=1
    while y>2:y/=2;e+=1
    def series(v):
        z=(v-1)/(v+1);total=sum((2*z**(2*k+1)/F(2*k+1) for k in range(40)),F(0));tail=2*z**81/(F(81)*(1-z*z));return total,total+tail
    a,b=series(y);c,d=series(F(2));lo=a+e*(c if e>=0 else d);hi=b+e*(d if e>=0 else c);return F(lo.numerator*D//lo.denominator,D),F(-((-hi).numerator*D//(-hi).denominator),D)
def moments(q,r):
    assert all(v>=0 for v in q.values()) and sum(q.values(),F(0))==1
    for c in range(r.F.shape[1]):
        for k in range(3):
            actual=sum((v for i,v in q.items() if r.F[i,c]==k),F(0));target=F(sum(int(r.F[i,c]==k) for i in r.posterior),r.count);assert actual==target
def main():
    sources=json.loads((O/'sources.json').read_text());assert all(sha(Path(p))==v for p,v in sources.items());selected=json.loads((O/'selection.json').read_text());complete=json.loads((O/'complete.json').read_text());checks=[]
    for tag in TAGS:
        rs=selection(tag,B.rows(B.load(R/'N9/results'/tag/'data.npz')));assert [r.kid for r in rs]==selected[tag];rows=[]
        for j,r in enumerate(rs):
            check();a=json.loads((O/tag/f'{r.kid:07d}.json').read_text());q={int(i):F(v) for i,v in a['mixture_weights'].items()};moments(q,r)
            P={int(i):F(1,r.count) for i in r.posterior};prop={int(i):F(v) for i,v in a['lp_proposal_weights'].items()}
            if a['lp_proposal_exact_feasible']:moments(prop,r)
            else:prop=P;assert a['fallback_P_if_infeasible']
            eps=F(a['epsilon']);assert eps==0 or eps.numerator==1 and eps.denominator in {2**k for k in range(1,41)};expected={i:(1-eps)*P.get(i,F(0))+eps*prop.get(i,F(0)) for i in P.keys()|prop.keys()};expected={i:v for i,v in expected.items() if v};assert q==expected
            lo=F(0);hi=F(0)
            for v in q.values():
                l,h=fresh_log(v);lo-=v*h;hi-=v*l
            l,h=fresh_log(F(r.count));lo-=h;hi-=l;oldlo,oldhi=map(F,a['entropy_gain_enclosure_exact_nats']);assert oldlo<=lo<=hi<=oldhi;assert F(a['certified_KL_lower_nats'])==max(F(0),oldlo)
            theta=a['theta'];U=r.F.shape[1];scores=[math.fsum(theta[3*c+int(k)] for c,k in enumerate(w)) for w in r.F];mx=max(scores);den=math.fsum(math.exp(s-mx) for s in scores);lz=mx+math.log(den);prob=[math.exp(s-lz) for s in scores];mu=[sum(int(r.F[i,c]==k) for i in r.posterior)/r.count for c in range(U) for k in range(3)];value=lz-math.fsum(t*m for t,m in zip(theta,mu))-math.log(r.count);grad=[math.fsum(p for p,w in zip(prob,r.F) if int(w[c])==k)-mu[3*c+k] for c in range(U) for k in range(3)];assert abs(value-a['numeric_free_potential_KL'])<1e-8 and abs(max(map(abs,grad))-a['numeric_gradient_linf'])<1e-8;assert float(F(a['certified_KL_lower_nats']))<=value+1e-7;rows.append(a)
            if j%8==0:save(O/'verification_progress.json',{'tag':tag,'keys_checked':j+1,'target':64,'utc':stamp()})
        summary=complete['summaries'][tag];reach=sum(a['reach'] for a in rows);lower=sum((F(a['certified_KL_lower_nats'])*a['reach'] for a in rows),F(0))/reach;assert summary['n_keys']==64 and summary['reach']==reach and F(summary['reach_weighted_certified_lower_exact_nats'])==lower;assert summary['n_strict_positive_certificates']==sum(F(a['certified_KL_lower_nats'])>0 for a in rows);assert summary['n_exact_LP_reconstruction_failures']==sum(not a['lp_proposal_exact_feasible'] for a in rows);assert summary['n_numerical_optimizer_failures']==sum(not a['numeric_optimizer_success'] for a in rows);assert abs(summary['reach_weighted_numeric_potential_KL']-math.fsum(a['reach']*a['numeric_free_potential_KL'] for a in rows)/reach)<1e-10;checks.append({'tag':tag,'n_keys':64,'all_exact_moments_and_alternate_log_enclosures_pass':True})
    save(O/'verification.json',{'utc':stamp(),'checks':checks,'n_keys':256,'source_sha256':sha(Path(__file__)),'complete_sha256':sha(O/'complete.json'),'scope':'Separate Codex exact rational/moment/log-series and independent floating potential check. New entropy-bound proof and certificate implementation PENDING Claude external; no solved optimum or novel theorem claim.'});log(H,'N13 separate256key rational/alternate-log/finite-potential checks passed; pendingClaude/local.')
if __name__=='__main__':main()
