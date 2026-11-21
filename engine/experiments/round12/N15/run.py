"""Exact primal/entropy witness for a single known conditional-product family."""
import os,sys,json,time
from pathlib import Path
from fractions import Fraction as F
H=Path(__file__).resolve().parent;R=H.parent;O=H/'results';sys.path.insert(0,str(R/'N13'))
from lib13 import B,save,sha,stamp,log,logbounds
TAG='S3R4n3_t-1';KID=105013
def factors(n):
    out={};d=2
    while d*d<=n:
        while n%d==0:out[d]=out.get(d,0)+1;n//=d
        d+=1
    if n>1:out[n]=out.get(n,0)+1
    return out
def formal_log(x,weight):
    a=factors(x.numerator);b=factors(x.denominator);return {p:weight*(a.get(p,0)-b.get(p,0)) for p in a.keys()|b.keys()}
def add(*terms):
    out={}
    for t in terms:
        for p,v in t.items():out[p]=out.get(p,F(0))+v
    return {p:v for p,v in out.items() if v}
def main():
    assert time.time()<float(os.environ['JUDGEMENT_JOB_DEADLINE']);O.mkdir(exist_ok=True);paths=[H/'run.py',H/'verify.py',H/'PROTOCOL.md',R/'N13/results/smallest_positive_example.json',R/'N13/results/verification.json',R/'N9/results'/TAG/'data.npz',R/'N13/lib13.py'];sources={str(p):sha(p) for p in paths};save(O/'sources.json',sources);r=next(r for r in B.rows(B.load(R/'N9/results'/TAG/'data.npz')) if r.kid==KID);hero,hand,hist=B.decode(r.key);assert (hero,hand,hist)==(0,(2,4,6),(0,0,2,0,2,0,3,1,11,10));assert r.posterior.tolist()==[0,5,10] and len(r.F)==12
    q={0:F(2,9),5:F(4,9),10:F(2,9),6:F(1,9)};P={i:F(1,3) for i in (0,5,10)};mu=[[F(sum(int(r.F[i,c]==k) for i in P),3) for k in range(3)] for c in range(4)]
    for c in range(4):
        for k in range(3):assert sum((v for i,v in q.items() if r.F[i,c]==k),F(0))==mu[c][k]
    entropy=add(*[formal_log(v,-v) for v in q.values()],formal_log(F(3),F(-1)));kl=add(*[formal_log(P[i]/q[i],P[i]) for i in P]);assert kl==entropy=={2:F(-4,3),3:F(1)}
    polynomial={};world_coeff=[]
    for owners in r.F:
        coeff=1;power=0
        for c,k0 in enumerate(owners):
            k=int(k0)
            if mu[c][k]==0:power+=1
            elif (c,k) in ((0,0),(1,2)):coeff*=2
        world_coeff.append((coeff,power));polynomial[power]=polynomial.get(power,0)+coeff
    assert polynomial[0]==9;assert all(world_coeff[i][1]==0 for i in P);assert {i:F(v,9) for i,(v,p) in enumerate(world_coeff) if p==0}==q
    a,b=logbounds(F(3));c,d=logbounds(F(2));lo=a-F(4,3)*d;hi=b-F(4,3)*c;assert lo>0
    secondary=None;path=R/'N12/results'/TAG/f'{KID:07d}.npz'
    if path.exists():
        t=B.load(path);s=t['Q'][t['posterior']].astype('int64').sum(0);secondary={'source_sha256':sha(path),'actions':t['actions'].tolist(),'exact_posterior_score_sums':s.tolist(),'support_n':len(t['posterior']),'kind':'descriptive existing same-context values, no samplingCI, no case selection by reward'}
    result={'utc':stamp(),'tag':TAG,'key_id':KID,'hero':hero,'own_initial_hand':hand,'public_history':hist,'structural_owner_arrays':r.F.tolist(),'posterior_indices':r.posterior.tolist(),'moments':[[str(v) for v in row] for row in mu],'qstar':{str(i):str(v) for i,v in q.items()},'formal_infimum_prime_log_coefficients':{str(p):str(v) for p,v in kl.items()},'partition_polynomial_coefficients':{str(p):v for p,v in polynomial.items()},'world_weight_coefficient_and_epsilon_power':world_coeff,'strict_infimum_enclosure_exact_nats':[str(lo),str(hi)],'approx_nats':float((lo+hi)/2),'secondary_existing_private_actions':secondary,'scope':'Exact finite single-context infimum in product-family closure under entropy lemma; not finite-positive attainment, full posterior/game optimum, population rate or novelty. New proof pending external audit.'};save(O/'certificate.json',result);assert all(sha(Path(p))==v for p,v in sources.items());log(H,'N15candidate exact certificate constructed; separate fullreach/RREF/polynomial check next, pendingexternal/local.')
if __name__=='__main__':main()
