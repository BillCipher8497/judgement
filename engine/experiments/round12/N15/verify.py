"""Separate reach enumeration, rational row reduction and polynomial reader."""
import os,sys,json,time,importlib.util
from pathlib import Path
from fractions import Fraction as F
H=Path(__file__).resolve().parent;R=H.parent;O=H/'results';sys.path.insert(0,str(R/'N9'))
import n9lib as B
from reference import direct_posterior,public
from bounded_ta import ta_card
from judgement.cfr.heuristics import myopic_bid
from common import save,sha,stamp,log
def main():
    assert time.time()<float(os.environ['JUDGEMENT_JOB_DEADLINE']);a=json.loads((O/'certificate.json').read_text());sources=json.loads((O/'sources.json').read_text());assert all(sha(Path(p))==v for p,v in sources.items());g=B.game(a['tag']);hero=a['hero'];hand=tuple(a['own_initial_hand']);hist=tuple(a['public_history']);ww,reached=direct_posterior(g,hero,hand,hist,myopic_bid,ta_card);played,_,_=public(g,hist);cards=sorted(set(g.deck)-set(hand)-set(hist[4:]));opps=[(hero+k)%4 for k in (1,2,3)];owners=[]
    for w in ww:owners.append(tuple(next(k for k,p in enumerate(opps) if c in w[p]) for c in cards))
    assert set(owners)==set(map(tuple,a['structural_owner_arrays']));posterior={owners[i] for i,w in enumerate(ww) if w in reached};assert len(reached)==3 and posterior=={tuple(a['structural_owner_arrays'][i]) for i in a['posterior_indices']};Fowners=a['structural_owner_arrays'];mu=[[F(sum(int(v[c]==k) for v in posterior),3) for k in range(3)] for c in range(4)];assert [[str(v) for v in row] for row in mu]==a['moments'];face=[i for i,w in enumerate(Fowners) if all(mu[c][k]>0 for c,k in enumerate(w))];assert set(face)=={0,5,6,10};order=[0,5,10,6]
    matrix=[[F(1)]*4+[F(1)]]+[[F(int(Fowners[i][c]==k)) for i in order]+[mu[c][k]] for c in range(4) for k in range(3)];row=0;pivots=[]
    for col in range(4):
        pivot=next((i for i in range(row,len(matrix)) if matrix[i][col]),None)
        if pivot is None:continue
        matrix[row],matrix[pivot]=matrix[pivot],matrix[row];div=matrix[row][col];matrix[row]=[x/div for x in matrix[row]]
        for i in range(len(matrix)):
            if i!=row:
                div=matrix[i][col];matrix[i]=[x-div*y for x,y in zip(matrix[i],matrix[row])]
        pivots.append(col);row+=1
    assert pivots==[0,1,2];base=[matrix[i][-1] for i in range(3)]+[F(0)];direction=[-matrix[i][3] for i in range(3)]+[F(1)];assert base==[F(1,3)]*3+[F(0)] and direction==[-1,1,-1,1];q=[b+F(1,9)*d for b,d in zip(base,direction)];assert q==[F(2,9),F(4,9),F(2,9),F(1,9)] and {str(i):str(v) for i,v in zip(order,q)}==a['qstar'];assert q[0]*q[2]==q[1]*q[3]
    # Independently multiply monomial factors as polynomial dictionaries.
    poly={}
    for w in Fowners:
        p={0:1}
        for c,k in enumerate(w):
            f={1:1} if mu[c][k]==0 else {0:2 if (c,k) in ((0,0),(1,2)) else 1};out={}
            for j,x in p.items():
                for k0,y in f.items():out[j+k0]=out.get(j+k0,0)+x*y
            p=out
        for j,x in p.items():poly[j]=poly.get(j,0)+x
    assert {str(k):v for k,v in poly.items()}==a['partition_polynomial_coefficients'] and poly[0]==9
    # The one-dimensional entropy derivative vanishes when (1/3-t)^2=t(1/3+t).
    # Exact solution t=1/9; entropy is strictly concave on this nonconstant line.
    t=F(1,9);assert (F(1,3)-t)**2==t*(F(1,3)+t);assert a['formal_infimum_prime_log_coefficients']=={'2':'-4/3','3':'1'}
    spec=importlib.util.spec_from_file_location('n15_independent_log_reader',R/'N13/verify.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);l3,h3=m.fresh_log(F(3));l2,h2=m.fresh_log(F(2));lo=l3-F(4,3)*h2;hi=h3-F(4,3)*l2;oldlo,oldhi=map(F,a['strict_infimum_enclosure_exact_nats']);assert 0<oldlo<=lo<=hi<=oldhi
    save(O/'verification.json',{'utc':stamp(),'all12_structural_worlds_and3_reaching_worlds_independently_reenumerated':True,'rational_RREF_affine_moment_family':[list(map(str,base)),list(map(str,direction))],'independent_partition_polynomial':poly,'exact_qstar_and_infimum_matching_checks_pass':True,'alternate_log_enclosure_pass':True,'source_sha256':sha(Path(__file__)),'certificate_sha256':sha(O/'certificate.json'),'scope':'Separate Codex mathematical/algebra/reach check, new single-context closure-infimum proof pending Claude external; not a generic solved posterior floor/game optimum.'});log(H,'N15separate reach/RREF/polynomial/log proof checks pass; pendingClaudeexternal/local.')
if __name__=='__main__':main()
