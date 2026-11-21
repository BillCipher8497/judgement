"""Synthetic-only N2 DP/reference tests, to run serially after N1 releases worker."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import math
import numpy as np
from joint import feasible_count,integer_partition,log_partition,card_product_partition,product_kl,allocations

def main():
    for cap in ((0,0,0),(1,1,1),(2,1,1),(2,2,2),(3,3,3)):
        U=sum(cap);allowed=np.ones((U,3),bool)
        if U>2:allowed[0,1]=False
        explicit=list(allocations(cap,allowed));n=len(explicit)
        assert feasible_count(cap,allowed)==n and n>=1
        weight=lambda j,mask:1+(j+1)*sum(i+1 for i in range(U) if mask>>i&1)
        direct_integer=sum(math.prod(weight(j,sum(1<<i for i in hand)) for j,hand in enumerate(hands)) for hands in explicit)
        assert integer_partition(cap,allowed,weight)==direct_integer
        assert abs(log_partition(cap,allowed,lambda j,h:0.)-math.log(n))<1e-12
        prop=allowed*np.array(cap);prop=prop/prop.sum(1,keepdims=True) if U else prop.astype(float)
        counts=np.zeros((U,3),np.int64)
        for hands in explicit:
            for j,h in enumerate(hands):counts[list(h),j]+=1
        result=product_kl(n,counts,cap,allowed,prop)
        assert abs(result['raw_KL_float'])<1e-12,'Prop product must normalize to uniform'
        # Positive nonuniform factors, no stochastic seed consumed.
        p=np.exp(((np.arange(U)[:,None]+1)*(np.arange(3)[None,:]+1)%7)/5)*allowed
        if U:p/=p.sum(1,keepdims=True)
        terms=[math.prod(p[i,j] for j,h in enumerate(hands) for i in h) for hands in explicit]
        z=math.fsum(terms);logz=card_product_partition(cap,allowed,p);assert abs(logz-math.log(z))<1e-12
        chosen=explicit[:max(1,n//3)];k=len(chosen);counts=np.zeros((U,3),np.int64)
        for hands in chosen:
            for j,h in enumerate(hands):counts[list(h),j]+=1
        direct=math.fsum(math.log(1/k)-math.log(w/z) for w in terms[:k])/k
        assert abs(product_kl(k,counts,cap,allowed,p)['raw_KL_float']-direct)<1e-12
    # Exact disconnected two-world feasible support used only as a math fixture.
    a=np.array([[1,0,1],[1,1,0],[0,1,1]],bool);assert feasible_count((1,1,1),a)==2
    from neural import loss_gradient,probabilities
    from features import encode
    from judgement.cfr.toy import ToyGame
    g=ToyGame(N=4,S=2,R=4,n=2,trump=-1)
    obs=dict(seat=0,hand=[0,1],history=[1,0,0,0],n=2,S=2,R=4,trump=-1)
    xx,mm,pp,public=encode(g,obs);assert len(xx)==6 and not mm[:,3].any() and np.allclose(pp.sum(1),1)
    try:encode(g,dict(obs,hidden_hands=[[0,1],[2,3],[4,5],[6,7]]));raise AssertionError('accepted hidden input')
    except AssertionError as error:
        assert str(error)!='accepted hidden input'
    params=[.03*np.sin(np.arange(35*32).reshape(35,32)/11),np.linspace(.1,.3,32),.02*np.cos(np.arange(32)).reshape(32,1),np.array([.04])]
    x=np.sin(np.arange(2*4*35).reshape(2,4,35)/7);mask=np.array([[1,1,0,0],[1,1,1,0]],bool);prop=np.array([[.4,.6,0,0],[.2,.3,.5,0]])
    target=np.array([[.3,.7,0,0],[.1,.2,.7,0]]);weights=np.array([.4,.6]);value,gradient=loss_gradient(x,mask,prop,target,weights,params)
    for k,index in ((0,(3,7)),(0,(20,2)),(1,(5,)),(2,(7,0)),(3,(0,))):
        eps=1e-6;original=params[k][index];params[k][index]=original+eps;hi=loss_gradient(x,mask,prop,target,weights,params)[0]
        params[k][index]=original-eps;lo=loss_gradient(x,mask,prop,target,weights,params)[0];params[k][index]=original
        assert abs((hi-lo)/(2*eps)-gradient[k][index])<1e-7
    params[2][:]=0;params[3][:]=0
    np.testing.assert_allclose(probabilities(x,mask,prop,params),prop,rtol=0,atol=1e-12)
    empty_x=np.empty((0,4,35));empty_m=np.empty((0,4),bool);empty=np.empty((0,4))
    value,gradient=loss_gradient(empty_x,empty_m,empty,empty,np.empty(0),params)
    expected=.5e-4*float(np.sum(params[0]**2)+np.sum(params[2]**2))
    assert abs(value-expected)<1e-12
    np.testing.assert_allclose(gradient[0],1e-4*params[0],rtol=0,atol=1e-12)
    assert probabilities(empty_x,empty_m,empty,params).shape==(0,4)
    from determinism_check import check
    print(check())
    from export import key,decode
    from corpus import hand_partitions
    original=(1,(1,3),(0,0,0,1,0))
    packed=np.array([key(*original)],dtype='S20')[0]
    assert decode(packed)==original,'Trailing zero card must survive fixed-width key roundtrip'
    for S,n,expected in ((2,2,[16,6,6]),(3,3,[132,44,44])):
        a=hand_partitions(ToyGame(N=4,S=S,R=4,n=n,trump=0));b=hand_partitions(ToyGame(N=4,S=S,R=4,n=n,trump=-1))
        assert a==b and [sum(v==s for v in a.values()) for s in range(3)]==expected
    print('Synthetic exact allocation/DP/direct-KL, soft-label gradient and observable encoder checks pass; no game data generated')
if __name__=='__main__':main()


