"""Four-holder observable residual classifier; labels never enter encoding."""
import numpy as np
F=35
SCHEMA={'seat','n','trump','dealer','hand','history','bids','voids','won','left'}
HISTORY_COLUMNS=[18,19,20,21,22,26,27,28,30]
def encode(o,table):
    assert set(o)==SCHEMA
    seat,n=o['seat'],o['n'];opps=[(seat+k)%4 for k in (1,2,3)]
    seen=o['hand']
    for _,c in o['history']:seen|=1<<c
    cards=[c for c in range(52) if not (seen>>c)&1]
    cap=np.array([o['left'][p] for p in opps]+[52-4*n],float)
    assert cap.sum()==len(cards)
    x=np.zeros((len(cards),4,F),np.float32);mask=np.zeros((len(cards),4),bool)
    bids=dict(o['bids'])
    for i,c in enumerate(cards):
        suit,rank=divmod(c,13)
        unseen_suit=sum(cc//13==suit for cc in cards)
        for j in range(4):
            stock=j==3;p=opps[j] if not stock else None
            valid=cap[j]>0 and (stock or not (o['voids'][p]>>suit)&1)
            mask[i,j]=valid; x[i,j,rank]=1;x[i,j,13+j]=1
            b=0 if stock else bids[p];won=0 if stock else o['won'][p];need=b-won
            played=[] if stock else [cc%13 for pp,cc in o['history'] if pp==p and cc//13==suit]
            led=0 if stock else sum(o['history'][k][0]==p and o['history'][k][1]//13==suit for k in range(0,len(o['history']),4))
            x[i,j,17:]=[b/n,won/n,need/max(1,cap[j]),need>0,need==0,need<0,
                cap[j]/max(1,len(cards)),o['trump']!=-1 and suit==o['trump'],o['trump']==-1,
                len(played)/n,max(played)/12 if played else -1,led/n,unseen_suit/13,
                rank/12*(need>0),stock,table,n/13,0 if stock else (p-o['dealer']-1)%4/3]
    assert mask.any(1).all()
    prop=mask*cap;prop/=prop.sum(1,keepdims=True)
    return x,mask,prop,cards
def init(seed):
    rng=np.random.default_rng(seed)
    return [rng.normal(0,1/np.sqrt(F),(F,64)),np.zeros(64),np.zeros((64,1)),np.zeros(1)]
def predict(x,mask,prop,p):
    h=np.maximum(0,x.reshape(-1,F)@p[0]+p[1]);z=(h@p[2]+p[3]).reshape(-1,4)
    z=np.where(mask,z+np.log(np.maximum(prop,1e-300)),-np.inf)
    e=np.exp(z-z.max(1,keepdims=True));return e/e.sum(1,keepdims=True)
def loss_grad(x,mask,prop,y,weights,p,l2=1e-4):
    assert mask[np.arange(len(y)),y].all() and np.isfinite(weights).all()
    flat=x.reshape(-1,F);h=np.maximum(0,flat@p[0]+p[1]);prob=predict(x,mask,prop,p)
    ii=np.arange(len(y));mixed=.999*prob+.001*prop
    value=float(-(weights*np.log(mixed[ii,y])).sum())
    dz=prob.copy();dz[ii,y]-=1
    dz*= (weights*.999*prob[ii,y]/mixed[ii,y])[:,None];dz=dz.reshape(-1,1)
    dh=(dz@p[2].T)*(h>0)
    grads=[flat.T@dh+l2*p[0],dh.sum(0),h.T@dz+l2*p[2],dz.sum(0)]
    value+=.5*l2*float((p[0]**2).sum()+(p[2]**2).sum())
    return value,grads
def ablate(x,variant):
    if variant=='full':return x
    assert variant=='bids';a=x.copy();a[:,:,HISTORY_COLUMNS]=0;return a
def decision_losses(prob,y,offset):
    logs=-np.log(np.maximum(prob[np.arange(len(y)),y],1e-300))
    return np.array([logs[a:b].mean() for a,b in zip(offset[:-1],offset[1:])])
