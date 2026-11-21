"""Baseline-preserving score models and matched objective gradients."""
from common import np,OBJECTIVES,BASE,DOMAIN
from learning import forward,head_loss,initialize,Adam
def init(arm,seed):
    widths=[683,2] if arm in (5,6) else [683,64,32,2]
    p=initialize(widths,np.random.SeedSequence([BASE,DOMAIN,20,seed]))
    p[-2][:]=0;p[-1][:]=0
    return p
def scores(x,base,p,arm,alpha):
    f,_=forward(x,p)
    return (1-alpha)*base+alpha*f if arm==7 else base+alpha*f
def gradient(x,base,target,weight,p,arm,l2=1e-4):
    f,acts=forward(x,p);s=f if arm==7 else base+f;obj=OBJECTIVES[arm]
    # All deployed policies maximize a score. For regression the supervised
    # quantity is cost=-score, so the score derivative changes sign.
    loss,dz=head_loss(-s if obj=='q_reg' else s,np.ones(s.shape,bool),target,weight,obj)
    if obj=='q_reg':dz=-dz
    grads=[None]*len(p)
    for i in range(len(p)-2,-1,-2):
        loss+=.5*l2*float((p[i]**2).sum());grads[i]=acts[i//2].T@dz+l2*p[i];grads[i+1]=dz.sum(0)
        if i:dz=(dz@p[i].T)*(1-acts[i//2]**2)
    return loss,grads
def targets(q,arm):
    # q[round,world,action] contains hit indicators, not actual hidden truth.
    maximum=q.max(2,keepdims=True);opt=q==maximum
    if arm in (0,5):return np.eye(2)[q.argmax(2)].mean(1)
    if arm==1:return (opt/opt.sum(2,keepdims=True)).mean(1)
    if arm in (2,6,7):return opt.mean(1)
    return (maximum-q).mean(1)
def policy_scores(x,base,models,arm):
    return np.mean([scores(x,base,m['params'],arm,m['alpha']) for m in models],axis=0)
