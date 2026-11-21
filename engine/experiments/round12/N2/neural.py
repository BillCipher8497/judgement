"""N2 grouped proper mixture loss; exact soft holder-count targets.
Independent of frozen N1 training; no game state or hidden deal encoding.
"""
import numpy as np
F=35

def initialize(seed):
    rng=np.random.default_rng(seed)
    return [rng.normal(0,1/np.sqrt(F),(F,32)),np.zeros(32),np.zeros((32,1)),np.zeros(1)]

def probabilities(x,mask,prop,params):
    hidden=np.maximum(0,x.reshape(-1,F)@params[0]+params[1]);score=(hidden@params[2]+params[3]).reshape(-1,4)
    score=np.where(mask,score+np.log(np.maximum(prop,1e-300)),-np.inf)
    score-=score.max(1,keepdims=True);p=np.exp(score);return p/p.sum(1,keepdims=True)

def loss_gradient(x,mask,prop,target,weights,params,l2=1e-4):
    """Weights already include key reach counts and per-key 1/unseen-card.
    Targets are exact posterior marginal probabilities, not one hard holder.
    Zero-unseen keys contribute zero data loss but retain objective denominator.
    """
    assert x.shape[1:]==(4,F) and mask.shape==prop.shape==target.shape==x.shape[:2]
    assert weights.shape==(len(x),) and np.all(weights>=0)
    np.testing.assert_allclose(target.sum(1),1,rtol=0,atol=1e-12)
    assert np.all(target[~mask]==0) and np.all(prop[mask]>0)
    p=probabilities(x,mask,prop,params);mixed=.999*p+.001*prop
    value=float(-np.sum(weights[:,None]*target*np.log(np.maximum(mixed,1e-300))))
    # v_j = target_j * .999*p_j/mixed_j; dL/dz = p*sum(v)-v.
    v=target*.999*p/np.where(mask,mixed,1.)
    dz=(weights[:,None]*(p*v.sum(1,keepdims=True)-v)).reshape(-1,1)
    flat=x.reshape(-1,F);pre=flat@params[0]+params[1];hidden=np.maximum(0,pre);dh=(dz@params[2].T)*(pre>0)
    gradient=[flat.T@dh+l2*params[0],dh.sum(0),hidden.T@dz+l2*params[2],dz.sum(0)]
    value+=.5*l2*float(np.sum(params[0]**2)+np.sum(params[2]**2))
    return value,gradient
