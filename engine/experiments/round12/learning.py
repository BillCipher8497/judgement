"""Small, auditable NumPy networks; observable inputs only, no game-state access."""
import numpy as np

OBJECTIVES = ('hard_ce', 'uniform_ce', 'set_bce', 'q_reg', 'direct')


def initialize(widths, seed):
    rng = np.random.default_rng(seed); params = []
    for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
        w = rng.normal(0., .01 if i == len(widths)-2 else np.sqrt(2/(a+b)), (a, b))
        params.extend([w, np.zeros(b)])
    return params


def forward(x, params):
    acts = [x]
    for i in range(0, len(params), 2):
        z = acts[-1] @ params[i] + params[i+1]
        acts.append(np.tanh(z) if i < len(params)-2 else z)
    return acts[-1], acts


def softmax(z, mask):
    z = np.where(mask, z, -np.inf)
    e = np.exp(z-z.max(1, keepdims=True))
    return e/e.sum(1, keepdims=True)


def head_loss(z, mask, target, weights, objective):
    """Returns minibatch mean loss and its logit derivative. Fixed weights."""
    assert z.shape == mask.shape == target.shape and mask.any(1).all()
    w = weights[:, None]/len(z); legal_n = mask.sum(1, keepdims=True)
    if objective in ('hard_ce', 'uniform_ce'):
        p = softmax(z, mask)
        logp = np.where(mask, z, -np.inf)
        maximum = logp.max(1, keepdims=True)
        logp = np.where(mask, logp-maximum-np.log(np.exp(logp-maximum).sum(1, keepdims=True)), 0.)
        loss = -(w*target*logp).sum(); grad = w*(p-target)
    elif objective == 'set_bce':
        p = np.exp(-np.logaddexp(0., -z))
        loss = (w*mask*(np.logaddexp(0., z)-target*z)/legal_n).sum()
        grad = w*mask*(p-target)/legal_n
    elif objective == 'q_reg':
        diff = mask*(z-target)
        loss = (.5*w*diff**2/legal_n).sum(); grad = w*diff/legal_n
    elif objective == 'direct':
        p = softmax(z, mask); mean = (p*target).sum(1, keepdims=True)
        loss = (w*mean).sum(); grad = w*p*(target-mean)
    else: raise ValueError(objective)
    return float(loss), np.where(mask, grad, 0.)


def loss_gradient(x, mask, target, weights, params, objective, l2=1e-4):
    z, acts = forward(x, params)
    value, dz = head_loss(z, mask, target, weights, objective)
    grads = [None]*len(params)
    for i in range(len(params)-2, -1, -2):
        value += .5*l2*float((params[i]**2).sum())
        grads[i] = acts[i//2].T @ dz+l2*params[i]; grads[i+1] = dz.sum(0)
        if i: dz = (dz @ params[i].T)*(1-acts[i//2]**2)
    return value, grads


class Adam:
    def __init__(self, params, lr=1e-3):
        self.m = [np.zeros_like(p) for p in params]; self.v = [p.copy() for p in self.m]
        self.t = 0; self.lr = lr

    def step(self, params, grads):
        self.t += 1
        norm = np.sqrt(sum(float((g*g).sum()) for g in grads))
        scale = min(1., 10/max(norm, 1e-300))
        for p, g, m, v in zip(params, grads, self.m, self.v):
            g = g*scale; m *= .9; m += .1*g; v *= .999; v += .001*g*g
            p -= self.lr*(m/(1-.9**self.t))/(np.sqrt(v/(1-.999**self.t))+1e-8)
        assert all(np.isfinite(p).all() for p in params)


def decisions(x, mask, params, objective):
    z, _ = forward(x, params)
    return (np.where(mask, z, np.inf).argmin(1) if objective == 'q_reg'
            else np.where(mask, z, -np.inf).argmax(1))
