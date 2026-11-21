"""Observable, canonical-offset neural bidder; new module, old policies untouched."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROUND = HERE.parent; ENGINE = HERE.parents[2]
sys.path[:0] = [str(ENGINE), str(ROUND)]
from judgement.bidding.model import BidModel, ModelBidder
from judgement.bidding.features import hand_features
from learning import forward, Adam, softmax

spec = importlib.util.spec_from_file_location('round12_frozen_r4_observables', ENGINE/'experiments/round11/E3/predictor.py')
features = importlib.util.module_from_spec(spec); spec.loader.exec_module(features)
WIDTH = len(features.names('B'))-1+2
MODEL_PATH = ENGINE/'experiments/phase1/results/model_ta_final.json'


def canonical_probabilities(ctx, model):
    # This path takes only the named observable context, never a full state.
    assert set(ctx) == {'hand', 'n', 'position', 'prior', 'legal'}
    x = hand_features(ctx['hand'], -1, ctx['n'], ctx['position'], ctx['prior'], 4)
    p = model.get(ctx['n'], -1).dist_all_bids(x)
    return np.diag(p)[ctx['legal']]


def design(ctx, model):
    base = canonical_probabilities(ctx, model); clipped = np.clip(base, 1e-6, 1-1e-6)
    offset = np.log(clipped)-np.log1p(-clipped)
    x = np.column_stack((features.design(ctx, 'B')[:, 1:], offset,
                         base*(10+np.array(ctx['legal']))/23))
    assert x.shape == (len(base), WIDTH)
    return x, offset, base


def initialize(seed):
    rng = np.random.default_rng(seed); widths = [WIDTH, 64, 32, 1]; params = []
    for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
        params.extend([np.zeros((a, b)) if i == 2 else rng.normal(0., np.sqrt(2/(a+b)), (a, b)), np.zeros(b)])
    return params


def probability(z): return np.exp(-np.logaddexp(0., -z))


def loss_gradient(x, mask, offset, scores, params, objective, bce_norm=1., l2=1e-4):
    B, A, F = x.shape; flat = x.reshape(-1, F); residual, acts = forward(flat, params)
    z = offset+residual.reshape(B, A); p = probability(z)
    payoff = 10+np.arange(A)[None, :]; legal_n = mask.sum(1, keepdims=True)
    if objective == 'bce':
        hit = scores/payoff; weight = mask*payoff/(23*legal_n*bce_norm*B)
        value = float((weight*(np.logaddexp(0., z)-hit*z)).sum()); dz = weight*(p-hit)
    elif objective == 'direct':
        pi = softmax(payoff*p, mask); target = scores/23
        mean = (pi*target).sum(1, keepdims=True)
        value = -float(mean.mean()); dz = -pi*(target-mean)*payoff*p*(1-p)/B
    else: raise ValueError(objective)
    dz = (dz*mask).reshape(-1, 1); grads = [None]*len(params)
    for i in range(len(params)-2, -1, -2):
        value += .5*l2*float((params[i]**2).sum())
        grads[i] = acts[i//2].T @ dz+l2*params[i]; grads[i+1] = dz.sum(0)
        if i: dz = (dz @ params[i].T)*(1-acts[i//2]**2)
    return value, grads


class Ensemble:
    def __init__(self, folder, packet, base_model):
        self.model = base_model; self.mu = np.array(packet['mu']); self.sd = np.array(packet['sd'])
        self.alpha = packet['alpha']; self.params = []
        for name in packet['selected_models']:
            saved = dict(np.load(Path(folder)/name)); self.params.append([saved[f'p{i}'] for i in range(6)])

    def probabilities(self, ctx):
        x, offset, base = design(ctx, self.model)
        corrected = np.mean([probability(offset+forward((x-self.mu)/self.sd, p)[0][:, 0]) for p in self.params], axis=0)
        return (1-self.alpha)*base+self.alpha*corrected

    def choose(self, ctx):
        return ctx['legal'][int(np.argmax((10+np.array(ctx['legal']))*self.probabilities(ctx)))]


class NTResidualBidder:
    def __init__(self, ensemble, canonical): self.ensemble, self.canonical = ensemble, canonical

    def bid(self, info, legal, rng):
        if info.trump != -1 or self.ensemble.alpha == 0:
            return self.canonical.bid(info, legal, rng)
        return self.ensemble.choose(features.context(info, legal))
