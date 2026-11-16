"""Checks for non-leading-hero LC0 and dealer legality."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def module(task):
    spec=importlib.util.spec_from_file_location('round8_'+task,ROOT/'experiments'/'round8'/task/'run.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_nonleading_first_trick_model_is_unconditional():
    m=module('T4')
    W=np.array([[0,1,0],[1,0,1],[0,0,1],[0,1,1]],float)
    result=m.models(W,W)
    for pmf in result.values():
        assert abs(pmf.sum()-1)<1e-12
        assert (pmf>=0).all()
    # Independently enumerate the 8 trick outcomes under fitted LC0.
    import itertools
    L=W[:,:-1];Y=W[:,1:]
    prob=[(Y[L==l].sum()+2*W.mean())/((L==l).sum()+2) for l in [0,1]]
    expected=np.zeros(4)
    for seq in itertools.product([0,1],repeat=3):
        q=W[:,0].mean() if seq[0] else 1-W[:,0].mean()
        for j in [1,2]:q*=prob[seq[j-1]] if seq[j] else 1-prob[seq[j-1]]
        expected[sum(seq)]+=q
    np.testing.assert_allclose(result['LC0'],expected)


def test_dealer_hook_excludes_the_empirical_mode():
    m=module('T4')
    law=np.array([.1,.4,.5])
    assert m.best_legal_bid(law,[0,1,2])==2
    assert m.best_legal_bid(law,[0,1])==1
