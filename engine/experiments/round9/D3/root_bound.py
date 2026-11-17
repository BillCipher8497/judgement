"""Optional exact first-decision information relaxation, for Claude review."""
from fractions import Fraction
from math import comb
import numpy as np

def root_information_bound(t):
    """Constrain the first spoiler decision by private infoset; reveal later.

    This remains a relaxation of every feasible hidden team. It may tighten PI,
    but need not equal the true hidden optimum. All root chance weights uniform.
    """
    expected=1
    for p in range(t.g.N):expected*=comb(len(t.g.deck)-p*t.g.n,t.g.n)
    if len(t.roots)!=expected:raise ValueError('This bound requires the complete ordered chance space')
    assert expected*(t.g.hit_base+t.g.n)<2**53
    v=t.U[:,0].astype(np.int64)
    np.testing.assert_array_equal(v,t.U[:,0])
    for a,b,seg,parents in reversed(t.chunks):v[parents]=np.minimum.reduceat(v[t.e_ch[a:b]],seg)
    n=len(t.roots);np.testing.assert_allclose(t.node_c[t.roots],np.full(n,1/n),atol=1e-15,rtol=0)
    pi_sum=int(v[t.roots].sum())
    at_root=np.zeros(t.n_nodes,bool);at_root[t.roots]=True
    edges=at_root[t.e_par]
    aggregated=np.bincount(t.e_ia[edges],weights=v[t.e_ch[edges]],minlength=t.n_is*t.g.A)
    np.testing.assert_array_equal(aggregated,np.rint(aggregated))
    aggregated=aggregated.astype(np.int64).reshape(t.n_is,t.g.A)
    decisions=t.roots[t.node_pl[t.roots]>=0]
    infos=np.unique(t.node_is[decisions])
    terminal=t.roots[t.node_pl[t.roots]<0]
    coupled_sum=int(v[terminal].sum())
    if len(infos):coupled_sum+=int(np.min(np.where(t.is_valid[infos],aggregated[infos],2**60),axis=1).sum())
    assert coupled_sum>=pi_sum
    return {'PI_fraction':str(Fraction(pi_sum,n)),
            'first_decision_hidden_lower_fraction':str(Fraction(coupled_sum,n)),
            'lower_bound_gain_fraction':str(Fraction(coupled_sum-pi_sum,n)),
            'n_exact_deals':n,'root_private_infosets':len(infos),
            'uncertainty':'Complete chance space, integer sums; no sampling CI. A lower bound, not a solved team value.'}
