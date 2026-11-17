import sys
from pathlib import Path
import numpy as np
from judgement.cfr.toy import ToyGame,Tree
from judgement.cfr.heuristics import profile
from judgement.cfr.solve import profile_value,best_response
_old_path=sys.path[:]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/round9/D3'))
from frozen_tree import FrozenTree
from compact_tree import CompactTree
from root_bound import root_information_bound
from reachable_ta import reachable_ta_profile
sys.path[:]=_old_path

def pi(t,sig,free):
    v=t.U[:,0].copy()
    for a,b,seg,parents in reversed(t.chunks):
        q=v[t.e_ch[a:b]]
        ordinary=np.add.reduceat(q*sig.ravel()[t.e_ia[a:b]],seg)
        v[parents]=np.where(free[t.node_is[parents]],np.minimum.reduceat(q,seg),ordinary)
    return t.node_c[t.roots]@v[t.roots]

def test_frozen_tree_preserves_values_and_private_best_responses():
    for trump in [0,-1]:
        g=ToyGame(N=3,S=2,R=3,n=2,trump=trump)
        full=Tree(g);reduced=FrozenTree(g)
        sf=profile(full,play='ta');sr=profile(reduced,play='ta')
        full.U[:,1:]=-full.U[:,0,None]
        free=(full.is_pl!=0)&~full.is_bid
        np.testing.assert_allclose(profile_value(full,sf)[0],profile_value(reduced,sr)[0],atol=1e-12)
        np.testing.assert_allclose(pi(full,sf,free),pi(reduced,sr,np.ones(reduced.n_is,bool)),atol=1e-12)
        for p in [1,2]:
            np.testing.assert_allclose(best_response(full,sf,p,free=free),best_response(reduced,sr,p),atol=1e-12)
        assert all(p!=0 and len(hist)>=g.N for p,_,hist in reduced.is_key)
        compact=CompactTree(g);sc=profile(compact,play='ta')
        assert (compact.n_nodes,compact.n_is,compact.n_edges)==(reduced.n_nodes,reduced.n_is,reduced.n_edges)
        np.testing.assert_allclose(profile_value(compact,sc),profile_value(reduced,sr),atol=1e-12)
        np.testing.assert_allclose(pi(compact,sc,np.ones(compact.n_is,bool)),pi(reduced,sr,np.ones(reduced.n_is,bool)),atol=1e-12)
        for p in [1,2]:np.testing.assert_allclose(best_response(compact,sc,p),best_response(reduced,sr,p),atol=1e-12)
        bound=root_information_bound(compact)
        from fractions import Fraction
        lower=float(Fraction(bound['first_decision_hidden_lower_fraction']))
        assert pi(compact,sc,np.ones(compact.n_is,bool))<=lower+1e-12
        assert lower<=profile_value(compact,sc)[0]+1e-12
        sparse,_=reachable_ta_profile(compact)
        np.testing.assert_allclose(profile_value(compact,sparse),profile_value(compact,sc),atol=1e-12)
        hs,ht=sparse.copy(),sc.copy()
        for p in [1,2]:
            vs,bs=best_response(compact,sparse,p,return_policy=True)
            vt,bt=best_response(compact,sc,p,return_policy=True)
            np.testing.assert_allclose(vs,vt,atol=1e-12)
            np.testing.assert_array_equal(bs[compact.is_pl==p],bt[compact.is_pl==p])
            hs[compact.is_pl==p]=bs[compact.is_pl==p];ht[compact.is_pl==p]=bt[compact.is_pl==p]
        np.testing.assert_allclose(profile_value(compact,hs),profile_value(compact,ht),atol=1e-12)
