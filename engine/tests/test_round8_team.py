"""Exact toy check of the perfect-information relaxation."""
import importlib.util
from pathlib import Path
import numpy as np
from types import SimpleNamespace

def test_pi_relaxation_with_frozen_policy():
    path=Path(__file__).resolve().parents[1]/'experiments/round8/T5/run.py'
    spec=importlib.util.spec_from_file_location('round8_team',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    t=m.Tree(m.ToyGame(N=3,S=2,R=3,n=2,trump=-1))
    fixed=t.uniform()
    none=np.zeros(t.n_is,bool)
    assert abs(m.pi_value(t,fixed,none)-m.profile_value(t,fixed)[0])<1e-10
    free=(t.is_pl!=0)&~t.is_bid
    assert m.pi_value(t,fixed,free)<=m.profile_value(t,fixed)[0]+1e-10

def test_support_certificate_rejects_a_bad_action_in_support():
    path=Path(__file__).resolve().parents[1]/'experiments/round8/T5/certify.py'
    spec=importlib.util.spec_from_file_location('round8_certificate',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    t=SimpleNamespace(U=np.array([[0.,0.],[0.,0.],[10.,0.]]),
        is_valid=np.array([[True,True]]),node_is=np.array([0,-1,-1]),
        e_ch=np.array([1,2]),e_ia=np.array([0,1]),
        chunks=[(0,2,np.array([0]),np.array([0]))],
        node_c=np.ones(3),roots=np.array([0]))
    fixed=np.array([[1.,0.]]);free=np.array([True])
    broad,_=m.certify(t,fixed,free,np.array([[.5,.5]]))
    assert not broad['certified'] and broad['max_pointwise_excess']==10
    narrow,_=m.certify(t,fixed,free,fixed)
    assert narrow['certified'] and narrow['PI_value_fraction']=='0'
