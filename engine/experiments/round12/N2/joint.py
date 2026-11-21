"""Exact finite allocation normalizers for the proposed N2 joint laws.
No sampler, learned policy, dataset writer or core modification.
"""
from functools import lru_cache
from itertools import combinations
import math
import numpy as np

def allocation_problem(capacities, allowed):
    caps=tuple(map(int,capacities));allowed=np.asarray(allowed,dtype=bool)
    assert len(caps)==3 and allowed.ndim==2 and allowed.shape[1]==3
    U=len(allowed);assert sum(caps)==U and min(caps)>=0
    masks=tuple(sum(int(allowed[i,j])<<i for i in range(U)) for j in range(3))
    full=(1<<U)-1
    def choices(j,remaining):
        if j==2:
            if remaining.bit_count()==caps[j] and not remaining&~masks[j]:yield remaining
            return
        valid=[i for i in range(U) if (remaining&masks[j])>>i&1]
        for subset in combinations(valid,caps[j]):yield sum(1<<i for i in subset)
    return caps,masks,full,choices

def feasible_count(capacities,allowed,hand_allowed=None):
    _,_,full,choices=allocation_problem(capacities,allowed)
    @lru_cache(None)
    def rec(j,left):
        if j==3:return int(left==0)
        return sum(rec(j+1,left^hand) for hand in choices(j,left) if hand_allowed is None or hand_allowed(j,hand))
    return rec(0,full)

def integer_partition(capacities,allowed,hand_weight):
    """Exact sum of products of nonnegative integer per-hand weights."""
    _,_,full,choices=allocation_problem(capacities,allowed)
    @lru_cache(None)
    def rec(j,left):
        if j==3:return int(left==0)
        value=0
        for hand in choices(j,left):
            w=hand_weight(j,hand)
            assert isinstance(w,(int,np.integer)) and w>=0
            value+=int(w)*rec(j+1,left^hand)
        return value
    return rec(0,full)

def log_partition(capacities,allowed,hand_logweight):
    """log sum_{feasible allocations} product_j exp(hand_logweight(j,mask)).
    mask indexes rows in allowed, not global card ids. All support laws are
    normalized over exactly the same structural support.
    """
    _,_,full,choices=allocation_problem(capacities,allowed)
    @lru_cache(None)
    def rec(j,left):
        if j==3:return 0. if left==0 else -math.inf
        terms=[]
        for hand in choices(j,left):
            tail=rec(j+1,left^hand)
            head=float(hand_logweight(j,hand))
            if tail!=-math.inf and head!=-math.inf:terms.append(head+tail)
        if not terms:return -math.inf
        largest=max(terms)
        return largest+math.log(math.fsum(math.exp(v-largest) for v in terms))
    result=rec(0,full)
    assert math.isfinite(result),'Empty legal support or nonpositive model'
    return result

def card_product_partition(capacities,allowed,probabilities):
    p=np.asarray(probabilities,dtype=float);allowed=np.asarray(allowed,bool)
    assert p.shape==allowed.shape and np.all(p[allowed]>0)
    logp=np.zeros_like(p);logp[allowed]=np.log(p[allowed]);cache={}
    def weight(j,mask):
        key=(j,mask)
        if key not in cache:cache[key]=math.fsum(logp[i,j] for i in range(len(p)) if mask>>i&1)
        return cache[key]
    return log_partition(capacities,allowed,weight)

def product_kl(reach_count,posterior_card_counts,capacities,allowed,probabilities):
    """Uniform reachable-deal posterior versus normalized card-product law.
    posterior_card_counts[c,j] counts exact reachable deals assigning c to j.
    The returned number uses floating logs of exact population quantities;
    it is not a sampling estimate or a rational optimization certificate.
    """
    counts=np.asarray(posterior_card_counts);allowed=np.asarray(allowed,bool);p=np.asarray(probabilities,float)
    assert reach_count>=1 and np.issubdtype(counts.dtype,np.integer)
    assert counts.shape==allowed.shape==p.shape
    assert np.all(counts.sum(1)==reach_count) and np.all(counts[~allowed]==0)
    assert np.all(counts.sum(0)==np.asarray(capacities)*reach_count)
    logz=card_product_partition(capacities,allowed,p)
    expected=math.fsum(int(counts[i,j])*math.log(p[i,j])/reach_count for i,j in zip(*np.nonzero(counts)))
    answer=logz-expected-math.log(reach_count)
    assert answer>=-1e-10,('Negative KL beyond numerical tolerance',answer)
    return {'KL_float':max(0.,answer),'raw_KL_float':answer,'log_normalizer':logz,'expected_log_factor':expected,'posterior_support_count':int(reach_count)}

def allocations(capacities,allowed):
    """Separate explicit support enumeration for direct-normalizer checks."""
    caps=tuple(map(int,capacities));allowed=np.asarray(allowed,bool);U=len(allowed)
    def rec(left,j,hands):
        if j==3:
            if not left:yield tuple(hands)
            return
        for h in combinations(left,caps[j]):
            if all(allowed[i,j] for i in h):yield from rec(tuple(i for i in left if i not in h),j+1,hands+[h])
    yield from rec(tuple(range(U)),0,[])
