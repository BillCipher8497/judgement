"""Finite moment/entropy witness helpers; no experiment on import."""
import os,sys,json,time,math,hashlib
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
H=Path(__file__).resolve().parent;R=H.parent;O=H/'results';sys.path.insert(0,str(R/'N9'))
import n9lib as B
import numpy as np
from fractions import Fraction as F
from functools import lru_cache
from common import save,sha,stamp,log
TAGS=B.TAGS;D=1<<128
def check():assert time.time()<float(os.environ['JUDGEMENT_JOB_DEADLINE'])
def down(x):return F(x.numerator*D//x.denominator,D)
def up(x):return -down(-x)
def unit(x,n):
    assert F(1)<=x<=F(2);z=(x-1)/(x+1);z2=z*z;p=z;s=F(0)
    for j in range(n):s+=2*p/F(2*j+1);p*=z2
    return s,s+2*p/(F(2*n+1)*(1-z2))
@lru_cache(None)
def logbounds(x,n=32):
    x=F(x);assert x>0;e=x.numerator.bit_length()-x.denominator.bit_length();y=x/F(2**e) if e>=0 else x*2**(-e)
    while y<1:e-=1;y*=2
    while y>2:e+=1;y/=2
    lo,hi=unit(y,n);a,b=unit(F(2),n)
    lo,hi=(lo+e*a,hi+e*b) if e>=0 else (lo+e*b,hi+e*a)
    return down(lo),up(hi)
def entropy_gain(q,Rcount):
    lo=F(0);hi=F(0)
    for a in q.values():
        if a:
            l,h=logbounds(a);lo-=a*h;hi-=a*l
    a,b=logbounds(F(Rcount));return down(lo-b),up(hi-a)
def matrix(r):
    U=len(r.x);A=np.vstack([np.ones(len(r.F)),np.array([(r.F[:,c]==k).astype(float) for c in range(U) for k in range(3)])]);counts=np.array([int(A[j,r.posterior].sum()) for j in range(len(A))]);return A,counts
def feasible(q,A,counts,Rcount):
    return all(v>=0 for v in q.values()) and all(sum((v*int(A[j,i]) for i,v in q.items()),F(0))==F(int(counts[j]),Rcount) for j in range(len(A)))
def selection(tag,rows):
    return sorted((r for r in rows if r.split==2),key=lambda r:hashlib.sha256(json.dumps(['N13',tag,bytes(r.key).ljust(20,b'\0').hex()],separators=(',',':')).encode()).digest())[:64]
