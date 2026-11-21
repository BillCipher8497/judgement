"""Synthetic entropy-variational and scaled-log fixtures, no game selection."""
from lib13 import *
from verify import fresh_log
def main():
    check();q={i:F(1,4) for i in range(4)};lo,hi=entropy_gain(q,2);assert F(69,100)<lo<hi<F(70,100)
    for x in (F(1),F(2),F(1,3),F(1,2**40),F(2**40),F(17,129),F(129,17)):
        a,b=logbounds(x);c,d=fresh_log(x);assert a<=c<=d<=b
    assert logbounds(F(1))==(0,0);save(H/'self_check.json',{'utc':stamp(),'synthetic_uniform_four_vs_correlated_two_entropy_gain_enclosure':[str(lo),str(hi)],'n_scaled_rational_log_cases':7,'alternate40term_containment':True,'scope':'synthetic arithmetic only, no game outcome/LP/model/seed draw','source_sha256':sha(Path(__file__))});print('N13rational entropy/log fixtures pass',flush=True)
if __name__=='__main__':main()
