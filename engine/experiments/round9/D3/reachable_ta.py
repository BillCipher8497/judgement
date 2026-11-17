"""Exact TA completion on baseline and all unilateral-deviation supports."""
import numpy as np
from bounded_ta import ta_card

def reachable_ta_profile(t,progress=None):
    sigma=np.zeros(t.is_valid.shape)
    sigma[np.arange(t.n_is),t.is_valid.argmax(1)]=1
    known=np.zeros(t.n_is,bool);computed=0
    for deviator in [-1,*range(1,t.g.N)]:
        seen=np.zeros(t.n_nodes,bool);seen[t.roots]=True
        for a,b,seg,parents in t.chunks:
            need=seen[parents]&(t.node_pl[parents]!=deviator)
            infos=np.unique(t.node_is[parents[need]])
            for i in infos[~known[infos]]:
                p,hand,hist=t.is_key[i];card=ta_card(t.g,p,hand,hist)
                sigma[i]=0;sigma[i,hand.index(card)]=1;known[i]=True;computed+=1
                if progress and computed%10000==0:progress(computed,t.n_is)
            choose=seen[t.e_par[a:b]]&((t.e_pl[a:b]==deviator)|(sigma.ravel()[t.e_ia[a:b]]>0))
            seen[t.e_ch[a:b][choose]]=True
    return sigma,{'TA_infosets_evaluated':computed,'total_infosets':t.n_is,
                  'scope':'Exact baseline plus every one-player deviation support; off-support defaults are irrelevant to these values.'}
