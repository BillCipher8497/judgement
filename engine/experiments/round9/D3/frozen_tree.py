"""Exact play-only tree: myopic bids and TA hero are deterministic forced moves.

Early zero terminals occur only after the hero's hit is mathematically impossible.
Private infosets otherwise use the original (seat, own hand, public history) key.
"""
import numpy as np
from judgement.cfr.toy import Tree,_play
from judgement.cfr.heuristics import myopic_bid
from bounded_ta import ta_card

class FrozenTree(Tree):
    def __init__(self,g,deals=None,checkpoint=None):
        self.g=g;N,A=g.N,g.A
        deals=g.deals() if deals is None else deals;pc=1/len(deals)
        node_len=[];node_pl=[];node_is=[];terminal={};roots=[]
        e_par=[];e_ch=[];e_is=[];e_a=[]
        keys={};is_pl=[];is_valid=[];is_rep=[];is_key=[]
        def node(length,player,infoset=-1):
            node_len.append(length);node_pl.append(player);node_is.append(infoset)
            return len(node_len)-1
        def rec(hands,left,bids,trick,leader,tricks,hist):
            while True:
                length=len(hist)
                remaining=len(left[0])+int(any(p==0 for p,_ in trick))
                need=bids[0]-tricks[0]
                if need<0 or need>remaining or length==N+N*g.n:
                    i=node(length,-1);terminal[i]=g.hit_base+bids[0] if tricks[0]==bids[0] and length==N+N*g.n else 0
                    return i
                p=(leader+len(trick))%N;legal=g.legal_cards(left[p],trick)
                if len(legal)==1 or p==0:
                    card=legal[0] if len(legal)==1 else ta_card(g,p,hands[p],hist)
                    left,trick,leader,tricks=_play(g,left,trick,leader,tricks,p,card)
                    hist=hist+(card,);continue
                key=(p,hands[p],hist);s=keys.get(key)
                i=node(length,p)
                if s is None:
                    s=len(is_pl);keys[key]=s;is_pl.append(p);is_rep.append(i);is_key.append(key)
                    valid=[False]*A
                    for c in legal:valid[hands[p].index(c)]=True
                    is_valid.append(valid)
                node_is[i]=s
                for c in legal:
                    l2,t2,ld2,tr2=_play(g,left,trick,leader,tricks,p,c)
                    child=rec(hands,l2,bids,t2,ld2,tr2,hist+(c,))
                    e_par.append(i);e_ch.append(child);e_is.append(s);e_a.append(hands[p].index(c))
                return i
        for d,hands in enumerate(deals):
            bids=[]
            for p in range(N):bids.append(myopic_bid(g,p,hands[p],tuple(bids)))
            roots.append(rec(hands,[list(h) for h in hands],bids,[],0,[0]*N,tuple(bids)))
            if (d+1)%250==0:
                if checkpoint:checkpoint(d+1,len(node_len),len(is_pl))
        self.n_deals=len(deals);self.roots=np.array(roots,np.int64)
        self.node_len=np.array(node_len,np.int16);self.node_pl=np.array(node_pl,np.int8)
        self.node_is=np.array(node_is,np.int64);self.n_nodes=len(node_len)
        self.node_c=np.full(self.n_nodes,pc)
        self.U=np.zeros((self.n_nodes,N))
        ids=np.fromiter(terminal,int);score=np.fromiter(terminal.values(),float)
        self.U[ids,0]=score;self.U[ids,1:]=-score[:,None]
        self.is_pl=np.array(is_pl,np.int8);self.is_valid=np.array(is_valid,bool).reshape(-1,A)
        self.is_rep=np.array(is_rep,np.int64);self.is_key=is_key;self.n_is=len(is_pl)
        self.is_bid=np.zeros(self.n_is,bool)
        parent=np.array(e_par,np.int64);order=np.lexsort((parent,self.node_len[parent]))
        self.e_par=parent[order];self.e_ch=np.array(e_ch,np.int64)[order]
        self.e_is=np.array(e_is,np.int64)[order];self.e_a=np.array(e_a,np.int64)[order]
        self.e_pl=self.node_pl[self.e_par].astype(np.int64);self.e_ia=self.e_is*A+self.e_a
        self.n_edges=len(self.e_par);self.chunks=[]
        if self.n_edges:
            lengths=self.node_len[self.e_par];cuts=np.flatnonzero(np.diff(lengths))+1
            for a,b in zip(np.r_[0,cuts],np.r_[cuts,self.n_edges]):
                par=self.e_par[a:b];seg=np.r_[0,np.flatnonzero(np.diff(par))+1]
                self.chunks.append((int(a),int(b),seg,par[seg]))
