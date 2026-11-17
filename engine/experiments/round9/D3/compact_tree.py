"""Same frozen game, with packed numeric buffers and private byte keys."""
from array import array
import numpy as np
from judgement.cfr.toy import Tree,_play
from judgement.cfr.heuristics import myopic_bid
from bounded_ta import ta_card

class PrivateKeys:
    def __init__(self,packed,n):self.packed=packed;self.n=n
    def __len__(self):return len(self.packed)
    def __getitem__(self,index):
        key=self.packed[index]
        return key[0],tuple(key[1:1+self.n]),key[1+self.n:]
    def __iter__(self):
        for i in range(len(self)):yield self[i]

class CompactTree(Tree):
    def __init__(self,g,deals=None,checkpoint=None):
        self.g=g;N,A=g.N,g.A
        assert N+N*g.n<128 and len(g.deck)<256
        deals=g.deals() if deals is None else deals;pc=1/len(deals)
        lengths=array('b');players=array('b');node_is=array('i');score=array('b');roots=array('i')
        ep=array('i');ec=array('i');es=array('i');ea=array('b')
        ids={};ip=array('b');ir=array('i');valid=bytearray();packed=[]
        def node(length,player):
            lengths.append(length);players.append(player);node_is.append(-1);score.append(0)
            return len(lengths)-1
        def rec(hands,left,bids,trick,leader,tricks,hist):
            while True:
                length=len(hist);remaining=len(left[0])+int(any(p==0 for p,_ in trick))
                need=bids[0]-tricks[0]
                if need<0 or need>remaining or length==N+N*g.n:
                    i=node(length,-1)
                    if length==N+N*g.n and tricks[0]==bids[0]:score[i]=g.hit_base+bids[0]
                    return i
                p=(leader+len(trick))%N;legal=g.legal_cards(left[p],trick)
                if len(legal)==1 or p==0:
                    c=legal[0] if len(legal)==1 else ta_card(g,p,hands[p],hist)
                    left,trick,leader,tricks=_play(g,left,trick,leader,tricks,p,c)
                    hist+=bytes((c,));continue
                key=bytes((p,*hands[p]))+hist;s=ids.get(key);i=node(length,p)
                if s is None:
                    s=len(ip);ids[key]=s;ip.append(p);ir.append(i);packed.append(key)
                    legal_indices={hands[p].index(c) for c in legal}
                    valid.extend(int(a in legal_indices) for a in range(A))
                node_is[i]=s
                for c in legal:
                    l2,t2,ld2,tr2=_play(g,left,trick,leader,tricks,p,c)
                    child=rec(hands,l2,bids,t2,ld2,tr2,hist+bytes((c,)))
                    ep.append(i);ec.append(child);es.append(s);ea.append(hands[p].index(c))
                return i
        for d,hands in enumerate(deals):
            bids=[]
            for p in range(N):bids.append(myopic_bid(g,p,hands[p],tuple(bids)))
            roots.append(rec(hands,[list(h) for h in hands],bids,[],0,[0]*N,bytes(bids)))
            if (d+1)%250==0 and checkpoint:checkpoint(d+1,len(lengths),len(ip))
        self.n_deals=len(deals);self.roots=np.frombuffer(roots,np.int32)
        self.node_len=np.frombuffer(lengths,np.int8);self.node_pl=np.frombuffer(players,np.int8)
        self.node_is=np.frombuffer(node_is,np.int32);self.n_nodes=len(lengths)
        self.node_c=np.full(self.n_nodes,pc);scores=np.frombuffer(score,np.int8)
        self.U=np.empty((self.n_nodes,N));self.U[:,0]=scores;self.U[:,1:]=-scores[:,None]
        self.is_pl=np.frombuffer(ip,np.int8);self.is_rep=np.frombuffer(ir,np.int32)
        self.is_valid=np.frombuffer(valid,np.bool_).reshape(-1,A)
        self.n_is=len(ip);self.is_bid=np.zeros(self.n_is,bool);self.is_key=PrivateKeys(packed,g.n)
        parent=np.frombuffer(ep,np.int32);order=np.lexsort((parent,self.node_len[parent]))
        self.e_par=parent[order];self.e_ch=np.frombuffer(ec,np.int32)[order]
        self.e_is=np.frombuffer(es,np.int32)[order];self.e_a=np.frombuffer(ea,np.int8)[order]
        self.e_pl=self.node_pl[self.e_par];self.e_ia=self.e_is*A+self.e_a
        self.n_edges=len(self.e_par);self.chunks=[]
        if self.n_edges:
            levels=self.node_len[self.e_par];cuts=np.flatnonzero(np.diff(levels))+1
            for a,b in zip(np.r_[0,cuts],np.r_[cuts,self.n_edges]):
                par=self.e_par[a:b];seg=np.r_[0,np.flatnonzero(np.diff(par))+1].astype(np.int32)
                self.chunks.append((int(a),int(b),seg,par[seg]))
