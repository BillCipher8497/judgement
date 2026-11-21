"""N2 toy-deck public/private summary encoder; never accepts full hands."""
import numpy as np
from export import decode
SCHEMA={'seat','hand','history','n','S','R','trump'}
F=35

def public_replay(g,hist):
    assert len(hist)>=4
    played=[[] for _ in range(4)];voids=[set() for _ in range(4)];won=[0]*4;trick=[];leader=0
    for card in hist[4:]:
        p=(leader+len(trick))%4
        if trick and card//g.R!=trick[0][1]//g.R:voids[p].add(trick[0][1]//g.R)
        played[p].append(card);trick.append((p,card))
        if len(trick)==4:leader=g.trick_winner(trick);won[leader]+=1;trick=[]
    return dict(played=played,voids=voids,won=won,trick=trick,leader=leader,to_act=(leader+len(trick))%4)

def observation(g,raw):
    seat,hand,hist=decode(raw)
    assert len(hand)==g.n and set(hand)<=set(g.deck)
    return {'seat':seat,'hand':list(hand),'history':list(hist),'n':g.n,'S':g.S,'R':g.R,'trump':g.trump}

def encode(g,o):
    assert set(o)==SCHEMA and (o['n'],o['S'],o['R'],o['trump'])==(g.n,g.S,g.R,g.trump)
    seat=o['seat'];hand=tuple(o['hand']);hist=tuple(o['history']);assert len(set(hand))==g.n and len(hist)<4+4*g.n
    public=public_replay(g,hist);assert public['to_act']==seat
    assert set(public['played'][seat])<=set(hand)
    assert all(not set(public['played'][p])&set(hand) for p in range(4) if p!=seat)
    assert set(hist[4:])<=set(g.deck)
    assert len(set(hist[4:]))==len(hist[4:])
    seen=set(hand)|set(hist[4:]);cards=tuple(c for c in g.deck if c not in seen);opps=tuple((seat+k)%4 for k in (1,2,3))
    caps=np.array([g.n-len(public['played'][p]) for p in opps],int);assert sum(caps)==len(cards) # all dealt toys
    x=np.zeros((len(cards),4,F),np.float32);mask=np.zeros((len(cards),4),bool);bids=hist[:4]
    # Reconstruct suit leads from public trick positions, never hidden cards.
    leads=[[] for _ in range(4)];leader=0;trick=[]
    for c in hist[4:]:
        p=(leader+len(trick))%4
        if not trick:leads[p].append(c//g.R)
        trick.append((p,c))
        if len(trick)==4:leader=g.trick_winner(trick);trick=[]
    for i,c in enumerate(cards):
        suit,rank=divmod(c,g.R);unseen_suit=sum(cc//g.R==suit for cc in cards)
        for j,p in enumerate(opps):
            mask[i,j]=caps[j]>0 and suit not in public['voids'][p]
            x[i,j,rank]=1.;x[i,j,13+j]=1.;b=bids[p];w=public['won'][p];need=b-w;played=[cc%g.R for cc in public['played'][p] if cc//g.R==suit]
            x[i,j,17:]=[b/g.n,w/g.n,need/max(1,caps[j]),need>0,need==0,need<0,
                caps[j]/max(1,len(cards)),g.trump!=-1 and suit==g.trump,g.trump==-1,
                len(played)/g.n,max(played)/(g.R-1) if played else -1,leads[p].count(suit)/g.n,
                unseen_suit/g.R,rank/(g.R-1)*(need>0),0,0,g.n/3,p/3]
    if len(cards):assert mask.any(1).all()
    cap4=np.r_[caps,0];prop=mask*cap4
    if len(cards):prop=prop/prop.sum(1,keepdims=True)
    public.update(cards=cards,opponents=opps,capacities=caps,observation=o)
    return x,mask,prop.astype(float),public
