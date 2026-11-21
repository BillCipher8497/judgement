"""Independent N2 structural worlds and deterministic prefix replay.
Uses no ToyGame replay/legal/winner method, root cache or DP normalizer.
Existing private bid/play functions are definitions of the fixed profile.
"""
import math
from itertools import combinations

def independent_deals(g):
    deck=tuple(range(g.S*g.R))
    for a in combinations(deck,g.n):
        first=tuple(c for c in deck if c not in a)
        for b in combinations(first,g.n):
            second=tuple(c for c in first if c not in b)
            for c in combinations(second,g.n):
                yield (a,b,c,tuple(k for k in second if k not in c))

def winner(g,trick):
    led=trick[0][1]//g.R
    def strength(pc):
        suit,rank=divmod(pc[1],g.R)
        return (2 if g.trump!=-1 and suit==g.trump else int(suit==led),rank)
    return max(trick,key=strength)[0]

def legal(g,hand,trick):
    follow=[c for c in hand if trick and c//g.R==trick[0][1]//g.R]
    return follow or list(hand)

def public(g,hist):
    leader=0;trick=[];played=[[] for _ in range(4)];voids=[set() for _ in range(4)]
    for card in hist[4:]:
        actor=(leader+len(trick))%4
        if trick and card//g.R!=trick[0][1]//g.R:voids[actor].add(trick[0][1]//g.R)
        played[actor].append(card);trick.append((actor,card))
        if len(trick)==4:leader=winner(g,trick);trick=[]
    return played,voids,(leader+len(trick))%4

def structural_worlds(g,seat,ownhand,hist):
    played,voids,actor=public(g,hist);assert actor==seat and set(played[seat])<=set(ownhand)
    unknown=tuple(c for c in range(g.S*g.R) if c not in ownhand and c not in hist[4:]);opps=[(seat+k)%4 for k in (1,2,3)];caps=[g.n-len(played[p]) for p in opps]
    def rec(j,left,assigned):
        if j==3:
            assert not left
            hands=[None]*4;hands[seat]=tuple(ownhand)
            for p,h in zip(opps,assigned):hands[p]=tuple(sorted(played[p]+list(h)))
            yield tuple(hands);return
        p=opps[j]
        for h in combinations(left,caps[j]):
            if any(c//g.R in voids[p] for c in h):continue
            yield from rec(j+1,tuple(c for c in left if c not in h),assigned+[h])
    yield from rec(0,unknown,[])

def follows_observed_history(g,hands,hist):
    flat=[c for h in hands for c in h]
    if any(len(h)!=g.n for h in hands) or len(set(flat))!=4*g.n or set(flat)!=set(range(g.S*g.R)):return False
    bids=hist[:4]
    for p,b in enumerate(bids):
        if not 0<=b<=g.n:return False
        if p==3 and g.hook and sum(bids)==g.n:return False
    left=[list(h) for h in hands];leader=0;trick=[]
    for c in hist[4:]:
        p=(leader+len(trick))%4
        if c not in legal(g,left[p],trick):return False
        left[p].remove(c);trick.append((p,c))
        if len(trick)==4:leader=winner(g,trick);trick=[]
    return True

def reaches(g,hands,seat,hist,bidder,player):
    bids=[]
    for p in range(4):
        b=bidder(g,p,hands[p],tuple(bids));legal_b=list(range(g.n+1))
        if p==3 and g.hook:
            forbidden=g.n-sum(bids)
            if forbidden in legal_b:legal_b.remove(forbidden)
        assert b in legal_b
        bids.append(b)
    if tuple(bids)!=tuple(hist[:4]):return False
    left=[list(h) for h in hands];prefix=tuple(bids);leader=0;trick=[]
    while len(prefix)<len(hist):
        p=(leader+len(trick))%4;choices=legal(g,left[p],trick);c=choices[0] if len(choices)==1 else player(g,p,hands[p],prefix)
        assert c in choices
        if c!=hist[len(prefix)]:return False
        left[p].remove(c);prefix+=(c,);trick.append((p,c))
        if len(trick)==4:leader=winner(g,trick);trick=[]
    return prefix==tuple(hist) and (leader+len(trick))%4==seat

def combination_rank(hand,deck):
    positions=[deck.index(c) for c in hand];n=len(hand);rank=0;previous=-1
    for j,where in enumerate(positions):
        for skipped in range(previous+1,where):rank+=math.comb(len(deck)-skipped-1,n-j-1)
        previous=where
    return rank

def deal_index(g,hands):
    rest=list(range(g.S*g.R));ranks=[];radices=[]
    for p in range(3):
        h=tuple(sorted(hands[p]));ranks.append(combination_rank(h,rest));radices.append(math.comb(len(rest),g.n));rest=[c for c in rest if c not in h]
    assert tuple(rest)==tuple(hands[3])
    return (ranks[0]*radices[1]+ranks[1])*radices[2]+ranks[2]

def direct_posterior(g,seat,hand,hist,bidder,player):
    worlds=list(structural_worlds(g,seat,hand,hist));assert worlds and all(follows_observed_history(g,w,hist) for w in worlds)
    reached=[w for w in worlds if reaches(g,w,seat,hist,bidder,player)]
    assert reached
    return worlds,reached
