"""Experimental shape/entry additions to the frozen human bidder.

Weights are half-trick points. Zero weights reproduce HumanBidder. The study
passes fitted weights explicitly; the existing guide and its defaults stay
unchanged until validation supports a change.
"""
from __future__ import annotations
from .human_guide import HumanBidder,bid_counts,_round_half_down
from ..cards import NT


def shape_features(hand:int,trump:int,n:int,pos:int)->tuple[int,int,int,int]:
    c=bid_counts(hand,trump,n)
    if trump!=NT:
        if n<5 or c['tlen']<2:return (0,0,0,0)
        capacity=c['tlen']-1
        return (min(2,capacity,c['voids']),min(2,capacity,c['singles']),0,0)
    masks=[(hand>>(13*s))&0x1fff for s in range(4)]
    suit=max(range(4),key=lambda s:(masks[s].bit_count(),bool(masks[s]&(1<<12)),bool(masks[s]&(1<<11)),-s))
    long=masks[suit];length=long.bit_count()
    units=min(2,max(0,length-3))
    ace=bool(long&(1<<12));control=bool(long&((1<<12)|(1<<11)))
    outside_ace=any(masks[s]&(1<<12) for s in range(4) if s!=suit)
    return (0,0,units*int(control and outside_ace),units*int(ace and pos==0))


class HumanBidderV2(HumanBidder):
    name='guide_bid_v2_experimental'

    def __init__(self,weights=(0,0,0,0),**kw):
        super().__init__(**kw)
        if len(weights)!=4 or any(int(w)!=w for w in weights):
            raise ValueError('four integer half-point weights required')
        self.weights=tuple(map(int,weights))

    def estimate(self,info):
        n=info.n_cards;N=info.n_players
        pos=(info.seat-info.dealer-1)%N
        extra=sum(w*x for w,x in zip(self.weights,shape_features(info.hand,info.trump,n,pos)))
        points=max(0,self.half_points(info.hand,info.trump,n)+extra)
        b=points//2+(points%2 if self.w['round_up'] else 0)
        prior=sum(x for _,x in info.bids)
        b-=_round_half_down((prior-pos*n/N)/self.w['prior_k'])
        return max(0,min(n,b)),points%2
