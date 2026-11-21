"""Detached own/public adapter; no hidden-state or teacher feature access."""
from common import np
from judgement.cards import cards_of
from judgement.rules import Rules
from judgement.state import InfoSnapshot,PLAY
from judgement.bidding.model import BidModel,ModelBidder,Agent
from judgement.policies.ta_plus import ta_plus_v2
from judgement.policies.target_aware import TargetAwarePolicy
from common import ENGINE
MODEL=BidModel.load(ENGINE/'experiments/phase1/results/model_ta_final.json')
RULES=Rules()
class PublicSnapshot(InfoSnapshot):
    """Complete the legacy detached API using public trick cards alone."""
    def _public_winner(self):
        if not self.current_trick:return -1,-1
        p,c=self.current_trick[0]
        for q,d in self.current_trick[1:]:
            if (d//13==c//13 and d>c) or (d//13!=c//13 and d//13==self.trump):p,c=q,d
        return p,c
    @property
    def winning_card(self):return self._public_winner()[1]
    @property
    def winning_seat(self):return self._public_winner()[0]
def detach(info):return info if isinstance(info,PublicSnapshot) else restore(observation(info))
def agents():return [Agent(ModelBidder(MODEL),TargetAwarePolicy()) for _ in range(4)]
def baseline():return ta_plus_v2()
def active(info,legal):
    b=dict(info.bids)[info.seat]
    return info.hand.bit_count()==2 and legal.bit_count()==2 and 0<=b-info.tricks_won[info.seat]<=2
def encode(info,legal,base_card):
    assert active(info,legal)
    cards=cards_of(legal);assert len(cards)==2 and base_card in cards
    x=[]
    x.extend(float(info.hand>>c&1) for c in range(52))
    played=np.zeros((4,52))
    for p,c in info.history:played[p,c]=1
    x.extend(played.ravel())
    for values in (dict(info.bids),dict(enumerate(info.tricks_won))):
        for p in range(4):x.extend(float(v==values[p]) for v in range(14))
    for k in range(3):
        c=info.current_trick[k][1] if k<len(info.current_trick) else 52
        x.extend(float(j==c) for j in range(53))
    x.extend(float(info.voids[p]>>s&1) for p in range(4) for s in range(4))
    x.extend(info.cards_left(p)/13 for p in range(4))
    x.extend(float(info.seat==p) for p in range(4));x.extend(float(info.dealer==p) for p in range(4))
    x.extend(float(info.trump==p) for p in (0,1,2,3,-1));x.extend(float(info.n_cards==n) for n in range(1,14))
    for c in cards:x.extend(float(j==c) for j in range(52))
    b=np.array([2.*float(c==base_card) for c in cards]);x.extend(b/2)
    x=np.asarray(x,float);assert x.shape==(683,) and np.isfinite(x).all()
    return x,b,cards
def observation(info):
    return {'seat':int(info.seat),'dealer':int(info.dealer),'n_cards':int(info.n_cards),'trump':int(info.trump),'hand':int(info.hand),'bids':list(info.bids),'tricks_won':list(info.tricks_won),'current_trick':list(info.current_trick),'history':list(info.history),'trick_winners':list(info.trick_winners),'voids':list(info.voids),'cards_left_counts':[info.cards_left(p) for p in range(4)]}
def restore(o):
    return PublicSnapshot(seat=o['seat'],n_players=4,dealer=o['dealer'],n_cards=o['n_cards'],trump=o['trump'],hand=o['hand'],bids=tuple(map(tuple,o['bids'])),tricks_won=tuple(o['tricks_won']),current_trick=tuple(map(tuple,o['current_trick'])),history=tuple(map(tuple,o['history'])),trick_winners=tuple(o['trick_winners']),voids=tuple(o['voids']),phase=PLAY,cards_left_counts=tuple(o['cards_left_counts']),rules=RULES)
