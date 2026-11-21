"""Manual small-endgame MaxN, independent of the production solver/state."""
from functools import lru_cache
def bits(mask):return [c for c in range(52) if mask>>c&1]
def winner(trick,trump):
    p,c=trick[0]
    for q,d in trick[1:]:
        if (d//13==c//13 and d>c) or (d//13!=c//13 and d//13==trump):p,c=q,d
    return p
def after(hands,won,trick,actor,card,trump):
    h=list(hands);h[actor]^=1<<card;t=trick+((actor,card),);w=list(won)
    if len(t)==4:
        p=winner(t,trump);w[p]+=1;return tuple(h),tuple(w),(),p
    return tuple(h),tuple(w),t,(actor+1)%4
def legal(hand,trick):
    cards=bits(hand)
    if trick:
        follow=[c for c in cards if c//13==trick[0][1]//13]
        if follow:cards=follow
    return cards
def move_scores(observation,hands):
    bids=dict(observation['bids']);trump=observation['trump'];root=observation['seat'];won=tuple(observation['tricks_won']);trick=tuple(map(tuple,observation['current_trick']))
    @lru_cache(None)
    def value(h,w,t,p):
        if not any(h):
            assert not t
            return tuple(10+bids[q] if w[q]==bids[q] else 0 for q in range(4))
        best=None
        for c in legal(h[p],t):
            v=value(*after(h,w,t,p,c,trump))
            if best is None or v[p]>best[p]:best=v
        assert best is not None
        return best
    return {c:value(*after(tuple(hands),won,trick,root,c,trump))[root] for c in legal(hands[root],trick)}
def prefix(observation,current_hands):
    # Reconstruct each sample's original hands and replay public chronology.
    h=list(map(int,current_hands))
    for p,c in observation['history']:h[p]|=1<<c
    assert len(set(c for mask in h for c in bits(mask)))==sum(mask.bit_count() for mask in h)
    n=observation['n_cards'];assert all(mask.bit_count()==n for mask in h)
    w=(0,0,0,0);t=();p=(observation['dealer']+1)%4;trump=observation['trump']
    for actor,c in observation['history']:
        assert actor==p and c in legal(h[p],t)
        h,w,t,p=after(tuple(h),w,t,p,c,trump)
    assert p==observation['seat'] and tuple(w)==tuple(observation['tricks_won']) and t==tuple(map(tuple,observation['current_trick']))
    assert tuple(h)==tuple(map(int,current_hands))
    return True
def generated_context(address,round_index,expected):
    # Fresh deal/spawn/turn/follow/void replay without engine.make_rngs/deal,
    # RoundState or production context(). Fixed policies reused as definitions.
    import numpy as np
    from observables import RULES,MODEL,restore,baseline
    from judgement.bidding.model import ModelBidder
    from judgement.policies.target_aware import TargetAwarePolicy
    n=RULES.ladder()[round_index];dealer=round_index%4;trump=RULES.trump_for_round(round_index);children=np.random.SeedSequence(address).spawn(5);perm=np.random.default_rng(children[0]).permutation(52);rngs=[np.random.default_rng(s) for s in children[1:]];h=[sum(1<<int(c) for c in perm[p*n:(p+1)*n]) for p in range(4)];initial=h.copy();won=(0,0,0,0);trick=();hist=[];winners=[];voids=[0]*4;bids=[];actor=(dealer+1)%4
    def observation_now(p):return {'seat':p,'dealer':dealer,'n_cards':n,'trump':trump,'hand':h[p],'bids':bids.copy(),'tricks_won':list(won),'current_trick':list(trick),'history':hist.copy(),'trick_winners':winners.copy(),'voids':voids.copy(),'cards_left_counts':[m.bit_count() for m in h]}
    for _ in range(4):
        choices=list(range(n+1))
        if actor==dealer:
            banned=n-sum(v for p,v in bids)
            if banned in choices:choices.remove(banned)
        b=ModelBidder(MODEL).bid(restore(observation_now(actor)),choices,rngs[actor]);assert b in choices;bids.append((actor,b));actor=(actor+1)%4
    hero=expected['seat']
    while len(hist)<len(expected['history']):
        choices=legal(h[actor],trick);mask=sum(1<<c for c in choices);policy=baseline() if actor==hero else TargetAwarePolicy();c=policy.play(restore(observation_now(actor)),mask,rngs[actor]);assert c in choices
        if actor==hero:assert not (h[hero].bit_count()==2 and len(choices)==2 and 0<=dict(bids)[hero]-won[hero]<=2),'Earlier observable active event should have stopped context generation'
        if trick and c//13!=trick[0][1]//13:voids[actor]|=1<<(trick[0][1]//13)
        hist.append((actor,c));new_h,new_w,new_t,new_actor=after(tuple(h),won,trick,actor,c,trump)
        if len(trick)==3:winners.append(new_actor)
        h=list(new_h);won=new_w;trick=new_t;actor=new_actor
    got=observation_now(actor)
    # Normalize tuple/list JSON differences without changing numerical content.
    import json
    assert json.loads(json.dumps(got))==json.loads(json.dumps(expected))
    return initial[hero]
def full_game(seed,hero,choose):
    """Independent actual ladder simulator; fixed policies supplied as definitions."""
    import numpy as np
    from observables import RULES,MODEL,restore
    from judgement.bidding.model import ModelBidder
    from judgement.policies.target_aware import TargetAwarePolicy
    children=np.random.SeedSequence(seed).spawn(5);dr=np.random.default_rng(children[0]);rngs=[np.random.default_rng(s) for s in children[1:]];totals=[0]*4;rounds=[]
    for ri,n in enumerate(RULES.ladder()):
        dealer=ri%4;trump=RULES.trump_for_round(ri);perm=dr.permutation(52);h=[sum(1<<int(c) for c in perm[p*n:(p+1)*n]) for p in range(4)];won=(0,0,0,0);trick=();hist=[];winners=[];voids=[0]*4;bids=[];actor=(dealer+1)%4
        def info(p):return restore({'seat':p,'dealer':dealer,'n_cards':n,'trump':trump,'hand':h[p],'bids':bids.copy(),'tricks_won':list(won),'current_trick':list(trick),'history':hist.copy(),'trick_winners':winners.copy(),'voids':voids.copy(),'cards_left_counts':[m.bit_count() for m in h]})
        for _ in range(4):
            choices=list(range(n+1))
            if actor==dealer:
                banned=n-sum(v for p,v in bids)
                if banned in choices:choices.remove(banned)
            b=ModelBidder(MODEL).bid(info(actor),choices,rngs[actor]);assert b in choices;bids.append((actor,b));actor=(actor+1)%4
        while any(h):
            choices=legal(h[actor],trick);mask=sum(1<<c for c in choices);c=choose(info(actor),mask,rngs[actor]) if actor==hero else TargetAwarePolicy().play(info(actor),mask,rngs[actor]);assert c in choices
            if trick and c//13!=trick[0][1]//13:voids[actor]|=1<<(trick[0][1]//13)
            hist.append((actor,c));new_h,new_w,new_t,new_actor=after(tuple(h),won,trick,actor,c,trump)
            if len(trick)==3:winners.append(new_actor)
            h=list(new_h);won=new_w;trick=new_t;actor=new_actor
        assert not trick and sum(won)==n;byseat=dict(bids);scores=[10+byseat[p] if byseat[p]==won[p] else 0 for p in range(4)]
        for p in range(4):totals[p]+=scores[p]
        rounds.append({'n':n,'trump':trump,'dealer':dealer,'bids':[byseat[p] for p in range(4)],'tricks':list(won),'scores':scores})
    return {'totals':totals,'rounds':rounds}
