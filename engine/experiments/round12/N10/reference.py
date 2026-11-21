"""Independent public transitions/prefix replay; no RoundState/InfoState methods."""
from n10lib import OBS,RULES,MODEL,np,DOMAIN,BASE,STAGES,ROUNDS,partition
from judgement.bidding.model import ModelBidder
from judgement.policies.target_aware import TargetAwarePolicy
def legal(hand,trick):
    follow=hand&sum(1<<c for c in range(13*(trick[0][1]//13),13*(trick[0][1]//13+1))) if trick else 0
    return follow or hand
def winner(trick,trump):
    led=trick[0][1]//13
    return max(trick,key=lambda pc:(2 if pc[1]//13==trump and trump!=-1 else int(pc[1]//13==led),pc[1]%13))[0]
def snapshot(state,p,phase=1):
    # Phase constants imported by name below; no hidden object/reference supplied.
    from judgement.state import PLAY
    return OBS.PublicSnapshot(seat=p,n_players=4,dealer=state['dealer'],n_cards=state['n'],trump=state['trump'],hand=state['left'][p],bids=tuple(state['bids']),tricks_won=tuple(state['won']),current_trick=tuple(state['trick']),history=tuple(state['history']),trick_winners=tuple(state['winners']),voids=tuple(state['voids']),phase=PLAY if phase==1 else phase,cards_left_counts=tuple(h.bit_count() for h in state['left']),rules=RULES)
def apply(state,p,c):
    assert state['actor']==p and legal(state['left'][p],state['trick'])>>c&1
    if state['trick'] and c//13!=state['trick'][0][1]//13:state['voids'][p]|=1<<(state['trick'][0][1]//13)
    state['left'][p]^=1<<c;state['trick'].append((p,c));state['history'].append((p,c))
    if len(state['trick'])==4:
        q=winner(state['trick'],state['trump']);state['won'][q]+=1;state['winners'].append(q);state['trick']=[];state['actor']=q
    else:state['actor']=(p+1)%4
def policy(p,hero):return OBS.baseline() if p==hero else TargetAwarePolicy()
def rollout(o,hidden,rootcard,hero,address):
    state={'n':o['n_cards'],'dealer':o['dealer'],'trump':o['trump'],'left':[o['hand'] if p==hero else hidden[p] for p in range(4)],'bids':list(map(tuple,o['bids'])),'won':list(o['tricks_won']),'trick':list(map(tuple,o['current_trick'])),'history':list(map(tuple,o['history'])),'winners':list(o['trick_winners']),'voids':list(o['voids']),'actor':hero}
    rngs=[np.random.default_rng(s) for s in np.random.SeedSequence(address).spawn(4)];players=[policy(p,hero) for p in range(4)];apply(state,hero,rootcard)
    while any(state['left']):
        p=state['actor'];info=snapshot(state,p);assert not hasattr(info,'_st');c=players[p].play(info,legal(state['left'][p],state['trick']),rngs[p]);apply(state,p,c)
    bid=dict(state['bids'])[hero];return float(10+bid if bid==state['won'][hero] else 0)
def prefix(name,g,attempt):
    from judgement.state import BID
    stage=STAGES[name];base=2220000000 if name=='fixtures' else BASE;hero=g%4;ri=ROUNDS[(g//4)%24];n=RULES.ladder()[ri];root=np.random.SeedSequence([base,DOMAIN,stage,g,attempt,3]);children=root.spawn(5);dr=np.random.default_rng(children[0]);rngs=[np.random.default_rng(s) for s in children[1:]];deck=dr.permutation(52).tolist();hands=[sum(1<<c for c in deck[p*n:(p+1)*n]) for p in range(4)]
    want=None if name=='fixtures' else 0 if name in ('pilot','train') else 1 if name=='validation' else 2
    if want is not None and partition(hands[hero])!=want:return None
    state={'n':n,'dealer':RULES.dealer_for_round(ri),'trump':RULES.trump_for_round(ri),'left':hands.copy(),'bids':[],'won':[0]*4,'trick':[],'history':[],'winners':[],'voids':[0]*4,'actor':(RULES.dealer_for_round(ri)+1)%4}
    bidder=ModelBidder(MODEL)
    for k in range(4):
        p=(state['dealer']+1+k)%4;choices=list(range(n+1))
        if k==3:
            forbidden=n-sum(b for _,b in state['bids'])
            if forbidden in choices:choices.remove(forbidden)
        b=bidder.bid(snapshot(state,p,phase=BID),choices,rngs[p]);assert b in choices;state['bids'].append((p,b))
    players=[policy(p,hero) for p in range(4)]
    while any(state['left']):
        p=state['actor'];info=snapshot(state,p);choices=legal(state['left'][p],state['trick'])
        if p==hero and OBS.active(info,choices):
            card=OBS.baseline().play(info,choices,rngs[p]);x,b,cards=OBS.encode(info,choices,card)
            return {'observation':OBS.observation(info),'original_hand':hands[hero],'x':x,'base':b/8,'cards':cards,'hit':10+dict(info.bids)[hero],'round_index':ri,'true_worlds':state['left'].copy()}
        apply(state,p,players[p].play(info,choices,rngs[p]))
    return None
