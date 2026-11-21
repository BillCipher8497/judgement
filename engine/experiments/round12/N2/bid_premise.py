"""M16 matched-bid premise checks, including an independent hook-legal argmax."""
from judgement.cfr.heuristics import myopic_bid,trick_dist_greedy

def manual_hook_argmax(g,seat,hand,prior):
    assert len(prior)==seat
    legal=list(range(g.n+1))
    if seat==3 and g.hook:
        forbidden=g.n-sum(prior)
        if forbidden in legal:legal.remove(forbidden)
    distribution=trick_dist_greedy(g,seat,tuple(hand))
    return max(legal,key=lambda b:((g.hit_base+b)*distribution[b],-b))

def assert_matched(g,hands,bids):
    assert len(bids)==4
    for seat in range(4):
        prior=tuple(bids[:seat]);prediction=myopic_bid(g,seat,tuple(hands[seat]),prior)
        assert prediction==manual_hook_argmax(g,seat,tuple(hands[seat]),prior)==bids[seat],('M16matched hook-legal bid premise failed',seat,bids)
    return True
