import sys
from pathlib import Path
import numpy as np
from judgement import Rules,RoundState
from judgement.engine import deal,run_round
from judgement.policies.spoiler import PIMCSpoilerPolicy,OmniscientSpoilerPolicy
from judgement.policies.ta_plus import ta_plus_v2
_old_path=sys.path[:]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/round9/D1'))
from intervention import InformationSpoiler,world_sampler
sys.path[:]=_old_path

def test_conditioning_preserves_revealed_cards_and_ignores_other_private_hands():
    rules=Rules();rng=np.random.default_rng(8)
    st=RoundState(rules,3,-1,3,deal(rules,3,rng))
    for b in [1,1,1,1]:
        st.apply_bid(b if b in st.legal_bids() else 0)
    info=st.info_state(1);other=st.copy();other.hands[2],other.hands[3]=other.hands[3],other.hands[2]
    for knowledge in ['own','hero']:
        a=world_sampler(info,0,knowledge)(np.random.default_rng(44))
        b=world_sampler(other.info_state(1),0,knowledge)(np.random.default_rng(44))
        assert a==b
        assert all(h.bit_count()==3 for h in a.values())
        assert len({c for h in [info.hand,*a.values()] for c in range(52) if h>>c&1})==12
        if knowledge=='hero':assert a[0]==st.hands[0]

def test_own_and_all_actions_match_existing_references():
    for seed in range(4):
        rules=Rules();st=RoundState(rules,2,seed%2-1,3,deal(rules,2,np.random.default_rng(seed)))
        for b in [1,0,0,0]:st.apply_bid(b if b in st.legal_bids() else 1)
        for knowledge,reference in [('own',lambda:PIMCSpoilerPolicy(0,4)),('all',lambda:OmniscientSpoilerPolicy(0))]:
            a,b=st.copy(),st.copy()
            pa=[ta_plus_v2()]+[InformationSpoiler(0,knowledge,4) for _ in range(3)]
            pb=[ta_plus_v2()]+[reference() for _ in range(3)]
            run_round(a,pa,[np.random.default_rng(seed+p) for p in range(4)])
            run_round(b,pb,[np.random.default_rng(seed+p) for p in range(4)])
            assert a.history==b.history

def test_hero_conditioning_preserves_midtrick_voids_and_played_cards():
    hands=[sum(1<<c for c in cards) for cards in [[10,11,12],[13,14,15],[26,27,28],[39,40,41]]]
    st=RoundState(Rules(),3,-1,3,hands)
    for b in [1,0,0,0]:st.apply_bid(b)
    st.apply_card(12);st.apply_card(13)
    info=st.info_state(2);sample=world_sampler(info,0,'hero');rng=np.random.default_rng(99)
    for _ in range(50):
        hidden=sample(rng)
        assert hidden[0]==st.hands[0] and not hidden[1]&((1<<13)-1)
        masks=[info.hand,*hidden.values()]
        assert all(not h&info.played_mask for h in masks)
        union=0
        for h in masks:assert not union&h;union|=h
        assert union.bit_count()==10
        assert all(hidden[p].bit_count()==info.cards_left(p) for p in hidden)
