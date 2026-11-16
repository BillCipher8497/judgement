from judgement import Rules,play_game
from judgement.cards import parse_hand,NT
from judgement.policies.human_guide import GuideAgent,HumanBidder,HumanPlayPolicy
from judgement.policies.human_guide_v2 import HumanBidderV2,shape_features

def test_zero_weights_preserve_the_full_guide_game():
    baseline=play_game(Rules(),[GuideAgent(HumanBidder(),HumanPlayPolicy()) for _ in range(4)],seed=14891,record_trace=False)
    candidate=play_game(Rules(),[GuideAgent(HumanBidderV2(),HumanPlayPolicy()) for _ in range(4)],seed=14891,record_trace=False)
    assert list(baseline.totals)==list(candidate.totals)
    assert [r.bids for r in baseline.rounds]==[r.bids for r in candidate.rounds]

def test_entry_and_opening_lead_are_distinct():
    hand=parse_hand('AS KS QS JS 2S AH 2D 3C')
    assert shape_features(hand,NT,8,0)==(0,0,2,2)
    assert shape_features(hand,NT,8,1)==(0,0,2,0)
    no_entry=parse_hand('AS KS QS JS 2S 2H 2D 3C')
    assert shape_features(no_entry,NT,8,0)==(0,0,0,2)

def test_shape_requires_spare_trumps_and_a_non_tiny_round():
    assert shape_features(parse_hand('AS KS 2S AH 2H'),0,5,1)==(2,0,0,0)
    assert shape_features(parse_hand('AS 2S AH'),0,3,1)==(0,0,0,0)
