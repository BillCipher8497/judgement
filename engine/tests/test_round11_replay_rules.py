"""Small exhaustive checks of the fresh forward replay's independent card rules."""
import importlib.util
import sys
from itertools import combinations, permutations
from pathlib import Path
from judgement.cfr.toy import ToyGame

PATH = Path(__file__).resolve().parents[1]/'experiments/round11/E5/replay.py'
SPEC = importlib.util.spec_from_file_location('round11_forward_replay', PATH)
replay = importlib.util.module_from_spec(SPEC)
old_path = sys.path[:]
try: SPEC.loader.exec_module(replay)
finally: sys.path[:] = old_path


def test_fresh_winner_exhaustive_cards():
    for trump in [0, -1]:
        g = ToyGame(N=4, S=3, R=4, n=3, trump=trump)
        # All11880ordered distinct-card tricks, including void/discard cases.
        for cards in permutations(range(12), 4):
            trick = list(enumerate(cards))
            assert replay.trick_winner(trick, trump) == g.trick_winner(trick)


def test_fresh_follow_suit_and_complete_deal_count():
    g = ToyGame(N=4, S=3, R=4, n=3)
    for hand in combinations(range(12), 3):
        for led in range(12):
            if led in hand: continue
            trick = [(0, led)]
            assert replay.legal_cards(list(hand), trick) == g.legal_cards(list(hand), trick)
    count = 0
    for hands in replay.independent_deals():
        assert len(set(c for h in hands for c in h)) == 12
        assert all(len(h) == 3 and tuple(sorted(h)) == h for h in hands)
        count += 1
    assert count == 369600
