import pytest

from crossval import naive_ref
from judgement.cards import NT, suit_str
from judgement.rules import Rules


@pytest.mark.parametrize("N", range(3, 8))
def test_ladder_and_schedule(N):
    r = Rules(n_players=N)
    M = 52 // N
    lad = r.ladder()
    assert len(lad) == 2 * M - 1 == r.n_rounds
    assert lad[0] == M and lad[-1] == M and min(lad) == 1 and lad[M - 1] == 1
    assert list(lad[:M]) == list(range(M, 0, -1))
    assert list(lad[M - 1:]) == list(range(1, M + 1))
    ref = naive_ref.schedule(N)
    for i, (n, trump, dealer) in enumerate(ref):
        assert lad[i] == n
        assert suit_str(r.trump_for_round(i)) == (trump or "NT")
        assert r.dealer_for_round(i) == dealer


def test_four_player_25_rounds():
    r = Rules()
    assert r.n_rounds == 25
    assert [r.trump_for_round(i) for i in range(6)] == [0, 1, 2, 3, NT, 0]


@pytest.mark.parametrize("mode,cases", [
    ("zero", [(3, 3, 13), (0, 0, 10), (3, 2, 0), (0, 4, 0)]),
    ("neg_bid", [(3, 3, 13), (3, 2, -3), (0, 1, 0)]),
    ("neg_10", [(2, 2, 12), (2, 5, -10), (0, 1, -10)]),
    ("neg_diff", [(2, 2, 12), (2, 5, -30), (4, 1, -30), (0, 1, -10)]),
])
def test_scoring_modes(mode, cases):
    r = Rules(miss_penalty=mode)
    for b, t, s in cases:
        assert r.score(b, t) == s
        assert naive_ref.score(b, t, mode) == s


def test_callable_penalty_and_hit_base():
    r = Rules(miss_penalty=lambda b, t: -(b - t) ** 2, hit_base=5)
    assert r.score(2, 2) == 7
    assert r.score(1, 4) == -9


def test_validation():
    with pytest.raises(ValueError):
        Rules(n_players=2)
    with pytest.raises(ValueError):
        Rules(n_players=8)
    with pytest.raises(ValueError):
        Rules(miss_penalty="bogus")
    with pytest.raises(ValueError):
        Rules(n_players=4, max_cards=14)


def test_frozen_and_dict_roundtrip():
    r = Rules(n_players=5, miss_penalty="neg_10", hook=False)
    with pytest.raises(Exception):
        r.n_players = 4
    assert Rules.from_dict(r.to_dict()) == r
