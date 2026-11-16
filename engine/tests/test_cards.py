import pytest

from judgement.cards import (SUIT_MASK, card_str, cards_of, hand_str, make_card, mask_of, nth_card,
                             parse_card, parse_hand)


def test_roundtrip_all_cards():
    for c in range(52):
        assert parse_card(card_str(c)) == c


def test_encoding():
    assert parse_card("2S") == 0
    assert parse_card("AS") == 12
    assert parse_card("AH") == 25
    assert parse_card("TD") == make_card(2, 8)
    assert parse_card("10d") == parse_card("TD")
    assert parse_card("AC") == 51


@pytest.mark.parametrize("bad", ["", "1S", "AX", "ASS", "S"])
def test_parse_errors(bad):
    with pytest.raises(ValueError):
        parse_card(bad)


def test_masks():
    h = parse_hand("AS KS 2H TD")
    assert h.bit_count() == 4
    assert cards_of(h) == sorted(cards_of(h))
    assert mask_of(cards_of(h)) == h
    assert hand_str(h) == "S:AK H:2 D:T C:-"
    assert (h & SUIT_MASK[0]).bit_count() == 2
    assert sum(m.bit_count() for m in SUIT_MASK) == 52
    assert [nth_card(h, i) for i in range(4)] == cards_of(h)
