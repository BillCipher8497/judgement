"""Interpretable decision features and abstract actions (F6 theorem extraction).

Abstract actions use only information the acting player can see (own hand, the
cards played, the card currently winning the trick), so a rule written in terms
of them is playable under imperfect information.
  LOW        lowest-ranked legal card (non-trump preferred on ties)
  HIGH       highest-ranked legal card (non-trump preferred on ties)
  CHEAPWIN   following: cheapest card beating the current winner (same-suit
             overcard before a ruff, then lowest rank)
  HIGHWIN    following: strongest card beating the current winner (trump first)
  HIGHLOSE   following: highest card that does not beat the current winner
  LOWLOSE    following: lowest card that does not beat the current winner
  LOWBOSS    leading: lowest own card that is the highest outstanding of its suit
  HIGHBOSS   leading: highest such boss card (trump first)
"""
from __future__ import annotations

import json

from ..cards import NT, SUIT_OF, cards_of
from .heuristic import _key_high, _key_low, beats, boss_cards

RANK = [c % 13 for c in range(52)]
ACTIONS = ("LOW", "HIGH", "CHEAPWIN", "HIGHWIN", "HIGHLOSE", "LOWLOSE", "LOWBOSS", "HIGHBOSS")


def abstract_actions(hand, legal, played, wc, trump):
    """{action: card} for the actions defined at this decision."""
    L = cards_of(legal)
    out = {"LOW": min(L, key=lambda c: _key_low(c, trump)),
           "HIGH": max(L, key=lambda c: _key_high(c, trump))}
    if wc >= 0:
        win = [c for c in L if beats(c, wc, trump)]
        lose = [c for c in L if not beats(c, wc, trump)]
        if win:
            out["CHEAPWIN"] = min(win, key=lambda c: (SUIT_OF[c] == trump, RANK[c]))
            out["HIGHWIN"] = max(win, key=lambda c: (SUIT_OF[c] == trump, RANK[c]))
        if lose:
            out["HIGHLOSE"] = max(lose, key=lambda c: _key_high(c, trump))
            out["LOWLOSE"] = min(lose, key=lambda c: _key_low(c, trump))
    else:
        boss = [c for c in boss_cards(hand, played) if legal >> c & 1]
        if boss:
            out["LOWBOSS"] = min(boss, key=lambda c: _key_low(c, trump))
            out["HIGHBOSS"] = max(boss, key=lambda c: (SUIT_OF[c] == trump, RANK[c]))
    return out


def status(need, rem):
    if need < 0 or need > rem:
        return "dead"
    return "zero" if need == 0 else "pos"


def decision_record(s, mv, heur_card=None):
    """Features of the decision at RoundState ``s`` with exact move values ``mv``
    ({card: score vector}).  None if the move is forced (one legal card)."""
    legal = s.legal_cards()
    if legal.bit_count() == 1:
        return None
    p = s.to_act
    N = s.N
    rem = s.n_cards - len(s.trick_winners)
    needs = [s.bids[q] - s.tricks_won[q] for q in range(N)]
    st = [status(x, rem) for x in needs]
    pos = len(s.trick)
    played = 0
    for _, c in s.history:
        played |= 1 << c
    hand = s.hands[p]
    wc = s.best_card
    trump = s.trump
    own = {c: v[p] for c, v in mv.items()}
    vmax = max(own.values())
    vmin = min(own.values())
    acts = abstract_actions(hand, legal, played, wc, trump)
    alive_need = sum(x for q, x in enumerate(needs) if st[q] != "dead")
    rec = {
        "seat": p, "pos": pos, "lead": int(pos == 0), "last": int(pos == N - 1),
        "nt": int(trump == NT), "rem": rem, "need": needs[p], "status": st[p],
        "need_eq_rem": int(needs[p] == rem),
        "bid": s.bids[p], "sum_bids_minus_n": sum(s.bids) - s.n_cards,
        "others_need": sum(max(x, 0) for q, x in enumerate(needs) if q != p and st[q] != "dead"),
        "others_zero": sum(1 for q in range(N) if q != p and st[q] == "zero"),
        "others_dead": sum(1 for q in range(N) if q != p and st[q] == "dead"),
        "slack": rem - alive_need,
        "winner_status": st[s.best_seat] if pos else "",
        "n_legal": legal.bit_count(),
        "has_trump": int(trump != NT and any(SUIT_OF[c] == trump for c in cards_of(hand))),
        "can_win": int("CHEAPWIN" in acts), "can_lose": int("HIGHLOSE" in acts),
        "has_boss": int("LOWBOSS" in acts),
        "vmax": vmax, "vmin": vmin, "indiff": int(vmax == vmin),
        "n_opt": sum(1 for v in own.values() if v == vmax),
        "hand": hand, "legal": legal, "played": played, "wc": wc, "trump_code": trump,
        "mv": json.dumps({str(c): v for c, v in sorted(own.items())}),
    }
    for a in ACTIONS:
        rec[f"v_{a}"] = own[acts[a]] if a in acts else ""
    rec["v_HEUR"] = own[heur_card] if heur_card is not None else ""
    return rec
