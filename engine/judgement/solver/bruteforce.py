"""Independent exhaustive reference solvers (no memo, no card equivalence, no cut-offs).

They walk the engine's own ``RoundState.legal_cards`` / ``apply_card`` /
``undo_card`` so they also cross-check the solver against engine legality and
trick resolution.  Exponential: use only for tiny n (tests, n <= 3).
"""
from __future__ import annotations

from ..cards import cards_of
from ..state import PLAY
from .core import NEUTRAL, SPITEFUL


def _scores(st):
    return tuple(st.scores())


def bf_maxn(st, mode: str = NEUTRAL):
    if st.phase != PLAY:
        return _scores(st)
    p = st.to_act
    best, bk = None, None
    for c in cards_of(st.legal_cards()):     # ascending card index = neutral order
        st.apply_card(c)
        v = bf_maxn(st, mode)
        st.undo_card()
        if mode == NEUTRAL:
            k = v[p]
        else:
            o = sum(v) - v[p]
            k = (v[p], -o if mode == SPITEFUL else o)
        if best is None or k > bk:
            best, bk = v, k
    return best


def bf_paranoid(st, t: int):
    if st.phase != PLAY:
        return st.scores()[t]
    vals = []
    for c in cards_of(st.legal_cards()):
        st.apply_card(c)
        vals.append(bf_paranoid(st, t))
        st.undo_card()
    return max(vals) if st.to_act == t else min(vals)


def bf_social(st):
    if st.phase != PLAY:
        return sum(st.scores())
    best = None
    for c in cards_of(st.legal_cards()):
        st.apply_card(c)
        v = bf_social(st)
        st.undo_card()
        if best is None or v > best:
            best = v
    return best
