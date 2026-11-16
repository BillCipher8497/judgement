"""Target-aware (TA) play and its competitive variant TA-spite.

Both policies track the unseen cards (deck minus own hand minus played cards)
and the opponents' known voids, and score every legal card by an estimate of
P(this card wins the current trick):

* exact zero  -> the card cannot beat the card currently winning (sure loser);
* exact one   -> no player still to act can possibly beat it (sure winner):
  no unseen higher card of its suit sits with a non-void player, and (for a
  non-trump) no non-void-in-trump player could ruff;
* otherwise   -> a product over the players still to act of
  1 - P(that player can beat it), each term a hypergeometric probability that a
  player holding k of the unseen cards (excluding suits it is known void in)
  holds a beating card (and, for ruffs, is void in the led suit).

Play, with need = bid - tricks_won and r = cards left in hand:

* need > 0: win efficiently.  Follow with the cheapest sure winner; else the
  cheapest card whose P(win) is within 0.1 of the best, if the best is high
  enough for the urgency need/r; else duck with the cheapest card.  Lead a sure
  winner (trumps first while opponents may hold trumps); with need/r >= 0.5
  and trump majority (own trumps >= unseen trumps) draw trumps with the top
  trump; with trump control (no unseen trumps) and fewer trumps than needed,
  cash side-suit sure winners, else lead the top of the longest side suit,
  keeping trumps as late winners; else the best-P(win) card when urgent, else a low card of the longest side suit.  Discard the
  lowest side-suit card when unable to win.
* need == 0: lose every remaining trick.  Follow/discard with the most valuable
  sure loser (sheds danger while it is safe); if none, the card with the lowest
  P(win).  Lead the lowest-P(win) card (ties: cheapest).
* missed (need < 0 or need > r): neutral -- play the cheapest legal card.
  Under miss_penalty="neg_diff" the distance still matters, so need < 0 plays
  as need == 0 and need > r plays as need == r ("damage control").

Card value for "cheap"/"valuable": rank, plus 13 if trump.

P(win) estimate (default ``joint=True``): the chance that no later player
follows with a higher card is computed jointly -- the x higher unseen cards are
spread uniformly over the remaining slots of holders not known void in the suit
(opponents + undealt stock), so a led K with the A unseen and no stock has
P(win) = 0.  Ruff risk: jointly for later players certainly void in the led
suit, per-player hypergeometric P(void & holds trump) otherwise.  ``joint=False``
is the first version (independent per-player probabilities), kept for comparison.

TASpitePolicy is identical while its own outcome is still open.  Once it has
missed (need < 0 or need > r) and the penalty does not depend on the distance,
every card is indifferent for itself, so it picks the card minimising the
opponents' expected benefit: each opponent q has utility u_q = +1 for a trick
if it still needs tricks and can still make its bid, -1 if it is exactly at its
bid, 0 if it has already missed; the policy minimises sum_q P(q wins trick)*u_q
using the same win-probability estimates.
"""
from __future__ import annotations

from math import comb

from ..cards import FULL_DECK, NT, SUIT_MASK, SUIT_OF
from .base import nearest_legal

_RANKMASK = 0x1FFF
# _C[n][k] = C(n, k) for 0 <= n, k <= 52
_C = [[comb(n, k) for k in range(53)] for n in range(53)]


def _p_none(u: int, x: int, k: int) -> float:
    """P(a hand of k cards drawn from u holds none of x specific cards)."""
    if x <= 0:
        return 1.0
    if k > u:
        k = u
    if u - x < k:
        return 0.0
    return _C[u - x][k] / _C[u][k]


def _bits(m: int) -> list[int]:
    out = []
    while m:
        low = m & -m
        out.append(low.bit_length() - 1)
        m ^= low
    return out


class TrickModel:
    """Per-decision view of the current trick with card-tracking estimates."""

    __slots__ = ("trump", "unseen", "later", "voids", "k", "wc", "ws", "led", "me", "N", "joint",
                 "cap", "stock")

    def __init__(self, info, joint: bool = True):
        self.me = me = info.seat
        self.N = N = info.n_players
        self.trump = info.trump
        hand = info.hand
        self.unseen = FULL_DECK & ~hand & ~info.played_mask
        tl = info.trick_len
        self.later = [(me + i) % N for i in range(1, N - tl)]
        self.voids = info.voids
        self.k = hand.bit_count()          # cards each later player still holds
        self.wc = info.winning_card
        self.ws = info.winning_seat
        trick = info.current_trick
        self.led = SUIT_OF[trick[0][1]] if trick else -1
        self.joint = joint
        self.cap = {q: info.cards_left(q) for q in range(N) if q != me}
        self.stock = self.unseen.bit_count() - sum(self.cap.values())

    # ---- joint ("someone must hold it") estimates --------------------------------
    def _p_none_joint(self, x: int, suit: int, players) -> float:
        """P(none of x specific unseen cards of ``suit`` sits with ``players``),
        placing them uniformly over the slots of holders not known void in suit
        (opponents' remaining cards + undealt stock)."""
        if x <= 0:
            return 1.0
        v = self.voids
        elig = self.stock
        for q, c in self.cap.items():
            if not v[q] >> suit & 1:
                elig += c
        K = 0
        for q in players:
            if not v[q] >> suit & 1:
                K += self.cap[q]
        if K == 0:
            return 1.0
        if elig - K < x:
            return 0.0
        return _C[elig - K][x] / _C[elig][x]

    def _p_void_and_holds(self, q: int, led: int, X: int) -> float:
        """Per-player P(q void in ``led`` and holds >= 1 of X cards) (hypergeometric)."""
        vq = self.voids[q]
        pool = self.unseen
        for s in range(4):
            if vq >> s & 1:
                pool &= ~SUIT_MASK[s]
        u = pool.bit_count()
        if u == 0 or X == 0:
            return 0.0
        L = (pool & SUIT_MASK[led]).bit_count()
        k = self.cap[q]
        return _p_none(u, L, k) - _p_none(u, L + X, k)

    def p_survive(self, best: int, led: int) -> float:
        """P(no player still to act can beat ``best`` in a trick led in ``led``)."""
        later = self.later
        if not later:
            return 1.0
        if not self.joint:
            p = 1.0
            for q in later:
                p *= 1.0 - self.p_beat(q, best, led)
            return max(p, 0.0)
        trump, unseen, v = self.trump, self.unseen, self.voids
        bs = SUIT_OF[best]
        higher = (unseen & SUIT_MASK[bs] & ~((2 << best) - 1)).bit_count()
        no_led_left = not (unseen & SUIT_MASK[led])
        sure_void = [q for q in later if no_led_left or v[q] >> led & 1]
        maybe_void = [q for q in later if q not in sure_void]
        if bs == led:
            # (a) a later player follows with a higher card of the led suit
            p = self._p_none_joint(higher, bs, later)
            # (b) a later player void in the led suit ruffs
            if trump != NT and bs != trump:
                T = (unseen & SUIT_MASK[trump]).bit_count()
                if T:
                    p *= self._p_none_joint(T, trump, sure_void)
                    for q in maybe_void:
                        if not v[q] >> trump & 1:
                            p *= 1.0 - self._p_void_and_holds(q, led, T)
            return max(p, 0.0)
        # best is a trump on a non-trump lead: only void players holding a higher trump beat it
        if not higher:
            return 1.0
        p = self._p_none_joint(higher, trump, sure_void)
        for q in maybe_void:
            if not v[q] >> trump & 1:
                p *= 1.0 - self._p_void_and_holds(q, led, higher)
        return max(p, 0.0)

    def beats(self, c: int, best: int) -> bool:
        s, bs = SUIT_OF[c], SUIT_OF[best]
        if s == bs:
            return c > best
        return s == self.trump

    def p_beat(self, q: int, best: int, led: int) -> float:
        """P(player q, still to act, can beat ``best`` in a trick led in ``led``)."""
        trump = self.trump
        vq = self.voids[q]
        pool = self.unseen
        for s in range(4):
            if vq >> s & 1:
                pool &= ~SUIT_MASK[s]
        u = pool.bit_count()
        k = self.k
        if u == 0:
            return 0.0
        bs = SUIT_OF[best]
        higher = (pool & SUIT_MASK[bs] & ~((2 << best) - 1)).bit_count()
        if bs == led:
            p = 1.0 - _p_none(u, higher, k)
            if trump != NT and bs != trump:
                L = (pool & SUIT_MASK[led]).bit_count()
                T = (pool & SUIT_MASK[trump]).bit_count()
                if T:
                    p += _p_none(u, L, k) - _p_none(u, L + T, k)
            return p
        # best is a trump, led suit is not trump: q must be void in led and hold a higher trump
        if not higher:
            return 0.0
        L = (pool & SUIT_MASK[led]).bit_count()
        return _p_none(u, L, k) - _p_none(u, L + higher, k)

    def p_win(self, c: int) -> float:
        """P(card c, played now, wins the trick)."""
        if self.wc >= 0:
            if not self.beats(c, self.wc):
                return 0.0
            led = self.led
        else:
            led = SUIT_OF[c]
        return self.p_survive(c, led)

    def winner_dist(self, c: int) -> dict[int, float]:
        """Rough P(each seat wins the trick | I play c)."""
        pw = self.p_win(c)
        out = {self.me: pw}
        rest = 1.0 - pw
        if self.wc >= 0 and not self.beats(c, self.wc):
            # current winner keeps it unless a later player beats it
            keep = self.p_survive(self.wc, self.led)
            out[self.ws] = out.get(self.ws, 0.0) + keep
            rest = 1.0 - keep
        if rest > 0 and self.later:
            share = rest / len(self.later)
            for q in self.later:
                out[q] = out.get(q, 0.0) + share
        elif rest > 0 and self.ws >= 0:
            out[self.ws] = out.get(self.ws, 0.0) + rest
        return out


def card_value(c: int, trump: int) -> int:
    return (c % 13) + (13 if SUIT_OF[c] == trump else 0)


class TargetAwarePolicy:
    """Selfish target-aware play.  Bidding is delegated (see judgement.bidding);
    the built-in bid is a crude honour count so the class is a full Policy."""

    name = "ta"

    def __init__(self, joint: bool = True):
        # joint=False reproduces the first version (v0): independent per-player
        # beat probabilities, which ignore that every unseen card is held by
        # someone (badly over-estimates e.g. a led K when the A is out).
        self.joint = joint
        if not joint:
            self.name = self.name + "_v0"

    # ---- bidding (fallback only) ------------------------------------------
    def bid(self, info, legal, rng):
        hand, trump = info.hand, info.trump
        est = 0
        for s in range(4):
            m = (hand >> (13 * s)) & _RANKMASK
            if m >> 12 & 1:
                est += 1
            if s == trump:
                est += (m >> 11 & 1) + (m >> 10 & 1) + max(0, m.bit_count() - 3)
        return nearest_legal(min(est, info.n_cards), legal)

    # ---- play ---------------------------------------------------------------
    def play(self, info, legal_mask, rng):
        if legal_mask & (legal_mask - 1) == 0:
            return legal_mask.bit_length() - 1
        seat = info.seat
        bid = None
        for p, b in info.bids:
            if p == seat:
                bid = b
                break
        trump = info.trump
        r = info.hand.bit_count()
        need = (bid if bid is not None else 0) - info.tricks_won[seat]
        missed = need < 0 or need > r
        if missed:
            if info.rules.miss_penalty == "neg_diff":
                need = 0 if need < 0 else r
            else:
                return self.on_missed(info, legal_mask, trump)
        tm = TrickModel(info, self.joint)
        cards = _bits(legal_mask)
        val = lambda c: card_value(c, trump)  # noqa: E731
        pw = {c: tm.p_win(c) for c in cards}
        if need > 0:
            return self._win(info, tm, cards, pw, need, r, trump, val)
        return self._lose(tm, cards, pw, val)

    def on_missed(self, info, legal_mask, trump):
        return min(_bits(legal_mask), key=lambda c: (card_value(c, trump), c))

    # need > 0
    def _win(self, info, tm, cards, pw, need, r, trump, val):
        sure = [c for c in cards if pw[c] >= 1.0]
        urgency = need / r
        if tm.wc >= 0:  # following / discarding
            if sure:
                return min(sure, key=val)
            best = max(pw.values())
            if best > 0 and best >= 0.5 * (1.0 - urgency):
                good = [c for c in cards if pw[c] >= best - 0.1]
                return min(good, key=val)
            # duck: cheapest card, preferring side suits
            return min(cards, key=lambda c: (SUIT_OF[c] == trump, val(c)))
        # leading
        hand = info.hand
        if self.joint and trump != NT and not tm.unseen & SUIT_MASK[trump]:
            mine_t = hand & SUIT_MASK[trump]
            side = [c for c in cards if SUIT_OF[c] != trump]
            if mine_t and side and mine_t.bit_count() < need:
                # trump control: my trumps are winners whenever led, so first
                # cash / establish side suits (longest first) while they remain entries
                ss = [c for c in side if pw[c] >= 1.0]
                if ss:
                    return min(ss, key=val)
                return max(side, key=lambda c: ((hand & SUIT_MASK[SUIT_OF[c]]).bit_count(), c % 13))
        if sure:
            if trump != NT and tm.unseen & SUIT_MASK[trump]:
                st = [c for c in sure if SUIT_OF[c] == trump]
                if st:
                    return max(st, key=val)
            return min(sure, key=val)
        hand = info.hand
        if self.joint and trump != NT and urgency >= 0.5:
            mine = hand & SUIT_MASK[trump]
            out_t = (tm.unseen & SUIT_MASK[trump]).bit_count()
            opp_may = any(not tm.voids[q] >> trump & 1 for q in tm.cap)
            if mine and out_t and opp_may and mine.bit_count() >= out_t:
                return mine.bit_length() - 1      # trump majority: draw trumps
        best = max(pw.values())
        if best >= 0.5 * (1.0 - urgency):
            good = [c for c in cards if pw[c] >= best - 0.1]
            return min(good, key=val)
        side = [c for c in cards if SUIT_OF[c] != trump] or cards
        return min(side, key=lambda c: (-(hand & SUIT_MASK[SUIT_OF[c]]).bit_count(), val(c)))

    # need == 0
    def _lose(self, tm, cards, pw, val):
        if tm.wc >= 0:
            losers = [c for c in cards if pw[c] <= 0.0]
            if losers:
                return max(losers, key=val)
            return min(cards, key=lambda c: (pw[c], -val(c)))
        return min(cards, key=lambda c: (round(pw[c], 9), val(c)))


class TASpitePolicy(TargetAwarePolicy):
    """TA for its own target; once its own outcome is decided (missed) it plays
    the card that most hurts the opponents' chances of hitting."""

    name = "ta_spite"

    def on_missed(self, info, legal_mask, trump):
        tm = TrickModel(info, self.joint)
        N = info.n_players
        bids = dict(info.bids)
        won = info.tricks_won
        r_after = info.hand.bit_count()  # tricks left including this one
        util = {}
        for q in range(N):
            if q == info.seat:
                continue
            nq = bids.get(q, 0) - won[q]
            if nq == 0:
                util[q] = -1.0
            elif 0 < nq <= r_after:
                util[q] = 1.0
            else:
                util[q] = 0.0
        if not any(util.values()):
            return super().on_missed(info, legal_mask, trump)
        best_c, best_key = -1, None
        for c in _bits(legal_mask):
            wd = tm.winner_dist(c)
            benefit = sum(p * util.get(q, 0.0) for q, p in wd.items())
            key = (round(benefit, 9), card_value(c, trump), c)
            if best_key is None or key < best_key:
                best_c, best_key = c, key
        return best_c
