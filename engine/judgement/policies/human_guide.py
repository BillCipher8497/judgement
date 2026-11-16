"""Human guide (Phase 5): a bidding count and a play checklist a person can apply at the table.

Everything here uses only *human-trackable* information:

* own hand, the current trick, all bids, tricks won (public);
* the number of trumps played (a running count), hence "trumps out" = 13 - mine - played;
* which honours (rank >= ``track``, default Q = A, K, Q of every suit) have been played;
* which opponents have shown out of which suit (voids).

No probability estimates.  A card is a *boss* if every higher card of its suit is in my hand or is a
remembered played honour (or is in the current trick); every trump is a boss once trumps out == 0.
A card *surely wins* the trick if it beats the current winner and either nobody plays after me, or it
is a boss and cannot be ruffed (it is a trump, or NT, or trumps out == 0, or every later player has
shown out of trumps).  *Sure tricks* E = number of bosses in my hand (a count, ruffs ignored).

Bidding: ``HumanBidder`` (integer half-trick count, see ``GUIDE_BID``).  Play: ``HumanPlayPolicy``,
the ordered checklist in ``play`` (each rule switchable for ablations).
"""
from __future__ import annotations

from ..cards import NT, RANK_OF, SUIT_MASK, SUIT_OF

A, K, Q, J, T10 = 12, 11, 10, 9, 8


def _bits(m: int) -> list[int]:
    out = []
    while m:
        low = m & -m
        out.append(low.bit_length() - 1)
        m ^= low
    return out


# ------------------------------------------------------------------------------------------------
# bidding
# ------------------------------------------------------------------------------------------------

def bid_counts(hand: int, trump: int, n: int) -> dict:
    """Human-countable hand facts."""
    c = dict(tA=0, tK=0, tQ=0, tlen=0, sA=0, sK2=0, sK1=0, sQ3=0, voids=0, singles=0, long5=0,
             sAlong=0, nt_long=0, low=0, sQ=0, sJ=0)
    for s in range(4):
        m = (hand >> (13 * s)) & 0x1FFF
        ln = m.bit_count()
        c["low"] += (m & 0x3F).bit_count()          # ranks 2..7, any suit
        hasA, hasK, hasQ = m >> A & 1, m >> K & 1, m >> Q & 1
        if s == trump:
            c["tlen"] = ln
            c["tA"], c["tK"], c["tQ"] = hasA, hasK, hasQ
            continue
        c["sA"] += hasA
        c["sAlong"] += hasA and ln >= 5
        c["sK2"] += hasK and ln >= 2
        c["sK1"] += hasK and ln == 1
        c["sQ3"] += hasQ and ln >= 3
        c["sQ"] += hasQ
        c["sJ"] += m >> J & 1
        c["voids"] += ln == 0
        c["singles"] += ln == 1
        c["nt_long"] += max(0, ln - 4)
    return c


# Weights are in HALF tricks.  Defaults = final guide rule (selected in experiments/guide/s1_back.py / s1_final.py).
GUIDE_BID = dict(
    # trump rounds
    t_each=2,        # every trump
    t_hon=1,         # extra for each trump A / K / Q (K needs >= 2 trumps, Q >= 3 if t_guard)
    t_guard=0,
    s_A=1, s_K2=1, s_K1=1, s_Q3=0,   # side aces / kings with >= 2 cards / singleton kings / queens with >= 3
    short=0,         # per side void (x2) or singleton, if holding >= 2 trumps
    t_sub=4,         # subtract one half-trick per t_sub cards dealt (n // t_sub); 99 = never
    # NT rounds
    n_A=2, n_K2=0, n_K1=0, n_Q3=0, n_long=0, n_sub=99,
    low_w=0,         # every card 2..7 (any suit, both round types): this many half-tricks (<= 0)
    s_Q=0, s_J=0,    # every side queen / jack (any length; both round types)
    # table
    small_cut=0,     # trump rounds with n <= small_cut: side aces count small_A, side kings (any length) small_K instead
    small_A=2, small_K=1,
    prior_k=4,       # earlier bids vs their share (pos * n / 4): -1 trick per prior_k tricks of excess (+1 per prior_k short)
    hook_up=0,       # dealer hooked on a whole count: 1 -> go up, 0 -> go down (a half remainder goes up)
    round_up=0,      # half-trick remainder: 1 -> round up, 0 -> down
)


def half_points_from_counts(w: dict, c: dict, trump: int, n: int) -> int:
    small = n <= w["small_cut"]
    if trump == NT:
        p = w["n_A"] * c["sA"] + w["n_K2"] * c["sK2"] + w["n_K1"] * c["sK1"]
        p += w["low_w"] * c["low"] + w["s_Q"] * c["sQ"] + w["s_J"] * c["sJ"]
        return p + w["n_Q3"] * c["sQ3"] + w["n_long"] * c["nt_long"] - n // w["n_sub"]
    L = c["tlen"]
    g = w["t_guard"]
    hon = c["tA"] + (c["tK"] and (L >= 2 or not g)) + (c["tQ"] and (L >= 3 or not g))
    p = w["t_each"] * L + w["t_hon"] * hon
    if small:
        p += w["small_A"] * c["sA"] + w["small_K"] * (c["sK2"] + c["sK1"])
    else:
        p += w["s_A"] * c["sA"] + w["s_K2"] * c["sK2"] + w["s_K1"] * c["sK1"]
    p += w["s_Q3"] * c["sQ3"] + w["low_w"] * c["low"] + w["s_Q"] * c["sQ"] + w["s_J"] * c["sJ"]
    if L >= 2:
        p += w["short"] * (2 * c["voids"] + c["singles"])
    return p - n // w["t_sub"]


def _round_half_down(x: float) -> int:
    import math
    return math.ceil(x - 0.5)


class HumanBidder:
    """bid = (half-trick count) / 2, rounded, +- table adjustment, hook-legal."""

    name = "guide_bid"

    def __init__(self, **kw):
        self.w = dict(GUIDE_BID)
        unknown = set(kw) - set(self.w)
        if unknown:
            raise ValueError(f"unknown guide bid params {unknown}")
        self.w.update(kw)

    def half_points(self, hand: int, trump: int, n: int) -> int:
        return half_points_from_counts(self.w, bid_counts(hand, trump, n), trump, n)

    def estimate(self, info) -> tuple[int, int]:
        """(bid before hook, half-trick remainder flag)."""
        n = info.n_cards
        p = max(0, self.half_points(info.hand, info.trump, n))
        b = p // 2 + (p % 2 if self.w["round_up"] else 0)
        N = info.n_players
        pos = (info.seat - info.dealer - 1) % N
        prior = sum(x for _, x in info.bids)
        share = pos * n / N
        b -= _round_half_down((prior - share) / self.w["prior_k"])
        return max(0, min(n, b)), p % 2

    def bid(self, info, legal, rng=None):
        b, half = self.estimate(info)
        if b in legal:
            return b
        # hooked (dealer): move toward the half-trick remainder, else by hook_up
        up = (half and not self.w["round_up"]) or (not half and self.w["hook_up"])
        if up and b + 1 in legal:
            return b + 1
        if b - 1 in legal:
            return b - 1
        return b + 1


# ------------------------------------------------------------------------------------------------
# play
# ------------------------------------------------------------------------------------------------

class HumanView:
    """What a careful human knows at a play decision."""

    __slots__ = ("trump", "hand", "known", "trumps_out", "voids", "later", "wc", "ws", "led", "seat")

    def __init__(self, info, track: int = Q):
        self.seat = me = info.seat
        self.trump = trump = info.trump
        self.hand = hand = info.hand
        played = info.played_mask
        trick = info.current_trick
        cur = 0
        for _, c in trick:
            cur |= 1 << c
        hon = 0
        for s in range(4):
            for r in range(track, 13):
                hon |= 1 << (13 * s + r)
        # remembered: honours played this round + the current trick (+ my hand)
        self.known = (played & hon) | cur | hand
        if trump != NT:
            tp = (played & SUIT_MASK[trump]).bit_count()
            self.trumps_out = 13 - (hand & SUIT_MASK[trump]).bit_count() - tp
        else:
            self.trumps_out = 0
        self.voids = info.voids
        N = info.n_players
        self.later = [(me + i) % N for i in range(1, N - len(trick))]
        self.wc = info.winning_card
        self.ws = info.winning_seat
        self.led = SUIT_OF[trick[0][1]] if trick else -1

    def boss(self, c: int) -> bool:
        s = SUIT_OF[c]
        if s == self.trump and self.trumps_out == 0:
            return True
        above = SUIT_MASK[s] & ~((2 << c) - 1)
        return (above & ~self.known) == 0

    def beats(self, c: int, best: int) -> bool:
        s, bs = SUIT_OF[c], SUIT_OF[best]
        if s == bs:
            return c > best
        return s == self.trump

    def can_win_now(self, c: int) -> bool:
        return self.wc < 0 or self.beats(c, self.wc)

    def ruff_safe(self) -> bool:
        if self.trump == NT or self.trumps_out == 0:
            return True
        return all(self.voids[q] >> self.trump & 1 for q in self.later)

    def sure_win(self, c: int) -> bool:
        if not self.can_win_now(c):
            return False
        if not self.later:
            return True
        if not self.boss(c):
            return False
        return SUIT_OF[c] == self.trump or self.ruff_safe()

    def sure_tricks(self) -> int:
        return sum(1 for c in _bits(self.hand) if self.boss(c))

    def value(self, c: int) -> int:
        return RANK_OF[c] + (13 if SUIT_OF[c] == self.trump else 0)


GUIDE_PLAY = dict(
    track=Q,        # remember played honours of rank >= this (Q: A, K, Q)
    r_hi_under=1,   # at target, following: play the highest card that cannot win (else lowest)
    r_lead_hi=0,    # at target, leading: lead the highest card beaten by an unplayed honour (1 always, 2 only n=13; 0 off)
    r_must=1,       # need == tricks left: cheapest sure winner / else highest; lead highest
    r_win_hi=1,     # need > 0 and a card surely wins: win with the highest sure winner if spare (E >= need), else cheapest
    shed_min=0,     # ... only if that card is a boss or ranks >= shed_min (13 = bosses only)
    shed_margin=0,  # ... and E (without it) >= need + shed_margin
    lead0=0,        # at target, leading: 0 lowest side card, 1 lowest card (trumps allowed), 2 lowest side card of the shortest suit
    nt_lead=1,      # need > 0, leading, NT, no boss: 0 low from longest suit, 1 highest card, 2 highest of longest suit
    win_margin=99,  # r_win_hi spends the highest sure winner when need == 1 or E >= need + win_margin
    trump_first=1,  # need > 0 leading with bosses: top trump first while trumps are out (0: side bosses first)
    r_draw=0,       # need > 0 leading, trump round, my trumps >= trumps out > 0 and 2*need >= tricks left: lead top trump
    r_shed=1,       # need > 0, not winning this trick: throw the highest loser if E (without it) >= need, else lowest
    r_lead_boss=1,  # need > 0 leading: lead a boss (top trump first while trumps are out)
    r_lead_high=1,  # need > 0 leading, no boss, trump round: lead highest side card (NT: low from longest)
    r_try=0,        # need > 0 following, no sure winner: play the highest card that beats the winner if 2*need >= tricks left
    r_boss_follow=1,  # need > 0 following, no sure winner: win with a boss that beats the winner (ruff risk accepted)
    r_ruff=1,       # need > 0 following, out of the led suit: ruff with the lowest trump that beats the winner
)


class HumanPlayPolicy:
    name = "guide_play"

    def __init__(self, **kw):
        self.w = dict(GUIDE_PLAY)
        unknown = set(kw) - set(self.w)
        if unknown:
            raise ValueError(f"unknown guide play params {unknown}")
        self.w.update(kw)

    def bid(self, info, legal, rng=None):  # so the class is a full Policy
        return HumanBidder().bid(info, legal, rng)

    def play(self, info, legal_mask, rng=None):
        if legal_mask & (legal_mask - 1) == 0:
            return legal_mask.bit_length() - 1
        w = self.w
        seat = info.seat
        bid = next((b for p, b in info.bids if p == seat), 0)
        r = info.hand.bit_count()
        need = bid - info.tricks_won[seat]
        hv = HumanView(info, w["track"])
        cards = _bits(legal_mask)
        val = hv.value
        low = lambda cs: min(cs, key=lambda c: (val(c), c))       # noqa: E731
        high = lambda cs: max(cs, key=lambda c: (val(c), -c))     # noqa: E731
        # R0 already missed: nothing matters for me -> lowest card
        if need < 0 or need > r:
            return low(cards)
        following = hv.wc >= 0
        if need == 0:
            if following:
                under = [c for c in cards if not hv.beats(c, hv.wc)]
                if under:
                    return high(under) if w["r_hi_under"] else low(under)
                # every card beats the winner: last -> highest (it wins anyway), else lowest
                return high(cards) if not hv.later else low(cards)
            if w["r_lead_hi"] and (w["r_lead_hi"] == 1 or 4 * info.n_cards >= 52):
                beaten = [c for c in cards if not hv.boss(c) and SUIT_OF[c] != hv.trump]
                # "beaten by an unplayed honour": some tracked honour above it is still out
                beaten = [c for c in beaten if _honour_out_above(hv, c, w["track"])]
                if len(beaten) >= 2:
                    return high(beaten)
            side = [c for c in cards if SUIT_OF[c] != hv.trump] or cards
            if w["lead0"] == 1:
                return low(cards)
            if w["lead0"] == 2:
                h = hv.hand
                return min(side, key=lambda c: ((h & SUIT_MASK[SUIT_OF[c]]).bit_count(), val(c)))
            return low(side)
        # need > 0
        E = hv.sure_tricks()
        sure = [c for c in cards if hv.sure_win(c)]
        if w["r_must"] and need == r:
            if following:
                return low(sure) if sure else high(cards)
            bosses = [c for c in cards if hv.boss(c)]
            tb = [c for c in bosses if SUIT_OF[c] == hv.trump]
            return high(tb or bosses or cards)
        if following:
            if sure:
                if w["r_win_hi"] and (need == 1 or E >= need + w["win_margin"]):
                    return high(sure)
                return low(sure)
            winners = [c for c in cards if hv.beats(c, hv.wc)]
            if w["r_boss_follow"]:
                bw = [c for c in winners if hv.boss(c)]
                if bw:
                    return high(bw) if (w["r_win_hi"] and (need == 1 or E >= need + w["win_margin"])) else low(bw)
            if w["r_ruff"] and hv.trump != NT and hv.led != hv.trump:
                tw = [c for c in winners if SUIT_OF[c] == hv.trump]
                if tw:
                    return low(tw)
            if w["r_try"] and winners and 2 * need >= r and (w["r_try"] == 1 or hv.trump == NT):
                bw = [c for c in winners if hv.boss(c)]
                return low(bw) if bw else high(winners)
            losers = [c for c in cards if not hv.beats(c, hv.wc)] or cards
            if w["r_shed"]:
                ok = [c for c in losers if E - hv.boss(c) >= need + w["shed_margin"]]
                if ok:
                    c = high(ok)
                    if RANK_OF[c] >= w["shed_min"] or hv.boss(c):
                        return c
            return min(losers, key=lambda c: (SUIT_OF[c] == hv.trump, val(c), c))
        # leading, need > 0
        if w["r_lead_boss"]:
            bosses = [c for c in cards if hv.boss(c)]
            if bosses:
                tb = [c for c in bosses if SUIT_OF[c] == hv.trump]
                if tb and hv.trumps_out > 0 and w["trump_first"]:
                    return high(tb)
                sb = [c for c in bosses if SUIT_OF[c] != hv.trump] or bosses
                if w["r_win_hi"] and (need == 1 or E >= need + w["win_margin"]):
                    return high(sb)
                return low(sb)
        hand = hv.hand
        if w["r_draw"] and hv.trump != NT and 2 * need >= r:
            mine = [c for c in cards if SUIT_OF[c] == hv.trump]
            if mine and 0 < hv.trumps_out <= len(mine):
                return high(mine)
        side = [c for c in cards if SUIT_OF[c] != hv.trump]
        if w["r_lead_high"] and side:
            if hv.trump == NT and w["nt_lead"] == 2:
                return max(side, key=lambda c: ((hand & SUIT_MASK[SUIT_OF[c]]).bit_count(), val(c)))
            if hv.trump != NT or w["nt_lead"]:
                return high(side)
            return min(side, key=lambda c: (-(hand & SUIT_MASK[SUIT_OF[c]]).bit_count(), val(c)))
        return low(side or cards)


def _honour_out_above(hv: HumanView, c: int, track: int) -> bool:
    s = SUIT_OF[c]
    for rr in range(max(track, RANK_OF[c] + 1), 13):
        x = 13 * s + rr
        if not hv.known >> x & 1:
            return True
    return False


class GuideAgent:
    """Guide bidding + guide play (or either half swapped for an ablation)."""

    def __init__(self, bidder=None, player=None, name="guide"):
        self.bidder = bidder or HumanBidder()
        self.player = player or HumanPlayPolicy()
        self.name = name

    def bid(self, info, legal, rng=None):
        return self.bidder.bid(info, legal, rng)

    def play(self, info, legal_mask, rng=None):
        return self.player.play(info, legal_mask, rng)


__all__ = ["HumanBidder", "HumanPlayPolicy", "HumanView", "GuideAgent", "GUIDE_BID", "GUIDE_PLAY", "bid_counts"]
