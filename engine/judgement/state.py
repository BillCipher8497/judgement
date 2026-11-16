"""Round state (full information) and per-player information views.

``RoundState`` is mutable with O(1) ``apply_card`` / ``undo_card`` and a cheap
``copy()`` so a perfect-information solver can search over it later.

``InfoState`` is a *read-only view* of a RoundState from one seat.  Every public
attribute is observable by that seat; other players' hands are not reachable
through its API.  (It is a view for speed; call ``snapshot()`` for a detached,
immutable copy with no reference to the underlying RoundState.)
"""
from __future__ import annotations

from dataclasses import dataclass

from .cards import BIT, FULL_DECK, NT, SUIT_MASK, SUIT_OF, card_str, cards_of, hand_str, suit_str
from .rules import Rules

BID, PLAY, DONE = 0, 1, 2


class IllegalAction(ValueError):
    pass


class RoundState:
    __slots__ = (
        "rules", "N", "n_cards", "trump", "dealer", "hands", "bids", "bid_seq",
        "tricks_won", "leader", "to_act", "trick", "best_card", "best_seat",
        "history", "voids", "trick_winners", "phase", "_undo",
    )

    def __init__(self, rules: Rules, n_cards: int, trump: int, dealer: int, hands: list[int]):
        N = rules.n_players
        if len(hands) != N:
            raise ValueError("need one hand per player")
        seen = 0
        for h in hands:
            if h.bit_count() != n_cards or h & seen or h & ~FULL_DECK:
                raise ValueError("hands must be disjoint, each with n_cards cards")
            seen |= h
        if trump != NT and not 0 <= trump < 4:
            raise ValueError("bad trump")
        self.rules = rules
        self.N = N
        self.n_cards = n_cards
        self.trump = trump
        self.dealer = dealer
        self.hands = list(hands)
        self.bids = [-1] * N
        self.bid_seq: list[tuple[int, int]] = []
        self.tricks_won = [0] * N
        self.leader = (dealer + 1) % N
        self.to_act = self.leader
        self.trick: list[tuple[int, int]] = []
        self.best_card = -1
        self.best_seat = -1
        self.history: list[tuple[int, int]] = []
        self.voids = [0] * N  # bit s set => player known void in suit s
        self.trick_winners: list[int] = []
        self.phase = BID
        self._undo: list[tuple] = []

    # ---- queries ---------------------------------------------------------
    @property
    def current_player(self) -> int:
        return -1 if self.phase == DONE else self.to_act

    @property
    def is_terminal(self) -> bool:
        return self.phase == DONE

    @property
    def led_suit(self) -> int:
        return SUIT_OF[self.trick[0][1]] if self.trick else -1

    def legal_bids(self) -> list[int]:
        if self.phase != BID:
            return []
        n = self.n_cards
        if self.rules.hook and len(self.bid_seq) == self.N - 1:
            forbidden = n - sum(b for _, b in self.bid_seq)
            return [b for b in range(n + 1) if b != forbidden]
        return list(range(n + 1))

    def legal_cards(self) -> int:
        """Bitmask of legal cards for the player to act."""
        if self.phase != PLAY:
            return 0
        hand = self.hands[self.to_act]
        if self.trick:
            follow = hand & SUIT_MASK[SUIT_OF[self.trick[0][1]]]
            if follow:
                return follow
        return hand

    def scores(self) -> list[float]:
        sc = self.rules.score
        return [sc(self.bids[p], self.tricks_won[p]) for p in range(self.N)]

    # ---- transitions -----------------------------------------------------
    def apply_bid(self, b: int) -> None:
        if self.phase != BID:
            raise IllegalAction("not in bidding phase")
        if b not in self.legal_bids():
            raise IllegalAction(f"illegal bid {b}")
        p = self.to_act
        self.bids[p] = b
        self.bid_seq.append((p, b))
        if len(self.bid_seq) == self.N:
            self.phase = PLAY
            self.to_act = self.leader
        else:
            self.to_act = (p + 1) % self.N

    def apply_card(self, c: int) -> None:
        if not self.legal_cards() & BIT[c]:
            raise IllegalAction(f"illegal card {card_str(c)} for seat {self.to_act}")
        self._apply_card(c)

    def _apply_card(self, c: int) -> None:
        """Unchecked apply (caller guarantees legality)."""
        p = self.to_act
        self.hands[p] ^= 1 << c
        trick = self.trick
        old_void = self.voids[p]
        if trick:
            s = SUIT_OF[c]
            bc = self.best_card
            bs = SUIT_OF[bc]
            led = SUIT_OF[trick[0][1]]
            if s != led:
                self.voids[p] = old_void | (1 << led)
            if s == bs:
                if c > bc:
                    self.best_card = c
                    self.best_seat = p
            elif s == self.trump:
                self.best_card = c
                self.best_seat = p
        else:
            self.best_card = c
            self.best_seat = p
        trick.append((p, c))
        self.history.append((p, c))
        if len(trick) == self.N:
            w = self.best_seat
            self.tricks_won[w] += 1
            self.trick_winners.append(w)
            self._undo.append((p, c, old_void, trick, self.leader))
            self.trick = []
            self.best_card = self.best_seat = -1
            self.leader = w
            self.to_act = w
            if len(self.trick_winners) == self.n_cards:
                self.phase = DONE
        else:
            self._undo.append((p, c, old_void, None, None))
            self.to_act = (p + 1) % self.N

    def undo_card(self) -> None:
        p, c, old_void, done_trick, old_leader = self._undo.pop()
        if done_trick is not None:
            w = self.trick_winners.pop()
            self.tricks_won[w] -= 1
            self.trick = done_trick
            self.leader = old_leader
            self.phase = PLAY
        self.trick.pop()
        self.history.pop()
        self.hands[p] |= 1 << c
        self.voids[p] = old_void
        self.to_act = p
        self._recompute_best()

    def _recompute_best(self) -> None:
        self.best_card = self.best_seat = -1
        trump = self.trump
        for p, c in self.trick:
            if self.best_card < 0:
                self.best_card, self.best_seat = c, p
                continue
            s, bs = SUIT_OF[c], SUIT_OF[self.best_card]
            if (s == bs and c > self.best_card) or (s != bs and s == trump):
                self.best_card, self.best_seat = c, p

    def copy(self) -> "RoundState":
        new = RoundState.__new__(RoundState)
        new.rules = self.rules
        new.N = self.N
        new.n_cards = self.n_cards
        new.trump = self.trump
        new.dealer = self.dealer
        new.hands = self.hands[:]
        new.bids = self.bids[:]
        new.bid_seq = self.bid_seq[:]
        new.tricks_won = self.tricks_won[:]
        new.leader = self.leader
        new.to_act = self.to_act
        new.trick = self.trick[:]
        new.best_card = self.best_card
        new.best_seat = self.best_seat
        new.history = self.history[:]
        new.voids = self.voids[:]
        new.trick_winners = self.trick_winners[:]
        new.phase = self.phase
        new._undo = self._undo[:]
        return new

    def info_state(self, seat: int) -> "InfoState":
        return InfoState(self, seat)

    def __repr__(self) -> str:
        hands = " | ".join(f"{p}:{hand_str(h)}" for p, h in enumerate(self.hands))
        return (f"RoundState(n={self.n_cards}, trump={suit_str(self.trump)}, dealer={self.dealer}, "
                f"phase={self.phase}, bids={self.bids}, tricks={self.tricks_won}, hands=[{hands}])")


class InfoState:
    """Read-only per-seat view.  Only observable information is exposed."""

    __slots__ = ("_s", "seat")

    def __init__(self, state: RoundState, seat: int):
        self._s = state
        self.seat = seat

    @property
    def rules(self) -> Rules:
        return self._s.rules

    @property
    def n_players(self) -> int:
        return self._s.N

    @property
    def dealer(self) -> int:
        return self._s.dealer

    @property
    def n_cards(self) -> int:
        return self._s.n_cards

    @property
    def trump(self) -> int:
        return self._s.trump

    @property
    def hand(self) -> int:
        return self._s.hands[self.seat]

    @property
    def bids(self) -> tuple[tuple[int, int], ...]:
        """(seat, bid) pairs in bidding order."""
        return tuple(self._s.bid_seq)

    @property
    def tricks_won(self) -> tuple[int, ...]:
        return tuple(self._s.tricks_won)

    @property
    def current_trick(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._s.trick)

    @property
    def trick_len(self) -> int:
        return len(self._s.trick)

    @property
    def led_suit(self) -> int:
        return self._s.led_suit

    @property
    def winning_card(self) -> int:
        """Card currently winning the trick in progress (-1 if none) — public info."""
        return self._s.best_card

    @property
    def winning_seat(self) -> int:
        return self._s.best_seat

    @property
    def history(self) -> tuple[tuple[int, int], ...]:
        """All (seat, card) plays this round, in order."""
        return tuple(self._s.history)

    @property
    def trick_winners(self) -> tuple[int, ...]:
        return tuple(self._s.trick_winners)

    @property
    def voids(self) -> tuple[int, ...]:
        return tuple(self._s.voids)

    @property
    def to_act(self) -> int:
        return self._s.current_player

    @property
    def phase(self) -> int:
        return self._s.phase

    @property
    def played_mask(self) -> int:
        m = 0
        for _, c in self._s.history:
            m |= 1 << c
        return m

    def cards_left(self, seat: int) -> int:
        """Number of cards seat still holds (public)."""
        return self._s.hands[seat].bit_count()

    def legal_bids(self) -> list[int]:
        return self._s.legal_bids() if self._s.to_act == self.seat else []

    def legal_cards(self) -> int:
        return self._s.legal_cards() if self._s.to_act == self.seat else 0

    def snapshot(self) -> "InfoSnapshot":
        s = self._s
        return InfoSnapshot(
            seat=self.seat, n_players=s.N, dealer=s.dealer, n_cards=s.n_cards, trump=s.trump,
            hand=self.hand, bids=self.bids, tricks_won=self.tricks_won,
            current_trick=self.current_trick, history=self.history,
            trick_winners=self.trick_winners, voids=self.voids, phase=s.phase,
            cards_left_counts=tuple(h.bit_count() for h in s.hands),
            rules=s.rules,
        )

    def __repr__(self) -> str:
        return (f"InfoState(seat={self.seat}, n={self.n_cards}, trump={suit_str(self.trump)}, "
                f"hand={hand_str(self.hand)}, bids={self.bids}, trick={[card_str(c) for _, c in self._s.trick]})")


@dataclass(frozen=True)
class InfoSnapshot:
    """Detached, immutable observation (no reference to the RoundState)."""
    seat: int
    n_players: int
    dealer: int
    n_cards: int
    trump: int
    hand: int
    bids: tuple
    tricks_won: tuple
    current_trick: tuple
    history: tuple
    trick_winners: tuple
    voids: tuple
    phase: int
    cards_left_counts: tuple
    rules: Rules

    def cards_left(self, seat: int) -> int:
        return self.cards_left_counts[seat]

    @property
    def played_mask(self) -> int:
        m = 0
        for _, c in self.history:
            m |= 1 << c
        return m

    @property
    def trick_len(self) -> int:
        return len(self.current_trick)

    @property
    def led_suit(self) -> int:
        return SUIT_OF[self.current_trick[0][1]] if self.current_trick else -1


def legal_bids(state: RoundState) -> list[int]:
    return state.legal_bids()


def legal_cards(state: RoundState) -> int:
    return state.legal_cards()


def state_from_info(info, hidden_hands: dict[int, int]) -> RoundState:
    """Rebuild a full RoundState from an observation plus a guess of the other
    players' *current* hands (as produced by sampling.HandSampler)."""
    rules = info.rules
    N = info.n_players
    played_by = [0] * N
    for p, c in info.history:
        played_by[p] |= 1 << c
    hands = []
    for p in range(N):
        cur = info.hand if p == info.seat else hidden_hands[p]
        hands.append(cur | played_by[p])
    st = RoundState(rules, info.n_cards, info.trump, info.dealer, hands)
    for _, b in info.bids:
        st.apply_bid(b)
    for _, c in info.history:
        st.apply_card(c)  # checked: a bad sample raises instead of corrupting
    return st


__all__ = ["RoundState", "InfoState", "InfoSnapshot", "IllegalAction", "legal_bids", "legal_cards",
           "state_from_info", "BID", "PLAY", "DONE", "cards_of"]
