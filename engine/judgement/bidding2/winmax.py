"""Standings-aware (win-the-game) bidder for the late game.

P(win | bid b) is estimated with a cheap model of everything except the hero's
own current-round outcome:

* hero, this round: hit with probability P_model(t = b | hand, b) (Phase-1 bidder
  model, which already prices how TA play pulls tricks toward the bid);
* opponents who already bid: hit with the empirical TA self-play rate
  P(hit | n, position, bid) from games_ta.npz;
* opponents still to bid, and every player's future rounds: whole score rows
  drawn from a random game of games_ta.npz (same game for all four seats, so
  within-round dependence such as "tricks are shared" is kept for those rows);
  seat/position alignment is exact because dealer rotation is fixed by round index.

The same L draws (game indices and Bernoulli uniforms) are used for every
candidate bid (CRN).  Fractional wins for ties, as in the engine.
"""
from __future__ import annotations

import numpy as np

from ..bidding.model import info_features, score_matrix


class ScoreBank:
    def __init__(self, path):
        d = np.load(path)
        self.scores = d["scores"].astype(float)       # (G, R, N)
        self.bids = d["bids"].astype(int)
        G, R, N = self.scores.shape
        self.G, self.R, self.N = G, R, N
        # suffix sums: future[g, r, p] = sum of rounds > r
        cs = np.cumsum(self.scores[:, ::-1, :], axis=1)[:, ::-1, :]
        self.future = np.concatenate([cs[:, 1:, :], np.zeros((G, 1, N))], axis=1)
        self._hit: dict = {}

    def hit_rate(self, r: int, pos: int, bid: int) -> float:
        """Empirical P(hit | round index r (=> n, trump), bidding position, bid)."""
        key = (r, pos, bid)
        v = self._hit.get(key)
        if v is None:
            dealer = r % self.N
            seat = (dealer + 1 + pos) % self.N
            m = self.bids[:, r, seat] == bid
            k = int(m.sum())
            hits = (self.scores[m, r, seat] > 0).sum() if k else 0
            v = (hits + 1.0) / (k + 2.0)          # Laplace-smoothed
            self._hit[key] = v
        return v


class WinMaxBidder:
    """Bid to maximise P(win the game).  Before each bid the game driver must set
    ``ctx = (totals_before_round, round_index)``.  Falls back to the score-max
    model bid when ctx is None or the round is not among the final k."""

    name = "winmax"

    def __init__(self, model, rules, bank: ScoreBank, k_final: int, L: int = 3000, seed: int = 0):
        self.model, self.rules, self.bank, self.k = model, rules, bank, k_final
        self.L = L
        self.rng = np.random.default_rng(seed)
        self.ctx = None
        self._S: dict = {}
        self.log = []          # (round, model_bid, chosen_bid, pwin_model_bid, pwin_chosen)
        self.detail = []       # per final-round decision: standings, P(hit) and P(win) of each legal bid

    def _model_ev(self, info):
        n = info.n_cards
        S = self._S.get(n)
        if S is None:
            S = self._S[n] = score_matrix(self.rules, n)
        P = self.model.get(n, info.trump).dist_all_bids(info_features(info))
        return (P * S).sum(1), P

    def pwin_by_bid(self, info, legal):
        totals, r = self.ctx
        N = info.n_players
        me = info.seat
        bank = self.bank
        g = self.rng.integers(bank.G, size=self.L)
        base = np.asarray(totals, float)[None, :] + bank.future[g, r, :]      # (L, N)
        bid_seats = {p: b for p, b in info.bids}
        cur = np.zeros((self.L, N))
        for p in range(N):
            if p == me:
                continue
            if p in bid_seats:
                pos = (p - info.dealer - 1) % N
                q = bank.hit_rate(r, pos, bid_seats[p])
                cur[:, p] = np.where(self.rng.random(self.L) < q, self.rules.score(bid_seats[p], bid_seats[p]),
                                     self.rules.score(bid_seats[p], -1))
            else:
                cur[:, p] = bank.scores[g, r, p]
        tot = base + cur
        others = np.delete(tot, me, axis=1)
        top = others.max(1)
        ntop = (others == top[:, None]).sum(1)

        def W(x):
            mine = tot[:, me] + x
            return np.where(mine > top, 1.0, np.where(mine == top, 1.0 / (ntop + 1), 0.0)).mean()

        ev, P = self._model_ev(info)
        miss = self.rules.score(1, 0)
        out = {}
        for b in legal:
            p = P[b, b]
            out[b] = p * W(self.rules.score(b, b)) + (1 - p) * W(miss)
        return out, ev

    def bid(self, info, legal, rng):
        ev, _ = self._model_ev(info)
        bm = max(legal, key=lambda b: (ev[b], -b))
        if self.ctx is None or self.ctx[1] < self.bank.R - self.k:
            return bm
        pw, _ = self.pwin_by_bid(info, legal)
        b = max(legal, key=lambda x: (pw[x], ev[x], -x))
        self.log.append((self.ctx[1], bm, b, pw[bm], pw[b]))
        _, P = self._model_ev(info)
        self.detail.append({"r": self.ctx[1], "totals": list(self.ctx[0]), "seat": info.seat, "n": info.n_cards,
                            "bm": bm, "b": b, "pw": {int(x): float(v) for x, v in pw.items()},
                            "phit": {int(x): float(P[x, x]) for x in legal}, "ev": {int(x): float(ev[x]) for x in legal}})
        return b
