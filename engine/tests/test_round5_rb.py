import importlib.util
from pathlib import Path

import numpy as np

from judgement import Rules, play_game
from judgement.bidding2.sampler import BidLikelihood
from judgement.policies.greedy import GreedyPolicy
from judgement.policies.hybrid_bi import _bool
from judgement.policies.hybrid_rb import BidderTypes, OpponentModel, temper, honour_bids

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("r5_agents", ROOT / "experiments" / "round5_bots" / "agents.py")
A = importlib.util.module_from_spec(_s); _s.loader.exec_module(A)


class _Rec:
    """record (info snapshot fields, legal, bid, model bid) for one seat's bidder"""
    def __init__(self, inner, mb=None):
        self.inner, self.mb, self.rec = inner, mb, []

    def bid(self, info, legal, rng):
        b = self.inner.bid(info, legal, rng)
        m = self.mb.bid(info, legal, rng) if self.mb else None
        pos = (info.seat - info.dealer - 1) % info.n_players
        self.rec.append((info.hand, info.n_cards, info.trump, pos, [x for _, x in info.bids], list(legal), b, m))
        return b

    def play(self, info, legal_mask, rng):
        return self.inner.play(info, legal_mask, rng)


def _records(kind, games=2):
    from judgement.bidding.model import ModelBidder
    recs = []
    for g in range(games):
        ags = [A.make_agent("ta_plus") if hasattr(A, "make_agent") else None for _ in range(4)]
        a = A.make_agent(kind, seed=g)
        ags[0] = _Rec(a.bidder, ModelBidder(A.M_TA))
        ags[0].play = a.player.play
        ags = [ags[0]] + [A.make_agent("ta_plus") for _ in range(3)]
        play_game(Rules(), ags, seed=900 + g, record_trace=False)
        recs += ags[0].rec
    return recs


def test_honour_vectorised_matches_greedy():
    class I:  # minimal info
        pass
    rng = np.random.default_rng(0)
    gp = GreedyPolicy()
    for _ in range(300):
        n = int(rng.integers(1, 14)); t = int(rng.integers(-1, 4))
        hand = rng.choice(52, n, replace=False)
        m = sum(1 << int(c) for c in hand)
        legal = sorted(set(range(n + 1)) - ({int(rng.integers(0, n + 1))} if rng.random() < .5 else set()))
        i = I(); i.hand, i.trump, i.n_cards = m, t, n
        assert honour_bids(_bool([m]), n, t, legal)[0] == gp.bid(i, legal, None)


def test_type_probs_exact_for_model_and_honour_bidders():
    T = BidderTypes(BidLikelihood(A.M_TA, Rules()), eps=0.02)
    for kind, col in (("hybrid", 0), ("hyb_honour", 4)):
        for hand, n, t, pos, prior, legal, b, _ in _records(kind):
            P = T.probs(_bool([hand]), n, t, pos, prior, b, legal)[0]
            assert P[col] > 0.9, (kind, P)
            assert np.all(P > 0) and abs(P[-1] - 1 / len(legal)) < 1e-12


def test_noisy_probs_cover_noisy_bids():
    T = BidderTypes(BidLikelihood(A.M_TA, Rules()), eps=0.0)
    for hand, n, t, pos, prior, legal, b, m in _records("hyb_noisy:p=0.3"):
        P = T.probs(_bool([hand]), n, t, pos, prior, b, legal)[0]
        assert P[2] > 0.1           # noisy0.25 gives every NoisyBidder outcome positive mass
        # probabilities over legal bids sum to 1 for each type
        allP = np.array([T.probs(_bool([hand]), n, t, pos, prior, bb, legal)[0] for bb in legal])
        assert np.allclose(allP.sum(0), 1.0)


def test_posterior_concentrates():
    T = BidderTypes(BidLikelihood(A.M_TA, Rules()), eps=0.02)
    for kind, col in (("hybrid", 0), ("hyb_honour", 4)):
        om = OpponentModel(T)
        for hand, n, t, pos, prior, legal, b, _ in _records(kind, games=1):
            om.update(1, _bool([hand])[0], n, t, pos, prior, b, legal)
        assert om.post(1)[col] > 0.8, (kind, om.post(1))


def test_temper_restores_ess():
    lw = np.array([0.0] + [-30.0] * 200)
    w, e, beta = temper(lw, 4.0)
    assert e >= 4.0 - 1e-6 and beta < 1 and abs(w.sum() - 1) < 1e-12
    w, e, beta = temper(np.zeros(50), 4.0)
    assert beta == 1.0 and abs(e - 50) < 1e-9


def test_rb_and_pf_play_legal_games_and_log_posteriors():
    for kind in ("hybrid_rb:k=4,c=3", "hybrid_rb:k=4,c=3,pf=1", "hybrid_bipf:k=4,c=3"):
        a = A.make_agent(kind, seed=1)
        ags = [a] + [A.make_agent("hyb_honour:k=4,c=3", seed=s) for s in range(3)]
        rules = Rules()
        play_game(rules, ags, seed=77, record_trace=False)
        assert a.player.rounds_seen == rules.n_rounds - 1
        if "bipf" not in kind:
            assert len(a.player.post_log) == 3 * (rules.n_rounds - 1)
            # honour opponents: posterior mass on honour after the game
            assert np.mean([a.player.opp.post(s)[4] for s in (1, 2, 3)]) > 0.5
        assert all(st["ess"] >= min(4.0, st["M"]) - 1e-6 or "kept" in st for st in a.player.log)


def test_particles_consistent_with_true_play():
    """PF worlds never contradict the public record (played cards, voids)."""
    from judgement.sampling import HandSampler  # noqa: F401
    a = A.make_agent("hybrid_rb:k=8,c=6,pf=1", seed=3)
    pl = a.player
    seen = []
    orig = pl.worlds.sample

    def chk(info, K, rng):
        ws, st = orig(info, K, rng)
        unseen = ~info.hand & ~info.played_mask
        for w in ws:
            tot = 0
            for p, h in w.items():
                assert h & unseen == h
                assert h.bit_count() == info.cards_left(p)
                for s in range(4):
                    if info.voids[p] >> s & 1:
                        assert h & (0x1FFF << (13 * s)) == 0
                assert tot & h == 0
                tot |= h
        seen.append(st)
        return ws, st
    pl.worlds.sample = chk
    ags = [a] + [A.make_agent("hybrid:k=4,c=3", seed=s) for s in range(3)]
    play_game(Rules(), ags, seed=5, record_trace=False)
    assert seen and any(st["kept"] > 0 for st in seen)


def test_particle_pool_resets_each_round():
    """Regression: the pool is keyed on the InfoState object itself (an id() key could be reused by a later
    round with the same n_cards and resurrect stale particles)."""
    a = A.make_agent("hybrid_rb:k=4,c=4,pf=1", seed=2)
    pl = a.player
    keys = []
    orig = pl.worlds.sample

    def chk(info, K, rng):
        ws, st = orig(info, K, rng)
        assert pl.worlds._key is info
        unseen = ~info.hand & ~info.played_mask
        assert all(h & unseen == h for w in ws for h in w.values())
        keys.append(info)
        return ws, st
    pl.worlds.sample = chk
    rules = Rules()
    ags = [a] + [A.make_agent("hyb_honour:k=4,c=3", seed=s) for s in range(3)]
    play_game(rules, ags, seed=11, record_trace=False)
    assert len({id(k) for k in keys}) >= 2
