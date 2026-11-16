"""Tests for Phase 3 additions: sequential bid-posterior sampler, oracle bidder, determinism."""
from pathlib import Path

import numpy as np
import pytest

from judgement import Rules
from judgement.bidding.model import Agent, BidModel, ModelBidder
from judgement.bidding2.sampler import BidLikelihood, exact_posterior_worlds
from judgement.bidding2.seqsampler import SeqPosteriorSampler, posterior_worlds
from judgement.engine import deal, run_round
from judgement.policies.target_aware import TargetAwarePolicy
from judgement.state import RoundState

MODEL = Path(__file__).resolve().parents[1] / "experiments/phase1/results/model_ta_final.json"


@pytest.fixture(scope="module")
def model():
    if not MODEL.exists():
        pytest.skip("model file missing")
    return BidModel.load(MODEL)


def _situation(model, n, pos, seed, trump=0):
    rules = Rules()
    rng = np.random.default_rng(seed)
    mb = ModelBidder(model)
    st = RoundState(rules, n, trump, 3, deal(rules, n, rng))
    for _ in range(pos):
        st.apply_bid(mb.bid(st.info_state(st.to_act), st.legal_bids(), rng))
    return rules, mb, st, rng


@pytest.mark.parametrize("n,pos", [(3, 2), (5, 3), (8, 1)])
def test_seq_worlds_consistent_with_bids(model, n, pos):
    rules, mb, st, rng = _situation(model, n, pos, n * 10 + pos)
    info = st.info_state(pos)
    lik = BidLikelihood(model, rules, 0.0, 0.0)
    worlds, stats = SeqPosteriorSampler(info, lik).sample(40, rng)
    assert len(worlds) == 40 and stats["ess"] > 1
    for w in worlds:
        hands = [info.hand if p == pos else w[p] for p in range(4)]
        assert all(h.bit_count() == n for h in hands)
        assert bin(hands[0] | hands[1] | hands[2] | hands[3]).count("1") == 4 * n
        s2 = RoundState(rules, n, st.trump, 3, hands)
        for p in range(pos):
            b = mb.bid(s2.info_state(p), s2.legal_bids(), rng)
            assert b == st.bids[p]
            s2.apply_bid(b)


def test_seq_matches_rejection_posterior_mean(model):
    rules, mb, st, rng = _situation(model, 5, 2, 7)
    info = st.info_state(2)
    lik = BidLikelihood(model, rules, 0.0, 0.0)
    hands, lw = SeqPosteriorSampler(info, lik).particles(3000, rng, B=32)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    tl = lambda m: ((int(m) & 0x1FFF).bit_count())  # noqa: E731  spade (trump) length
    seq = [(w * np.array([tl(x) for x in hands[j]])).sum() for j in range(2)]
    rej, _ = exact_posterior_worlds(info, lik, 3000, rng, batch=8000, max_batches=60)
    ref = [np.mean([tl(x[j]) for x in rej]) for j in range(2)]
    assert np.allclose(seq, ref, atol=0.12)


def test_posterior_worlds_returns_k_and_uniform_without_bids(model):
    rules, mb, st, rng = _situation(model, 13, 3, 11)
    lik = BidLikelihood(model, rules, 0.0, 0.0)
    w, _ = posterior_worlds(st.info_state(3), lik, 25, rng, max_reject=4000)
    assert len(w) == 25
    rules, mb, st0, rng = _situation(model, 4, 0, 12)
    w0, s0 = posterior_worlds(st0.info_state(0), lik, 10, rng)
    assert len(w0) == 10 and s0["ess"] == 10


def test_ta_table_round_is_deterministic(model):
    rules = Rules()
    ag = Agent(ModelBidder(model), TargetAwarePolicy())
    hands = deal(rules, 9, np.random.default_rng(3))
    out = []
    for s in (1, 2):
        st = RoundState(rules, 9, 2, 1, hands)
        run_round(st, [ag] * 4, [np.random.default_rng(s)] * 4, check=True)
        out.append((tuple(st.bids), tuple(st.tricks_won)))
    assert out[0] == out[1]


def test_oracle_bidder_legal(model):
    from judgement.bidding2.oracle import InformedOracleBidder
    rules, mb, st, rng = _situation(model, 3, 3, 21)
    ag = Agent(ModelBidder(model), TargetAwarePolicy())
    ob = InformedOracleBidder(model, rules, [ag] * 4, K=8, seed=1)
    legal = st.legal_bids()
    assert ob.bid(st.info_state(3), legal, rng) in legal
