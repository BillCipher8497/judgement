"""Shared helpers for the play experiments (T1-T4).

Agents are built from a KIND string:
  greedy | ta | ta_spite | ta_plus | ta_plus:<k=v,...> | hybrid[:k=,c=,nt=] | tap_oracle[:K=,span=,nmax=]
  | hyb_oracle[:k=,c=,K=,span=,nmax=]
ta_plus options (defaults = final TA+ = ta_plus_v2): shed, duck, thm, sm, dm, dlo, lh, sc, wh, wm (see TAPlusPolicy).
All agents bid with ModelBidder(model_ta_final) except greedy (greedy model) and
tap_oracle (InformedOracleBidder for rounds n <= nmax, ModelBidder otherwise).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from judgement.bidding.model import Agent, BidModel, ModelBidder  # noqa: E402
from judgement.policies import GreedyPolicy  # noqa: E402
from judgement.policies.hybrid import HybridPolicy  # noqa: E402
from judgement.policies.ta_plus import TAPlusPolicy  # noqa: E402
from judgement.policies.target_aware import TargetAwarePolicy, TASpitePolicy  # noqa: E402

RES = Path(__file__).parent / "results"
RES.mkdir(exist_ok=True)
P1 = ROOT / "experiments" / "phase1" / "results"
M_TA = BidModel.load(P1 / "model_ta_final.json")
M_GR = BidModel.load(P1 / "model_greedy_final.json")


def _kw(kind):
    return dict(x.split("=") for x in kind.split(":", 1)[1].split(",")) if ":" in kind else {}


def ta_plus_from(kw) -> TAPlusPolicy:
    return TAPlusPolicy(shed=bool(int(kw.get("shed", 1))), surplus_duck=bool(int(kw.get("duck", 0))),
                        theorems=bool(int(kw.get("thm", 1))), shed_margin=float(kw.get("sm", -0.5)),
                        duck_margin=float(kw.get("dm", 1.0)), duck_last_only=bool(int(kw.get("dlo", 0))),
                        lead_high=int(kw.get("lh", 3)), scarce=bool(int(kw.get("sc", 0))),
                        win_high=int(kw.get("wh", 2)), win_margin=float(kw.get("wm", -0.5)))


class SmallRoundOracleBidder:
    """Bid-informed MC oracle (judgement.bidding2.InformedOracleBidder) in rounds
    with n <= nmax, ModelBidder otherwise.  Rollouts: TA agents for the opponents,
    ``hero_player`` (with the TA bid model for later bids, unused for the hero)
    for the bidding seat."""

    name = "oracle_small"

    def __init__(self, hero_player, K=24, span=2, nmax=5, seed=0):
        from judgement.bidding2.oracle import InformedOracleBidder
        from judgement import Rules
        self.mb = ModelBidder(M_TA)
        self.ta_agent = Agent(ModelBidder(M_TA), TargetAwarePolicy())
        self.hero_agent = Agent(ModelBidder(M_TA), hero_player)
        self.orc = InformedOracleBidder(M_TA, Rules(), [self.ta_agent] * 4, K=K, span=span, informed=True, seed=seed)
        self.nmax = nmax

    def bid(self, info, legal, rng):
        if info.n_cards > self.nmax:
            return self.mb.bid(info, legal, rng)
        ags = [self.ta_agent] * info.n_players
        ags[info.seat] = self.hero_agent
        self.orc.agents = ags
        return self.orc.bid(info, legal, rng)


def make_agent(kind: str, seed: int = 0) -> Agent:
    kw = _kw(kind)
    base = kind.split(":", 1)[0]
    if base == "greedy":
        return Agent(ModelBidder(M_GR), GreedyPolicy(), name="greedy")
    if base == "ta":
        return Agent(ModelBidder(M_TA), TargetAwarePolicy(), name="ta")
    if base == "ta_spite":
        return Agent(ModelBidder(M_TA), TASpitePolicy(), name="ta_spite")
    if base == "ta_plus":
        return Agent(ModelBidder(M_TA), ta_plus_from(kw), name=kind)
    if base == "hybrid":
        p = HybridPolicy(n_samples=int(kw.get("k", 16)), pimc_cards=int(kw.get("c", 5)),
                         pimc_in_nt=not bool(int(kw.get("nt", 0))), base=ta_plus_from({}))
        return Agent(ModelBidder(M_TA), p, name=kind)
    if base == "tap_oracle":
        pl = ta_plus_from({})
        b = SmallRoundOracleBidder(pl, K=int(kw.get("K", 24)), span=int(kw.get("span", 2)),
                                   nmax=int(kw.get("nmax", 5)), seed=seed)
        return Agent(b, pl, name=kind)
    if base == "hyb_oracle":
        # Hybrid play + bid-informed oracle bidder; the oracle's rollouts use TA+ for the hero (PIMC
        # inside rollouts would be far too slow), so the bidder slightly mis-models its own play.
        pl = HybridPolicy(n_samples=int(kw.get("k", 16)), pimc_cards=int(kw.get("c", 5)), base=ta_plus_from({}))
        b = SmallRoundOracleBidder(ta_plus_from({}), K=int(kw.get("K", 24)), span=int(kw.get("span", 2)),
                                   nmax=int(kw.get("nmax", 13)), seed=seed)
        return Agent(b, pl, name=kind)
    raise ValueError(kind)


def mean_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else float("nan"), float("nan"), n
    return float(x.mean()), 1.96 * float(x.std(ddof=1)) / np.sqrt(n), n


def dump(obj, name):
    (RES / name).write_text(json.dumps(obj, indent=1))
