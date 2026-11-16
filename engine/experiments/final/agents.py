"""Agent factory for experiments/final: experiments/play kinds plus
  ta_plus3[:ll=1|2|0]          TA+ v2 + lead-highest-sure-loser at target (judgement.policies.ta_plus3)
  hybrid3[:k=16,c=5]           Hybrid with TA+3 as the early-trick base
  hyb3_oracle[:k=,c=,K=,nmax=] Hybrid3 play + bid-informed oracle bidder (rollouts: TA opponents, TA+3 hero)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "play"))
from common import M_TA, SmallRoundOracleBidder, _kw, make_agent as play_make_agent  # noqa: E402
from judgement.bidding.model import Agent, ModelBidder  # noqa: E402
from judgement.policies.hybrid import HybridPolicy  # noqa: E402
from judgement.policies.ta_plus3 import ta_plus3  # noqa: E402


def make_agent(kind: str, seed: int = 0) -> Agent:
    kw = _kw(kind)
    base = kind.split(":", 1)[0]
    if base == "ta_plus3":
        return Agent(ModelBidder(M_TA), ta_plus3(lead_loser=int(kw.get("ll", 1))), name=kind)
    if base == "hybrid3":
        p = HybridPolicy(n_samples=int(kw.get("k", 16)), pimc_cards=int(kw.get("c", 5)), base=ta_plus3())
        return Agent(ModelBidder(M_TA), p, name=kind)
    if base == "hyb3_oracle":
        p = HybridPolicy(n_samples=int(kw.get("k", 16)), pimc_cards=int(kw.get("c", 5)), base=ta_plus3())
        b = SmallRoundOracleBidder(ta_plus3(), K=int(kw.get("K", 24)), span=int(kw.get("span", 2)),
                                   nmax=int(kw.get("nmax", 13)), seed=seed)
        return Agent(b, p, name=kind)
    return play_make_agent(kind, seed=seed)
