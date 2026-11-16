"""Agent factory for round3_play: experiments/final kinds plus
  hybrid_bi[:k=16,c=5,eps=0.02,tau=0,unif=0,m=512,ess=1,val=0]   Hybrid with bid-informed PIMC worlds
  hybrid_bi_orc[:...,K=24,nmax=13]                                  same + bid-informed oracle bidder
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "play"))
from common import M_TA, SmallRoundOracleBidder, _kw, ta_plus_from  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("final_agents", ROOT / "experiments" / "final" / "agents.py")
_fa = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_fa)
final_make_agent = _fa.make_agent
from judgement import Rules  # noqa: E402
from judgement.bidding.model import Agent, ModelBidder  # noqa: E402
from judgement.bidding2.sampler import BidLikelihood  # noqa: E402
from judgement.policies.hybrid_bi import HybridBIPolicy  # noqa: E402


def make_bi(kw, validate=False):
    lik = BidLikelihood(M_TA, Rules(), tau=float(kw.get("tau", 0)), eps=float(kw.get("eps", 0.02)))
    return HybridBIPolicy(lik, n_samples=int(kw.get("k", 16)), pimc_cards=int(kw.get("c", 5)), base=ta_plus_from({}),
                          informed=not bool(int(kw.get("unif", 0))), m_max=int(kw.get("m", 512)),
                          ess_target=float(kw.get("ess", 1)), validate=validate or bool(int(kw.get("val", 0))))


def make_agent(kind: str, seed: int = 0):
    kw = _kw(kind)
    base = kind.split(":", 1)[0]
    if base == "hybrid_bi":
        return Agent(ModelBidder(M_TA), make_bi(kw), name=kind)
    if base == "hybrid_bi_orc":
        b = SmallRoundOracleBidder(ta_plus_from({}), K=int(kw.get("K", 24)), span=int(kw.get("span", 2)),
                                   nmax=int(kw.get("nmax", 13)), seed=seed)
        return Agent(b, make_bi(kw), name=kind)
    return final_make_agent(kind, seed=seed)
