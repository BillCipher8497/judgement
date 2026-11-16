"""Agent factory for round5_bots: round4 kinds plus
  hybrid_rb[:k=16,c=5,eps=0.02,pf=0,learn=1,u=0.5,essmin=4,m=512]  Hybrid-RB (online bidder-type posterior);
      pf=1 -> persistent particle pool (B2).  u = prior mass on 'unknown' (rest split evenly over the 5 types).
  hybrid_bipf[:k,c]  Hybrid-BI likelihood (posterior pinned to 'model', eps .02, no tempering) + particle pool.
Pools: POOLS[name](g) -> list of 3 opponent kinds for game g (seats after the challenger, in seat order).
"""
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "play"))
from common import M_TA, _kw, ta_plus_from  # noqa: E402
_spec = importlib.util.spec_from_file_location("r4_agents", ROOT / "experiments" / "round4" / "agents.py")
_r4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_r4)
from judgement import Rules  # noqa: E402
from judgement.bidding.model import Agent, ModelBidder  # noqa: E402
from judgement.bidding2.sampler import BidLikelihood  # noqa: E402
from judgement.policies.hybrid_rb import HybridRBPolicy, TYPES  # noqa: E402

HYB, NOISY, HON = "hybrid:k=16,c=5", "hyb_noisy:k=16,c=5,p=0.3", "hyb_honour:k=16,c=5"
TRUE_TYPE = {HYB: "model", NOISY: "noisy0.25", HON: "honour"}   # nearest family member (true noisy p = 0.3)
_MIX = [HYB, NOISY, HON]
POOLS = {
    "hyb": lambda g: [HYB] * 3,
    "noisy": lambda g: [NOISY] * 3,
    "honour": lambda g: [HON] * 3,
    "mix": lambda g: [_MIX[(g // 4 + i) % 3] for i in range(3)],   # each type in each relative seat
}


def make_rb(kw, validate=False):
    lik = BidLikelihood(M_TA, Rules(), eps=float(kw.get("eps", 0.02)))
    u = float(kw.get("u", 0.5))
    prior = [(1 - u) / (len(TYPES) - 1)] * (len(TYPES) - 1) + [u]
    fix = kw.get("fix")
    if fix is not None:          # posterior pinned to one type
        prior = [1.0 if t == fix else 1e-12 for t in TYPES]
    return HybridRBPolicy(lik, n_samples=int(kw.get("k", 16)), pimc_cards=int(kw.get("c", 5)), base=ta_plus_from({}),
                          prior=prior, eps=float(kw.get("eps", 0.02)), ess_min=float(kw.get("essmin", 4)),
                          m_max=int(kw.get("m", 512)), particles=bool(int(kw.get("pf", 0))),
                          learn=bool(int(kw.get("learn", 1))) and fix is None, validate=validate)


def make_agent(kind: str, seed: int = 0):
    kw = _kw(kind)
    base = kind.split(":", 1)[0]
    if base == "hybrid_rb":
        return Agent(ModelBidder(M_TA), make_rb(kw), name=kind)
    if base == "hybrid_bipf":
        kw = dict(kw, fix="model", essmin=0, pf=1)
        return Agent(ModelBidder(M_TA), make_rb(kw), name=kind)
    return _r4.make_agent(kind, seed=seed)
