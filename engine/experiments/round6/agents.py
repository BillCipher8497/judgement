"""Agent factory for round6: round5 kinds plus
  hybrid_rb2[:k=16,c=5,eps=0.02,learn=1,...]  Hybrid-RB2 (continuous per-opponent bid model + RB types + unknown)
  hyb_guide[:k=16,c=5]                        'guide-bidding Hybrid': HumanBidder (human_guide.py point count) + Hybrid play
Pools: POOLS[name](g) -> 3 opponent kinds (seats after the challenger).  hyb/mix as round5; guide = 3x hyb_guide;
g2 = 3x hybrid_bi:k=32,c=6 (round4 G2 pool)."""
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "play"))
from common import M_TA, _kw, ta_plus_from  # noqa: E402
_spec = importlib.util.spec_from_file_location("r5_agents", ROOT / "experiments" / "round5_bots" / "agents.py")
_r5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_r5)
from judgement import Rules  # noqa: E402
from judgement.bidding.model import Agent, ModelBidder  # noqa: E402
from judgement.bidding2.sampler import BidLikelihood  # noqa: E402
from judgement.policies.human_guide import HumanBidder  # noqa: E402
from judgement.policies.hybrid import HybridPolicy  # noqa: E402

GUIDE = "hyb_guide:k=16,c=5"
G2 = "hybrid_bi:k=32,c=6"
POOLS = dict(_r5.POOLS)
POOLS["guide"] = lambda g: [GUIDE] * 3
POOLS["g2"] = lambda g: [G2] * 3


def make_rb2(kw, validate=False):
    from judgement.policies.hybrid_rb2 import HybridRB2Policy
    lik = BidLikelihood(M_TA, Rules(), eps=float(kw.get("eps", 0.02)))
    return HybridRB2Policy(lik, n_samples=int(kw.get("k", 16)), pimc_cards=int(kw.get("c", 5)), base=ta_plus_from({}),
                           eps=float(kw.get("eps", 0.02)), cont_mass=float(kw.get("cm", 0.3)),
                           u=float(kw.get("u", 0.3)), ess_min=float(kw.get("essmin", 4)), m_max=int(kw.get("m", 512)),
                           learn=bool(int(kw.get("learn", 1))), validate=validate)


def make_agent(kind: str, seed: int = 0):
    kw = _kw(kind)
    base = kind.split(":", 1)[0]
    if base == "hybrid_rb2":
        return Agent(ModelBidder(M_TA), make_rb2(kw), name=kind)
    if base == "hyb_guide":
        return Agent(HumanBidder(), HybridPolicy(n_samples=int(kw.get("k", 16)), pimc_cards=int(kw.get("c", 5)),
                                                 base=ta_plus_from({})), name=kind)
    return _r5.make_agent(kind, seed=seed)
