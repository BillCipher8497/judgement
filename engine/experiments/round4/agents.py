"""Agent factory for round4: round3_play kinds plus
  hyb_noisy[:k=16,c=5,p=0.3]  Hybrid play, ModelBidder bid perturbed +-1 w.p. p (hook-legal; own rng)
  hyb_honour[:k=16,c=5]       Hybrid play, crude honour-count bid (GreedyPolicy.bid)
  hybrid_bi_c7 etc. are just hybrid_bi:k=..,c=..,eps=..,tau=..
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "play"))
from common import M_TA, _kw, ta_plus_from  # noqa: E402
_spec = importlib.util.spec_from_file_location("r3_agents", ROOT / "experiments" / "round3_play" / "agents.py")
_r3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_r3)
from judgement.bidding.model import Agent, ModelBidder  # noqa: E402
from judgement.policies.greedy import GreedyPolicy  # noqa: E402
from judgement.policies.hybrid import HybridPolicy  # noqa: E402


class NoisyBidder:
    """ModelBidder bid b; w.p. p replaced by b+s (s=+-1 at random); if illegal (range/hook) try b-s, else keep b.
    Uses its own rng (seeded from the agent seed) so the game rng stream is untouched."""
    name = "noisy_model"

    def __init__(self, model, p=0.3, seed=0):
        self.mb, self.p, self.rng = ModelBidder(model), float(p), np.random.default_rng(seed)

    def bid(self, info, legal, rng):
        b = self.mb.bid(info, legal, rng)
        u, s = self.rng.random(), 1 if self.rng.random() < 0.5 else -1
        if u >= self.p:
            return b
        for c in (b + s, b - s):
            if c in legal:
                return c
        return b


def _hyb(kw):
    return HybridPolicy(n_samples=int(kw.get("k", 16)), pimc_cards=int(kw.get("c", 5)), base=ta_plus_from({}))


def make_agent(kind: str, seed: int = 0):
    kw = _kw(kind)
    base = kind.split(":", 1)[0]
    if base == "hyb_noisy":
        return Agent(NoisyBidder(M_TA, float(kw.get("p", 0.3)), seed=seed), _hyb(kw), name=kind)
    if base == "hyb_honour":
        return Agent(GreedyPolicy(), _hyb(kw), name=kind)
    return _r3.make_agent(kind, seed=seed)
