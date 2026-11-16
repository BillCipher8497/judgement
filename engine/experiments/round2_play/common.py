"""Shared helpers for round-2 play experiments."""
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


def make_agent(kind: str) -> Agent:
    if kind == "greedy":
        return Agent(ModelBidder(M_GR), GreedyPolicy(), name="greedy")
    if kind == "ta":
        return Agent(ModelBidder(M_TA), TargetAwarePolicy(), name="ta")
    if kind == "ta_spite":
        return Agent(ModelBidder(M_TA), TASpitePolicy(), name="ta_spite")
    if kind == "ta_plus":
        return Agent(ModelBidder(M_TA), TAPlusPolicy(), name="ta_plus")
    if kind.startswith("ta_plus:"):
        # ta_plus:shed=1,duck=0,thm=1,sm=-0.5,dm=1.0,lh=3,sc=0 (defaults = TAPlusPolicy defaults).
        # NOTE: the screen runs (results/1v3_scr_*) used older defaults duck=1, sm=0.5, lh=0.
        kw = dict(x.split("=") for x in kind.split(":", 1)[1].split(","))
        p = TAPlusPolicy(shed=bool(int(kw.get("shed", 1))), surplus_duck=bool(int(kw.get("duck", 0))),
                         theorems=bool(int(kw.get("thm", 1))), shed_margin=float(kw.get("sm", -0.5)),
                         duck_margin=float(kw.get("dm", 1.0)), duck_last_only=bool(int(kw.get("dlo", 0))),
                         lead_high=int(kw.get("lh", 3)), scarce=bool(int(kw.get("sc", 0))))
        return Agent(ModelBidder(M_TA), p, name=kind)
    if kind.startswith("hybrid"):
        # hybrid or hybrid:k=16,c=5
        kw = dict(x.split("=") for x in kind.split(":", 1)[1].split(",")) if ":" in kind else {}
        p = HybridPolicy(n_samples=int(kw.get("k", 16)), pimc_cards=int(kw.get("c", 5)))
        return Agent(ModelBidder(M_TA), p, name=kind)
    raise ValueError(kind)


def mean_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else float("nan"), float("nan"), n
    return float(x.mean()), 1.96 * float(x.std(ddof=1)) / np.sqrt(n), n


def prop_ci(k, n):
    """Wilson 95% interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def dump(obj, name):
    (RES / name).write_text(json.dumps(obj, indent=1))
