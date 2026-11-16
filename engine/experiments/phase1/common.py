"""Shared helpers for Phase 1 experiments."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from judgement.bidding.model import Agent, BidModel, ModelBidder
from judgement.policies import GreedyPolicy
from judgement.policies.target_aware import TargetAwarePolicy, TASpitePolicy

RES = Path(os.environ.get("P1_RES", Path(__file__).parent / "results"))
RES.mkdir(exist_ok=True)


def load_model(tag="ta"):
    return BidModel.load(RES / f"model_{tag}_final.json")


def agents(kind: str, model) -> Agent:
    play = {"ta": TargetAwarePolicy(), "greedy": GreedyPolicy(), "ta_spite": TASpitePolicy()}[kind]
    return Agent(ModelBidder(model), play, name=f"model+{kind}")


def mean_se(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def prop_ci(k, n):
    """Wilson 95% interval."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def fmt(m, se, d=2):
    return f"{m:.{d}f} ± {1.96 * se:.{d}f}"


def dump(obj, name):
    with open(RES / name, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def dist_stats(p):
    p = np.asarray(p, float)
    t = np.arange(len(p))
    m = float((p * t).sum())
    v = float((p * (t - m) ** 2).sum())
    nz = p[p > 0]
    ent = float(-(nz * np.log2(nz)).sum())
    return m, v, ent, int(p.argmax())


def is_bimodal(p, min_peak=0.05, ratio=0.5, min_gap=0.03):
    """Dip criterion: exists k with i < k < j such that P(i), P(j) >= min_peak and
    P(k) <= ratio * min(P(i), P(j)) and min(P(i), P(j)) - P(k) >= min_gap."""
    p = np.asarray(p, float)
    K = len(p)
    if K < 3:
        return False
    left = np.maximum.accumulate(p)
    right = np.maximum.accumulate(p[::-1])[::-1]
    for k in range(1, K - 1):
        lo = min(left[k - 1], right[k + 1])
        if lo >= min_peak and p[k] <= ratio * lo and lo - p[k] >= min_gap:
            return True
    return False
