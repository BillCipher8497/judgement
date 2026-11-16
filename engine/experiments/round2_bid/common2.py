"""Shared helpers for Phase 3 (round2_bid) experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from judgement.bidding.model import Agent, BidModel, ModelBidder  # noqa: E402
from judgement.policies.target_aware import TargetAwarePolicy  # noqa: E402

RES = Path(__file__).parent / "results"
RES.mkdir(exist_ok=True)
MODEL_PATH = ROOT / "experiments/phase1/results/model_ta_final.json"
GAMES_TA = ROOT / "experiments/phase1/results/games_ta.npz"


def load_model():
    return BidModel.load(MODEL_PATH)


def table_agent(model):
    return Agent(ModelBidder(model), TargetAwarePolicy(), name="model+ta")


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan"), len(x)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), len(x)


def fmt(x, d=2):
    m, h, n = ci(x)
    return f"{m:+.{d}f} ± {h:.{d}f} (n={n})"


def fmtu(x, d=2):
    m, h, n = ci(x)
    return f"{m:.{d}f} ± {h:.{d}f} (n={n})"


def dump(obj, name):
    with open(RES / name, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
