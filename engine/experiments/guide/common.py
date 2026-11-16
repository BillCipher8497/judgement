"""Shared helpers for experiments/guide (Phase 5: human-playable strategy)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from judgement.bidding.model import Agent, BidModel, ModelBidder  # noqa: E402
from judgement.policies import GreedyPolicy  # noqa: E402
from judgement.policies.target_aware import TargetAwarePolicy  # noqa: E402
from judgement.policies.ta_plus3 import ta_plus3  # noqa: E402

RES = Path(__file__).parent / "results"
RES.mkdir(exist_ok=True)
P1 = ROOT / "experiments" / "phase1" / "results"
M_TA = BidModel.load(P1 / "model_ta_final.json")
M_GR = BidModel.load(P1 / "model_greedy_final.json")
STATUS = Path(__file__).parent / "STATUS.txt"


def status(msg: str):
    import os
    os.environ.setdefault("TZ", "America/New_York")
    time.tzset()
    with open(STATUS, "a") as fh:
        fh.write(time.strftime("%H:%M") + " " + msg.rstrip() + "\n")
