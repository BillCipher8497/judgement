"""Phase 4 Q2/Q3 helpers: BT rating recovers a known order; ladder configs are well-formed."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "phase4"))


def test_bt_recovers_order():
    import q3_mixed
    rng = np.random.default_rng(0)
    skill = np.array([0.0, 5.0, 10.0, 15.0])
    tot = skill + rng.normal(0, 20, (3000, 4))
    elo = q3_mixed.bt(tot)
    assert elo[1] == 0.0
    assert np.all(np.diff(elo) > 0)


def test_ladder_configs():
    import q2_ladder
    tags = [c[0] for c in q2_ladder.CONFIGS]
    assert len(set(tags)) == len(tags)
    pools = {c[1] for c in q2_ladder.CONFIGS}
    for p in pools:   # every pool has its all-pool reference table
        assert any(c[1] == p and c[2] == p for c in q2_ladder.CONFIGS)
