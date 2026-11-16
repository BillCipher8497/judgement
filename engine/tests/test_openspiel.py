"""Quick OpenSpiel cross-check (the full >=10k run is crossval/openspiel_check.py)."""
import pytest

pytest.importorskip("pyspiel")

from crossval import openspiel_check as oc


@pytest.mark.parametrize("N", range(3, 8))
@pytest.mark.parametrize("off_bid", [False, True])
def test_openspiel_agrees(N, off_bid):
    s = oc.run(300, N, seed=N, off_bid_penalty=off_bid, naive=True)
    assert not s.get("failures"), s.get("failures")[:3]
    assert s["deals"] == 300


def test_crossval_detects_bugs(monkeypatch):
    """Sanity: a deliberately wrong trump suit must be caught."""
    from judgement.state import RoundState
    orig = RoundState.__init__

    def bad(self, rules, n, trump, dealer, hands):
        orig(self, rules, n, (trump + 1) % 4, dealer, hands)
    monkeypatch.setattr(RoundState, "__init__", bad)
    s = oc.run(100, 4, seed=0)
    assert len(s.get("failures", [])) > 50
