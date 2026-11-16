"""JSON-serialisable traces.  ``replay(trace)`` re-executes every recorded
action through the rules engine (checking legality) and rebuilds the trace,
which must equal the input exactly."""
from __future__ import annotations

import json

from .cards import card_str, cards_of, parse_card, parse_suit, suit_str
from .engine import fractional_wins
from .rules import Rules
from .state import RoundState

TRACE_VERSION = 1


def round_trace(st: RoundState, initial_hands: list[int], round_index: int) -> dict:
    return {
        "round": round_index,
        "n_cards": st.n_cards,
        "trump": suit_str(st.trump),
        "dealer": st.dealer,
        "hands": [[card_str(c) for c in cards_of(h)] for h in initial_hands],
        "bids": [[p, b] for p, b in st.bid_seq],
        "plays": [[p, card_str(c)] for p, c in st.history],
        "trick_winners": list(st.trick_winners),
        "tricks": list(st.tricks_won),
        "scores": st.scores(),
    }


def game_trace(rules: Rules, seed, rounds, totals, wins, policy_names) -> dict:
    return {
        "version": TRACE_VERSION,
        "rules": rules.to_dict(),
        "seed": seed if isinstance(seed, (int, type(None))) else repr(seed),
        "policies": list(policy_names),
        "rounds": [r.trace for r in rounds],
        "totals": list(totals),
        "wins": list(wins),
    }


def replay_round(rules: Rules, rt: dict) -> dict:
    hands = [sum(1 << parse_card(c) for c in h) for h in rt["hands"]]
    st = RoundState(rules, rt["n_cards"], parse_suit(rt["trump"]), rt["dealer"], hands)
    for p, b in rt["bids"]:
        if st.to_act != p:
            raise ValueError(f"bid out of turn: seat {p}, expected {st.to_act}")
        st.apply_bid(b)
    for p, c in rt["plays"]:
        if st.to_act != p:
            raise ValueError(f"play out of turn: seat {p}, expected {st.to_act}")
        st.apply_card(parse_card(c))
    if not st.is_terminal:
        raise ValueError("trace round is incomplete")
    return round_trace(st, hands, rt["round"])


def replay(trace: dict, rules: Rules | None = None) -> dict:
    rules = rules or Rules.from_dict(trace["rules"])
    if "rounds" not in trace:  # a single round trace
        return replay_round(rules, trace)
    rounds = [replay_round(rules, rt) for rt in trace["rounds"]]
    totals = [0.0] * rules.n_players
    for rt in rounds:
        for p, s in enumerate(rt["scores"]):
            totals[p] += s
    return {
        "version": trace["version"],
        "rules": rules.to_dict(),
        "seed": trace["seed"],
        "policies": list(trace["policies"]),
        "rounds": rounds,
        "totals": totals,
        "wins": fractional_wins(totals),
    }


def dumps(trace: dict) -> str:
    return json.dumps(trace, sort_keys=True, separators=(",", ":"))


def loads(s: str) -> dict:
    return json.loads(s)
