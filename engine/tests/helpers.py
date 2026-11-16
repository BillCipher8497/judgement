from judgement.cards import parse_hand
from judgement.rules import Rules
from judgement.state import RoundState


def make_state(hands: list[str], trump: int, dealer: int = 0, rules: Rules | None = None,
               bids: list[int] | None = None) -> RoundState:
    rules = rules or Rules(n_players=len(hands))
    masks = [parse_hand(h) for h in hands]
    st = RoundState(rules, masks[0].bit_count(), trump, dealer, masks)
    for b in bids if bids is not None else []:
        st.apply_bid(b)
    return st


def bid_through(st: RoundState) -> RoundState:
    """Apply first legal bid for everyone."""
    while st.phase == 0:
        st.apply_bid(st.legal_bids()[0])
    return st
