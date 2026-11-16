# Cross-validation Report

Method: drive pyspiel `oh_hell` with identical deal/dealer/trump/bids/card actions (card mapping built from OpenSpiel card names); compare legal bid sets, legal card sets, trick winners; naive reference stepped in parallel.

Results (0 failures in all runs):
- N=4, 20,000 deals: 80,000 bid-set, 521,824 legal-card-set, 130,456 trick-winner checks.
- N=3,5,6,7: 5,000 deals each. N=4 with off_bid_penalty: 10,000 deals.
- Planted bug (wrong trump suit) fails 188/200 deals — harness catches real errors.

Rule differences (documented, not forced):
- Trump: OpenSpiel uses a turned-up card's suit (card out of play); no NT rounds; max hand at N=4 is 12 not 13. NT and 13-card rounds are verified against the naive reference only.
- Hook rule: identical.
- Scoring: exact = 10+bid in both; OpenSpiel miss = +tricks (default) or −|b−t| (off_bid_penalty), expressed as callables.
- OpenSpiel plays a single round with dealer/num-tricks as chance nodes; deals card k to player k%N.

Re-run: python -m crossval.openspiel_check --deals 20000 --players 4 --naive
