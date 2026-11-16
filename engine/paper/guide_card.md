# Judgement — table card

**Four players · exact bid: 10 + bid · miss: 0 · dealer bids last.**
Follow suit when able. Remember **trumps played**, **A/K/Q played**, and **who is void in each suit**.

## Bid: count points, then divide by two

| Trump round | No-trump round |
|---|---|
| Every trump: **2 points**; trump A, K, Q: **1 extra each**. | Every ace: **2 points**. Kings count nothing. |
| Every side-suit ace or king: **1 point**. | |
| Subtract **1 point per 4 cards dealt**: 1–3 cards: 0; 4–7: 1; 8–11: 2; 12–13: 3. | |

**Divide by 2, rounded down.**

**Earlier bids:** compare their total with their fair share (each earlier bidder: cards dealt ÷ 4). More than 2 above: bid 1 less; more than 6 above: 2 less. At least 2 below: bid 1 more; at least 6 below: 2 more.

**Dealer hook:** your bid cannot make all bids sum to the cards dealt. If your counted bid is forbidden, go **up 1 for odd points**, otherwise **down 1**; at zero, go up.

## Play: use the first applicable rule

**NEED** = bid − tricks already taken. **BOSS** = no unplayed card of its suit can beat it (track higher A/K/Q); once no trumps remain outside your hand, every trump is a boss. **SURE WINNER** = beats the current winner and you play last, or is a boss that cannot be ruffed. **SURE TRICKS** = bosses in your hand.

1. **Bid already impossible:** play your lowest card.
2. **NEED = 0:** following, play your **highest losing card**. If all win, play highest when last, otherwise lowest. Leading, play your lowest non-trump.
3. **Need every remaining trick:** following, use your cheapest sure winner; if none, your highest card. Leading, use your highest boss, trumps first.
4. **Following, can win:** first use a sure winner; if none, a boss that beats the current winner. Choose **highest when NEED = 1**, otherwise cheapest.
5. **Following, void in the led suit:** ruff with your lowest trump that beats the current winner.
6. **Following, still not winning:** if your sure tricks, excluding this card, cover NEED, throw your **highest card**; otherwise your lowest non-trump.
7. **Leading with a boss:** while trumps are out, lead your top trump boss; otherwise a side boss—highest when NEED = 1, otherwise cheapest.
8. **Leading, no boss:** lead your highest non-trump.

*Condensed from `experiments/guide/GUIDE.txt`; bot-checked guide, not human-playtested.*
