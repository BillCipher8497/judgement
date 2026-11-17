"""Original toy-TA code with bounded memoization; policy arithmetic is unchanged."""
from functools import lru_cache
from types import FunctionType
from judgement.cfr import heuristics

@lru_cache(maxsize=None)
def initial(g,p,hand):return heuristics._pwin_cached.__wrapped__(g,p,hand,())

@lru_cache(maxsize=50000)
def middle(g,p,hand,hist):return heuristics._pwin_cached.__wrapped__(g,p,hand,hist)

def pwin(g,p,hand,hist):return middle(g,p,hand,hist) if hist else initial(g,p,hand)
def clear_mid():middle.cache_clear()

# Clone the exact existing code object into a separate globals dictionary.
# Nothing in the existing policy module or its global namespace is changed.
namespace=heuristics.ta_card.__globals__.copy();namespace['_pwin_cached']=pwin
ta_card=FunctionType(heuristics.ta_card.__code__,namespace,'ta_card')
