"""PIMC spoiler with a preregistered change only to the available cards."""
from judgement.policies.spoiler import SpoilerPolicy
from judgement.policies.target_aware import TrickModel,_bits,card_value
from judgement.sampling import HandSampler
from judgement.solver.core import Paranoid
from judgement.state import state_from_info

class RevealedHeroView:
    """Remove the revealed hero cards from the unknown allocation, then restore."""
    def __init__(self,info,target,hand):self.info=info;self.target=target;self.revealed=hand
    def __getattr__(self,key):return getattr(self.info,key)
    @property
    def hand(self):return self.info.hand|self.revealed
    def cards_left(self,p):return 0 if p==self.target else self.info.cards_left(p)

def world_sampler(info,target,knowledge):
    if knowledge=='own':return HandSampler(info).sample
    if knowledge=='hero':
        hand=info._s.hands[target]  # Authorized revelation of this hand only.
        sampler=HandSampler(RevealedHeroView(info,target,hand))
        def sample(rng):
            hidden=sampler.sample(rng);hidden[target]=hand;return hidden
        return sample
    if knowledge=='all':
        known={p:h for p,h in enumerate(info._s.hands) if p!=info.seat}
        return lambda rng:known.copy()
    raise ValueError(knowledge)

class InformationSpoiler(SpoilerPolicy):
    """Same objective, K, solver, fallback and tie-breaking; changed sampler only."""
    def __init__(self,target,knowledge,n_samples=16):
        super().__init__(target);self.knowledge=knowledge;self.n_samples=n_samples
    def play(self,info,legal_mask,rng):
        if legal_mask&(legal_mask-1)==0:return legal_mask.bit_length()-1
        m=self.mode(info)
        if m==0 or info.seat==self.target:return self.base.play(info,legal_mask,rng)
        cards=_bits(legal_mask);tot=dict.fromkeys(cards,0.)
        sample=world_sampler(info,self.target,self.knowledge)
        for _ in range(self.n_samples):
            st=state_from_info(info,sample(rng))
            par=Paranoid(st.rules,st.trump,st.bids,self.target,st.N)
            for c in cards:
                child=st.copy();child.apply_card(c);tot[c]+=par.value(child)
        tm=TrickModel(info,True)
        def key(c):
            p=self.p_target_wins(info,tm,c);v=card_value(c,info.trump)
            tie=(round(-p,9),-v,-c) if m>0 else (round(p,9),v,c)
            return (round(tot[c],6),)+tie
        return min(cards,key=key)
