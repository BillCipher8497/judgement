"""Pre-data synthetic gradients, decoder algebra and detached feature fixtures."""
from common import *
from network import init,gradient,targets,scores
from observables import restore,encode,active
from reference import move_scores,prefix
def main():
    # Diagnostic-only fixed fixture RNG42; no reserved experiment stream drawn.
    rng=np.random.default_rng(42);x=rng.normal(0,.1,(3,683));base=np.array([[2.,0.],[0.,2.],[2.,0.]]);q=rng.integers(0,2,(3,16,2));weights=np.array([10.,12.,13.]);maximum=0.
    for arm in range(8):
        # Use fixture initialization, not an unregistered experiment fit root.
        from learning import initialize
        p=initialize([683,2] if arm in (5,6) else [683,4,3,2],42);target=targets(q,arm);value,grads=gradient(x,base,target,weights,p,arm)
        locations=[(0,(3,1)),(len(p)-2,(1,0)),(len(p)-1,(0,))]
        if len(p)>4:locations.append((2,(1,1)))
        for layer,index in locations:
            orig=p[layer][index];eps=1e-6;p[layer][index]=orig+eps;hi=gradient(x,base,target,weights,p,arm)[0];p[layer][index]=orig-eps;lo=gradient(x,base,target,weights,p,arm)[0];p[layer][index]=orig;error=abs((hi-lo)/(2*eps)-grads[layer][index]);maximum=max(maximum,error);assert error<1e-6,(arm,layer,error)
        np.testing.assert_array_equal(scores(x,base,p,arm,0).argmax(1),base.argmax(1))
    # Exhaust binary-world counts, including all-tied and mixed tie cases.
    for both in range(17):
        for a in range(17-both):
            for b in range(17-both-a):
                raw=np.array([[1,1]]*both+[[1,0]]*a+[[0,1]]*b+[[0,0]]*(16-both-a-b))[None,:,:]
                membership=targets(raw,2)[0];mean=raw.mean(1)[0];assert np.sign(membership[0]-membership[1])==np.sign(mean[0]-mean[1])
    o={'seat':1,'dealer':0,'n_cards':2,'trump':-1,'hand':(1<<2)|(1<<17),'bids':[[1,1],[2,0],[3,0],[0,0]],'tricks_won':[0]*4,'current_trick':[],'history':[],'trick_winners':[],'voids':[0]*4,'cards_left_counts':[2]*4}
    info=restore(o);assert active(info,info.hand);xx,bb,cards=encode(info,info.hand,2);assert xx.shape==(683,) and cards==[2,17] and not hasattr(info,'_s')
    hands=[(1<<0)|(1<<15),o['hand'],(1<<3)|(1<<18),(1<<4)|(1<<19)];assert prefix(o,hands);values=move_scores(o,hands);assert set(values)==set(cards)
    from judgement.state import RoundState,InfoState
    from judgement.rules import Rules
    from judgement.solver.core import MaxN
    from observables import observation
    # Independent solver checks include real card IDs, stock, NT/trump and
    # mid-trick positions. Diagnostic RNG42 only, not reserved experiment roots.
    solver_checks=0
    for j in range(96):
        cards_fixture=rng.choice(52,8,replace=False);hh=[sum(1<<int(c) for c in cards_fixture[2*p:2*p+2]) for p in range(4)];st=RoundState(Rules(),2,(-1,0,1,2,3)[j%5],j%4,hh)
        for _ in range(4):st.apply_bid(1)
        for _ in range(j%4):st.apply_card(min(c for c in range(52) if st.legal_cards()>>c&1))
        who=st.to_act;oo=observation(InfoState(st,who));assert prefix(oo,st.hands);ref=move_scores(oo,st.hands);got=MaxN(st.rules,st.trump,st.bids,4).move_values(st)
        assert all(ref[c]==got[c][who] for c in ref);solver_checks+=1
    from data import context
    from reference import generated_context
    class FixtureRegistry:
        def add(self,address):assert address[0]==42
    for fixture_g in range(4):
        row=context(0,fixture_g,FixtureRegistry(),time.time()+60,base_seed=42);oo=row['observation'];assert generated_context([42,DOMAIN,0,fixture_g,row['attempt'],3],row['round_index'],oo)==row['original_hand']
        for k,w in enumerate(row['worlds']):
            ref=move_scores(oo,w);np.testing.assert_array_equal([ref[c]/row['hit'] for c in row['cards']],row['q'][k])
    from evaluate import Student
    from observables import agents,MODEL,baseline
    from judgement.bidding.model import Agent,ModelBidder
    from judgement.engine import play_game
    from reference import full_game
    params=initialize([683,4,2],42);params[-2][:]=0;params[-1][:]=0;fixture_models=[{'params':params,'alpha':1.} for _ in range(5)]
    for arm in (2,7):
        hero=arm%4;policies=agents();policies[hero]=Agent(ModelBidder(MODEL),Student(fixture_models,arm));game=play_game(Rules(),policies,seed=42,record_trace=False)
        def fixture_choice(info,legal,rng):
            if arm==7 and active(info,legal):return min(c for c in range(52) if legal>>c&1)
            return baseline().play(info,legal,rng)
        manual=full_game(42,hero,fixture_choice);assert game.totals==manual['totals'];assert [r.scores for r in game.rounds]==[r['scores'] for r in manual['rounds']]
    save(H/'self_check.json',{'status':'pass','gradient_max_error':maximum,'binary_ordering_exhaustive_16world_multisets':True,'feature_length':683,'detached_observation_fixture':True,'manual_prefix_and_solver_fixture':True,'independent_solver_fixtures':solver_checks,'fixed_seed42_end_to_end_contexts':4,'manual_vs_engine_full_game_fixtures':2,'scope':'Synthetic diagnostic only; fixed nonreserved42fixtures, no scientific seed consumed or experiment dataset created','utc':utc()});print('N3 synthetic checks PASS')
if __name__=='__main__':main()
