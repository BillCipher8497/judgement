"""Private-observable conditional contexts and sampled privileged teacher labels."""
from common import *
from observables import RULES,MODEL,agents,baseline,active,encode,observation,detach
from judgement.bidding.model import Agent,ModelBidder
from judgement.engine import make_rngs,deal
from judgement.state import RoundState,InfoState,BID,PLAY,state_from_info
from judgement.sampling import HandSampler
from judgement.solver.core import MaxN
ROUND_INDICES=[r for r,n in enumerate(RULES.ladder()) if n>1]
def partition(hand):
    bucket=int.from_bytes(hashlib.sha256(('N3-hand-v1:'+str(hand)).encode()).digest()[:8],'big')%10
    return 0 if bucket<6 else 1 if bucket<8 else 2
def context(stage,g,registry,deadline,base_seed=BASE):
    hero=g%4;ri=ROUND_INDICES[(g//4)%24];n=RULES.ladder()[ri];wanted=0 if stage in (0,1) else stage-1
    for attempt in range(1000):
        check_time(deadline);address=[base_seed,DOMAIN,stage,g,attempt,3];registry.add(address)
        dr,pr=make_rngs(address,4);hands=deal(RULES,n,dr);original=hands[hero]
        if partition(original)!=wanted:continue
        st=RoundState(RULES,n,RULES.trump_for_round(ri),RULES.dealer_for_round(ri),hands)
        policies=agents();policies[hero]=Agent(ModelBidder(MODEL),baseline())
        while st.phase==BID:
            p=st.to_act;st.apply_bid(policies[p].bid(InfoState(st,p),st.legal_bids(),pr[p]))
        while st.phase==PLAY:
            p=st.to_act;info=InfoState(st,p);legal=st.legal_cards()
            if p==hero and active(info,legal):
                detached=detach(info);base=baseline().play(detached,legal,pr[p]);x,b,cards=encode(detached,legal,base)
                assert max(detached.cards_left(q) for q in range(4))<=2
                world_address=[base_seed,DOMAIN,stage,g,attempt,5];registry.add(world_address);rng=np.random.default_rng(np.random.SeedSequence(world_address));sampler=HandSampler(detached)
                q=[];worlds=[]
                for _ in range(16):
                    hidden=sampler.sample(rng);worlds.append([hidden.get(p,detached.hand) for p in range(4)]);full=state_from_info(detached,hidden);mv=MaxN(full.rules,full.trump,full.bids,full.N).move_values(full);hit=10+dict(detached.bids)[hero]
                    scores=[mv[c][hero] for c in cards];assert set(scores)<=set((0,hit));q.append([v/hit for v in scores])
                return {'x':x,'base':b,'q':q,'worlds':worlds,'cards':cards,'hit':hit,'g':g,'attempt':attempt,'stratum':g%96,'original_hand':original,'observation':observation(detached),'round_index':ri}
            st.apply_card(policies[p].play(info,legal,pr[p]))
    raise RuntimeError(f'N3 bounded rejection exhausted stage={stage} g={g}; retain scope failure, do not change acceptance')
def generate(name,registry,freeze,deadline):
    folder=O/'data'/name;folder.mkdir(exist_ok=True,parents=True);N=COUNTS[name]
    for start in range(0,N,64):
        check_time(deadline);verify_sources(freeze);p=folder/f'{start:06d}.npz';meta=p.with_suffix('.json')
        if p.exists() and meta.exists():
            r=json.loads(meta.read_text());assert r['arrays_sha256']==digest(p) and r['source_freeze_sha256']==digest(O/'source_freeze.json');continue
        # A raw orphan is never overwritten; retain it and compare with replay.
        started=time.monotonic();rows=[context(STAGES[name],g,registry,deadline) for g in range(start,min(start+64,N))];elapsed=time.monotonic()-started
        a={k:np.asarray([r[k] for r in rows]) for k in ('x','base','q','worlds','cards','hit','g','attempt','stratum','original_hand','round_index')}
        if p.exists():
            old=np.load(p)
            for k,v in a.items():np.testing.assert_array_equal(old[k],v)
        else:arrays(p,**a)
        save(meta,{'observations':[r['observation'] for r in rows],'arrays_sha256':digest(p),'source_freeze_sha256':digest(O/'source_freeze.json'),'stage':name,'first_g':start,'count':len(rows),'generation_wall_seconds':elapsed})
        status(f'N3data {name} {min(start+64,N)}/{N} rounds; incremental labels/worlds/prefixes saved')
    return load(name)
def load(name):
    folder=O/'data'/name;paths=sorted(folder.glob('*.npz'));chunks=[np.load(p) for p in paths]
    for p in paths:assert json.loads(p.with_suffix('.json').read_text())['arrays_sha256']==digest(p)
    data={k:np.concatenate([v[k] for v in chunks]) for k in chunks[0].files};assert len(data['g'])==COUNTS[name]
    np.testing.assert_array_equal(data['g'],np.arange(COUNTS[name]));assert np.all(data['g']%4==data['stratum']%4)
    return data
