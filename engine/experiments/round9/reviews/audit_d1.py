"""Fresh D1 audit: independent legality/scoring, raw estimators and bootstrap.

Does not import the experiment runner or its analyzer. Summary comparisons are
made only after the independently derived audit output has been saved.
"""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROUND=HERE.parent;ENGINE=ROUND.parents[1]
sys.path.insert(0,str(ENGINE))
OUT=HERE/'results';OUT.mkdir(exist_ok=True)
ARMS=['self','own','hero','all','omni'];BASES=[1600000000,1610000000]
REVIEW_SEED=1650000001;REPLAY=[0,137,274,411,548,685,822,959]

def replay_rules(row,arm):
    hands=[int(h) for h in row['initial_hands']];bids=row['fixed_bids']
    assert len(hands)==4 and all(h.bit_count()==5 for h in hands)
    seen=0
    for h in hands:assert not seen&h and 0<=h<1<<52;seen|=h
    prior=0
    for k in range(4):
        p=(row['dealer']+1+k)%4;b=bids[p]
        assert 0<=b<=5 and (k<3 or prior+b!=5);prior+=b
    leader=(row['dealer']+1)%4;trick=[];wins=[0]*4
    history=arm['history'];assert len(history)==20
    for p,c in history:
        assert p==(leader+len(trick))%4 and 0<=c<52 and hands[p]>>c&1
        if trick:
            suit=trick[0][1]//13;follow=hands[p]&(((1<<13)-1)<<(13*suit))
            assert not follow or c//13==suit
        hands[p]^=1<<c;trick.append((p,c))
        if len(trick)==4:
            trump=[(q,d) for q,d in trick if row['trump']>=0 and d//13==row['trump']]
            eligible=trump or [(q,d) for q,d in trick if d//13==trick[0][1]//13]
            leader=max(eligible,key=lambda qc:qc[1]%13)[0];wins[leader]+=1;trick=[]
    assert not any(hands) and not trick
    scores=[10+b if b==w else 0 for b,w in zip(bids,wins)]
    assert arm['bids']==bids and arm['tricks']==wins and arm['scores']==scores
    return scores[0]

all_data={};bootstrap={};manifest={};counts={}
for base in BASES:
    paths=sorted((ROUND/'D1/results').glob(f'{base}_*.json'));assert len(paths)==1000
    raw=[];strata={0:[],-1:[]};positions={}
    for expected_g,path in enumerate(paths):
        payload=path.read_bytes();r=json.loads(payload);g=r['g']
        assert g==expected_g and r['base']==base and r['seed']==[base,5,g]
        assert r['hero']==0 and r['n']==5 and r['trump']==(0 if g%2==0 else -1) and r['dealer']==(g//2)%4
        dr=np.random.default_rng(np.random.SeedSequence([base,5,g]).spawn(5)[0])
        perm=dr.permutation(52)
        expected_hands=[sum(1<<int(c) for c in perm[p*5:(p+1)*5]) for p in range(4)]
        assert r['initial_hands']==expected_hands
        assert set(r['arms'])==set(ARMS) and r['arms']['all']==r['arms']['omni']
        raw.append([replay_rules(r,r['arms'][a]) for a in ARMS]);strata[r['trump']].append(g)
        key=f"trump{r['trump']}/dealer{r['dealer']}";positions[key]=positions.get(key,0)+1
        manifest[path.name]=hashlib.sha256(payload).hexdigest()
    x=np.asarray(raw,float);assert set(positions.values())=={125}
    means=(x[strata[0]].mean(0)+x[strata[-1]].mean(0))/2
    shares=1-means/means[0]
    rng=np.random.Generator(np.random.PCG64DXSM([REVIEW_SEED,base]));draws=[]
    for batch in range(100):
        sampled=np.zeros((100,5))
        for indices in strata.values():
            sample=rng.choice(np.asarray(indices),size=(100,len(indices)),replace=True)
            sampled+=x[sample].sum(1)/(2*len(indices))
        draws.append(sampled)
    draw=np.concatenate(draws);bs=1-draw/draw[:,0,None];primary=bs[:,2]-bs[:,1]
    def interval(point,values):
        return {'point':float(point),'ci95':np.quantile(values,[.025,.975]).tolist(),'n_deals':1000}
    result={'n_deals':1000,'trump_dealer_counts':positions,
            'hero_scores':{a:interval(means[i],draw[:,i]) for i,a in enumerate(ARMS)},
            'surplus_shares':{a:interval(shares[i],bs[:,i]) for i,a in enumerate(ARMS) if a!='self'},
            'primary_revealed_minus_own':interval(shares[2]-shares[1],primary),
            'ratios_to_omniscient':{a:interval(shares[i]/shares[4],bs[:,i]/bs[:,4]) for i,a in enumerate(ARMS) if a not in ['self','omni']}}
    all_data[str(base)]=result;bootstrap[str(base)]=primary
    counts[str(base)]={'legal_rounds_replayed':5000,'complete_deals':1000,'all_omni_identical':1000}

diff=bootstrap[str(BASES[0])]-bootstrap[str(BASES[1])]
compat={'point':all_data[str(BASES[0])]['primary_revealed_minus_own']['point']-all_data[str(BASES[1])]['primary_revealed_minus_own']['point'],
        'ci95':np.quantile(diff,[.025,.975]).tolist(),'n_deals_per_stage':1000,
        'qualification':'No detected incompatibility is not an equivalence result.'}
audit={'method':'Independent card legality/trick/scoring checker and fresh stratified paired-deal bootstrap',
       'review_bootstrap_seed':REVIEW_SEED,'bootstrap_replicates':10000,'results':all_data,
       'discovery_minus_replication_primary':compat,'integrity':counts,'raw_sha256':manifest}
(OUT/'D1_raw_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
np.savez_compressed(OUT/'D1_review_bootstrap.npz',discovery=bootstrap[str(BASES[0])],replication=bootstrap[str(BASES[1])])

# Independently replay a prespecified set of policy trajectories against references.
from judgement import Rules,RoundState
from judgement.engine import run_round
from judgement.policies.ta_plus import ta_plus_v2
from judgement.policies.spoiler import PIMCSpoilerPolicy,OmniscientSpoilerPolicy
spec=importlib.util.spec_from_file_location('d1_review_intervention',ROUND/'D1/intervention.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
replayed=0
for base in BASES:
    for g in REPLAY:
        row=json.loads((ROUND/'D1/results'/f'{base}_{g:04d}.json').read_text())
        for arm in ARMS:
            st=RoundState(Rules(),5,row['trump'],row['dealer'],row['initial_hands'])
            for k in range(4):
                p=(row['dealer']+1+k)%4;st.apply_bid(row['fixed_bids'][p])
            players=[ta_plus_v2() if p==0 or arm=='self' else
                     PIMCSpoilerPolicy(0,16) if arm=='own' else
                     OmniscientSpoilerPolicy(0) if arm in ['all','omni'] else
                     module.InformationSpoiler(0,'hero',16) for p in range(4)]
            children=np.random.SeedSequence([base,5,g]).spawn(5)
            run_round(st,players,[np.random.default_rng(s) for s in children[1:]],check=True)
            assert [list(pc) for pc in st.history]==row['arms'][arm]['history']
            replayed+=1
audit['reference_policy_replays']={'selected_g':REPLAY,'rounds':replayed,'all_histories_identical':True}
(OUT/'D1_raw_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')

# Only now consult the original reported summary for numerical agreement.
old=json.loads((ROUND/'D1/results/summary.json').read_text())
comparisons={}
for base in BASES:
    original=old[str(base)]['primary_hero_minus_own_share'];fresh=all_data[str(base)]['primary_revealed_minus_own']
    assert abs(original['point']-fresh['point'])<1e-12
    endpoint_error=max(abs(a-b) for a,b in zip(original['ci95'],fresh['ci95']))
    assert endpoint_error<.01
    comparisons[str(base)]={'same_point':True,'max_CI_endpoint_difference':endpoint_error}
audit['original_summary_comparison_after_recomputation']=comparisons
(OUT/'D1_raw_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps({'primary':{k:v['primary_revealed_minus_own'] for k,v in all_data.items()},'compatibility':compat,'replays':replayed,'original_comparison':comparisons},indent=2))
