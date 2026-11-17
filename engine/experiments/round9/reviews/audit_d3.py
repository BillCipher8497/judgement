"""Separate D3 review. No experiment evaluator, root bound or BR imports.

Uses source-fingerprinted derived trees; rebuild instructions remain in D3.
Independent grouped recurrences, integer lower bound, counterfactual BR and
manual game replay are compared with reported summaries only after saving.
"""
import gc,hashlib,json,pickle,sys
from fractions import Fraction
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;D3=HERE.parent/'D3';ENGINE=HERE.parents[2]
sys.path[:0]=[str(ENGINE),str(D3)]
from judgement.cfr.heuristics import myopic_bid,ta_card
OUT=HERE/'results';OUT.mkdir(exist_ok=True)
SELECTED=[0,52799,105598,158397,211196,263995,316794,369599]

def groups(t):
    lengths=t.node_len[t.e_par]
    assert np.all(np.diff(lengths)>=0)
    cuts=np.r_[0,np.flatnonzero(np.diff(lengths))+1,t.n_edges]
    out=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        par=t.e_par[a:b];unique,starts=np.unique(par,return_index=True)
        assert np.array_equal(unique,par[starts])
        out.append((int(a),int(b),unique,starts))
    return out

def evaluate(t,sigma,levels):
    v=t.U[:,0].copy()
    for a,b,parents,starts in reversed(levels):
        v[parents]=np.add.reduceat(v[t.e_ch[a:b]]*sigma[t.e_is[a:b],t.e_a[a:b]],starts)
    return v[t.roots].mean()

def private_br(t,fixed,p,levels):
    # Opponent/chance reach excludes the deviating seat's own action factors.
    cf=np.zeros(t.n_nodes);cf[t.roots]=1/t.n_deals
    for a,b,parents,starts in levels:
        factor=np.where(t.e_pl[a:b]==p,1.,fixed[t.e_is[a:b],t.e_a[a:b]])
        cf[t.e_ch[a:b]]=cf[t.e_par[a:b]]*factor
    v=t.U[:,0].copy();policy=fixed.copy()
    for a,b,parents,starts in reversed(levels):
        ids=t.e_is[a:b];act=t.e_a[a:b];child=t.e_ch[a:b];par=t.e_par[a:b]
        own=t.e_pl[a:b]==p
        if np.any(own):
            infos,inv=np.unique(ids[own],return_inverse=True)
            q=np.zeros((len(infos),t.g.A))
            np.add.at(q,(inv,act[own]),cf[par[own]]*v[child[own]])
            chosen=np.where(t.is_valid[infos],q,np.inf).argmin(1)
            policy[infos]=0;policy[infos,chosen]=1
        weights=np.where(own,policy[ids,act],fixed[ids,act])
        v[parents]=np.add.reduceat(v[child]*weights,starts)
    return policy,v[t.roots].mean()

def manual_oracle(g,hands):
    # No tree builder/_play/replay helpers: follow suit, trick winner and hit
    # impossibility are checked here from the cards. Hero policy is frozen.
    bids=[]
    for p in range(4):bids.append(myopic_bid(g,p,hands[p],tuple(bids)))
    def rec(left,trick,leader,wins,hist):
        remaining=len(left[0])+int(any(p==0 for p,c in trick));need=bids[0]-wins[0]
        if need<0 or need>remaining:return 0
        if not any(left) and not trick:return 10+bids[0] if need==0 else 0
        p=(leader+len(trick))%4;legal=left[p]
        if trick:
            follow=[c for c in legal if c//g.R==trick[0][1]//g.R]
            legal=follow or legal
        if p==0 and len(legal)>1:legal=[ta_card(g,p,hands[p],tuple(bids)+hist)]
        scores=[]
        for c in legal:
            nxt=[list(h) for h in left];nxt[p].remove(c);tr=trick+[(p,c)];ld=leader;w=wins.copy()
            if len(tr)==4:
                trump=[pc for pc in tr if g.trump>=0 and pc[1]//g.R==g.trump]
                eligible=trump or [pc for pc in tr if pc[1]//g.R==tr[0][1]//g.R]
                ld=max(eligible,key=lambda pc:pc[1]%g.R)[0];w[ld]+=1;tr=[]
            scores.append(rec(nxt,tr,ld,w,hist+(c,)))
        return min(scores)
    return rec([list(h) for h in hands],[],0,[0]*4,())

report={}
for trump in [0,-1]:
    meta=json.loads((D3/'results'/f'tree_cache_{trump}.json').read_text())
    fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in [D3/'compact_tree.py',D3/'bounded_ta.py',ENGINE/'judgement/cfr/toy.py',ENGINE/'judgement/cfr/heuristics.py'])).hexdigest()
    assert fingerprint==meta['source_sha256']
    with Path(meta['derived_cache']).open('rb') as f:t=pickle.load(f)
    assert (t.g.N,t.g.S,t.g.R,t.g.n,t.g.trump,t.n_deals)==(4,3,4,3,trump,369600)
    assert t.n_edges+len(t.roots)==t.n_nodes
    assert len(np.unique(np.r_[t.roots,t.e_ch]))==t.n_nodes
    assert np.all(t.node_len[t.e_ch]>t.node_len[t.e_par])
    assert np.all(t.node_pl[t.e_par]==t.is_pl[t.e_is])
    assert np.all(t.node_is[t.e_par]==t.e_is) and np.all(t.is_valid[t.e_is,t.e_a])
    levels=groups(t);v=t.U[:,0].astype(np.int64)
    np.testing.assert_array_equal(v,t.U[:,0])
    for a,b,parents,starts in reversed(levels):v[parents]=np.minimum.reduceat(v[t.e_ch[a:b]],starts)
    pi_sum=int(v[t.roots].sum());isroot=np.zeros(t.n_nodes,bool);isroot[t.roots]=True
    mask=isroot[t.e_par];ids=t.e_is[mask];acts=t.e_a[mask]
    infos,inv=np.unique(ids,return_inverse=True);q=np.zeros((len(infos),t.g.A),np.int64)
    np.add.at(q,(inv,acts),v[t.e_ch[mask]])
    terminal=t.roots[t.node_pl[t.roots]<0]
    j1_sum=int(v[terminal].sum())+int(np.where(t.is_valid[infos],q,np.iinfo(np.int64).max).min(1).sum())
    assert j1_sum>pi_sum
    deals=t.g.deals();assert len(deals)==369600 and len(set(deals))==369600
    checks=[]
    for i in SELECTED:
        manual=manual_oracle(t.g,deals[i]);assert manual==v[t.roots[i]]
        checks.append({'deal_index':i,'PI_score':manual,'hands':deals[i]})
    del deals,mask,ids,acts,infos,inv,q,isroot
    with np.load(D3/'results'/f'trump_{trump}_profile.npz') as z:fixed=z['fixed'].copy();main=z['sigma'].copy()
    profiles={'baseline':fixed,'main':main}
    if trump==0:
        with np.load(D3/'results/trump_0_optimized_profile.npz') as z:profiles['optimized']=z['sigma'].copy()
    scores={}
    for name,sig in profiles.items():
        assert sig.shape==t.is_valid.shape and np.isfinite(sig).all() and np.all(sig>=0) and np.all(sig[~t.is_valid]==0)
        np.testing.assert_allclose(sig.sum(1),1,atol=1e-10,rtol=0)
        scores[name]=float(evaluate(t,sig,levels))
    heuristic=fixed.copy();brvalues={}
    for p in [1,2,3]:
        br,val=private_br(t,fixed,p,levels);brvalues[str(p)]=float(val)
        heuristic[t.is_pl==p]=br[t.is_pl==p];del br
    scores['independent_BR']=float(evaluate(t,heuristic,levels))
    achieved=min(value for key,value in scores.items() if key!='baseline')
    lower=float(Fraction(j1_sum,t.n_deals));pi=float(Fraction(pi_sum,t.n_deals))
    assert pi<lower<=achieved+1e-8
    report[str(trump)]={'n_exact_deals':t.n_deals,'source_sha256':fingerprint,
        'PI_fraction':str(Fraction(pi_sum,t.n_deals)),'J1_fraction':str(Fraction(j1_sum,t.n_deals)),
        'strict_gap_fraction':str(Fraction(j1_sum-pi_sum,t.n_deals)),
        'scores':scores,'private_BR_hero_values':brvalues,'manual_game_PI_checks':checks,
        'hidden_optimum_bounds':[lower,achieved],
        'uncertainty':'Complete chance population, no sampling CI; exact integer lower bounds, floating feasible expectation. Proof pending external audit.'}
    (OUT/'D3_raw_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({str(trump):{k:v for k,v in report[str(trump)].items() if k!='manual_game_PI_checks'}}),flush=True)
    del t,profiles,fixed,main,heuristic,v,levels,sig;gc.collect()

# Comparisons are deliberately deferred until independent results are saved.
old=json.loads((D3/'results/summary_final.json').read_text())
for key,r in report.items():
    x=old[key];assert r['PI_fraction']==x['PI_fraction'] and r['J1_fraction']==x['root_hidden_lower_fraction']
    assert r['strict_gap_fraction']==x['certified_information_cost_lower_fraction']
    assert abs(r['scores']['baseline']-x['baseline_TA'])<1e-8
    assert abs(r['scores']['independent_BR']-x['independent_BR_hero_score'])<1e-8
    assert abs(r['hidden_optimum_bounds'][1]-x['achieved_score'])<1e-8
report['comparison_after_recomputation']='All reported headline values agree within1e-8; exact rationals identical.'
(OUT/'D3_raw_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
