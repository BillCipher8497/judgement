"""Manual public transitions, independent factor forwards and paired summaries."""
from lib12 import *
from reference import legal,winner,structural_worlds,follows_observed_history
import math

def manual(g,w,hero,hist,action):
    left=[list(h) for h in w];won=[0]*4;trick=[];leader=0
    for c in hist[4:]:
        p=(leader+len(trick))%4;assert c in legal(g,left[p],trick);left[p].remove(c);trick.append((p,c))
        if len(trick)==4:leader=winner(g,trick);won[leader]+=1;trick=[]
    h=hist;used=False
    while len(h)<4+4*g.n:
        p=(leader+len(trick))%4;choices=legal(g,left[p],trick)
        if p==hero:
            if not used:c=action;used=True
            else:assert len(choices)==1;c=choices[0]
        else:c=choices[0] if len(choices)==1 else ta_card(g,p,w[p],h)
        assert c in choices;left[p].remove(c);trick.append((p,c));h+=(c,)
        if len(trick)==4:leader=winner(g,trick);won[leader]+=1;trick=[]
    return 10+hist[hero] if won[hero]==hist[hero] else 0

def fresh_z(r,p,rich):
    a=r.x.copy()
    if not rich:a[:,:,35:]=0
    a=a.reshape(-1,245)
    for k in range(0,len(p),2):
        a=np.einsum('ij,jk->ik',a,p[k])+p[k+1]
        if k+2<len(p):a=np.maximum(a,0)
    a=a.reshape(-1,4)+np.log(np.maximum(r.prop,1e-300));return np.where(r.mask,a,-np.inf)

def check_ci(c,v):
    m=math.fsum(map(float,v))/5;s=math.sqrt(math.fsum((float(x)-m)**2 for x in v)/4);h=2.7764451051977987*s/math.sqrt(5);assert abs(c['mean']-m)<1e-12 and np.allclose(c['ci95'],[m-h,m+h],rtol=0,atol=1e-11) and c['n_optimizer_seeds']==5

def main():
    sources=json.loads((O/'sources.json').read_text());assert all(sha(Path(p))==v for p,v in sources.items());selection=json.loads((O/'selection.json').read_text());e=json.loads((O/'evaluation.json').read_text());fresh=[];nvalues=0
    for tag in TAGS:
        g=B.game(tag);rs=[r for r in rows(tag) if r.split==2 and eligible(g,r)];assert [r.kid for r in rs]==selection[tag]['key_ids'];tables={};uniform=0;reach=sum(r.count for r in rs)
        for j,r in enumerate(rs):
            check();a=B.load(O/tag/f'{r.kid:07d}.npz');hero,hand,hist=B.decode(r.key);ww=[tuple(tuple(map(int,h)) for h in w) for w in a['hands']];assert set(ww)==set(structural_worlds(g,hero,hand,hist));assert all(follows_observed_history(g,w,hist) for w in ww)
            for i,w in enumerate(ww):
                for k,c in enumerate(a['actions']):assert manual(g,w,hero,hist,int(c))==a['Q'][i,k];nvalues+=1
            assert np.array_equal(a['posterior'],r.posterior);v=a['Q'][r.posterior].astype(np.int64).sum(0);uniform+=int(v.max()-v[a['Q'].astype(np.int64).sum(0).argmax()]);tables[r.kid]=a;clear_mid()
            if j%64==0:save(O/'verification_progress.json',{'tag':tag,'stage':'manual_F_action_values','keys':j+1,'target':len(rs),'n_integer_values':nvalues,'utc':stamp()})
        assert str(Fraction(uniform,reach))==e['controls'][tag]['uniform_F_private_regret_exact'];assert e['controls'][tag]['reach']==reach
        for row in [a for a in e['rows'] if a['tag']==tag]:
            check();cfg=row['config'];loss=row['loss'];seed=row['seed'];budget=row['budget'];folder=R/'N9/results'/tag/'models'/f'{cfg}_{loss}_s{seed}';d=B.load(folder/f'selected{budget}.npz');p=[d[f'p{i}'] for i in range(len(d))];saved=json.loads((O/tag/f'{cfg}_{loss}_s{seed}_b{budget}.json').read_text());decisions=[];num=0;KL=0.;ties=0
            for r in rs:
                z=fresh_z(r,p,cfg=='rich_mlp');scores=[math.fsum(float(z[c,k]) for c,k in enumerate(owners)) for owners in r.F];mx=max(scores);den=math.fsum(math.exp(s-mx) for s in scores);lp=[s-mx-math.log(den) for s in scores];prob=[math.exp(a) for a in lp];q=tables[r.kid]['Q'];v=[math.fsum(a*int(s) for a,s in zip(prob,q[:,k])) for k in range(2)];choice=next(k for k in range(2) if v[k]>=max(v)-1e-10);true=q[r.posterior].astype(np.int64).sum(0);deficit=int(true.max()-true[choice]);num+=deficit;decisions.append([r.kid,choice,deficit]);KL+=r.count*(-math.log(r.count)-math.fsum(lp[int(i)] for i in r.posterior)/r.count);ties+=abs(v[0]-v[1])<=1e-10
            assert decisions==saved['decisions_key_choice_deficit'] and num==row['regret_numerator'] and row['private_regret_exact']==str(Fraction(num,reach));assert abs(KL/reach-row['joint_KL'])<1e-9 and ties==row['n_numeric_nearties'];fresh.append(row)
        save(O/'verification_progress.json',{'tag':tag,'stage':'models','n_selected_recipes_checked':len(fresh),'utc':stamp()})
    def v(tag,cfg,loss,budget,metric='private_regret'):return [r[metric] for s in range(5) for r in fresh if (r['tag'],r['config'],r['loss'],r['budget'],r['seed'])==(tag,cfg,loss,budget,s)]
    nc=0
    for tag in TAGS:
        for cfg in CONFIGS:
            for loss in LOSSES:
                for budget in (20,80):
                    key=f'{tag}/{cfg}/{loss}/{budget}';check_ci(e['arm_regret'][key],v(tag,cfg,loss,budget));check_ci(e['arm_KL_same_keys'][key],v(tag,cfg,loss,budget,'joint_KL'));nc+=2
                check_ci(e['long_minus_short'][f'{tag}/{cfg}/{loss}'],np.array(v(tag,cfg,loss,80))-v(tag,cfg,loss,20));nc+=1
            for budget in (20,80):check_ci(e['joint_minus_marginal'][f'{tag}/{cfg}/{budget}'],np.array(v(tag,cfg,'joint',budget))-v(tag,cfg,'marginal',budget));nc+=1
        for loss in LOSSES:
            for budget in (20,80):check_ci(e['rich_minus_summary'][f'{tag}/{loss}/{budget}'],np.array(v(tag,'rich_mlp',loss,budget))-v(tag,'summary_mlp',loss,budget));nc+=1
    assert len(fresh)==240;save(O/'verification.json',{'utc':stamp(),'all_structural_F_sets_and_integer_payoffs_manually_replayed':True,'n_integer_world_action_values':nvalues,'n_selected_recipes':len(fresh),'all_forwards_probability_decisions_rationals_match':True,'n_intervals_recomputed':nc,'source_sha256':sha(Path(__file__)),'evaluation_sha256':sha(O/'evaluation.json'),'scope':'Separate Codex check; original R membership relies on separately audited immutable N9/N2supports; not external validation, new seed replication or a solved private game. Pending Claude, local.'});log(H,'N12 separate checks pass; pendingClaude/local.')
if __name__=='__main__':main()
