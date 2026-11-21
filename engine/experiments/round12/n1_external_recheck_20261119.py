"""Verify preserved Claude N1 diagnostics; reproduce a fixed subset, no new study."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,math,hashlib
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;N1=H/'N1';A=H/'CLAUDE_EXTERNAL_20261118';O=H/'CODEX_REVIEW_20261119';O.mkdir(exist_ok=True)
sys.path[:0]=[str(N1),str(H),str(H.parents[1])]
import run as frozen
from judgement.state import InfoSnapshot,PLAY
from judgement.sampling import HandSampler
from judgement.policies.hybrid_bi import BidInformedWorlds
DEADLINE=float(os.environ['N123_REVIEW_DEADLINE'])
def check():
    if time.time()>=DEADLINE:raise TimeoutError('N1 diagnostic verification stop')
def save(name,d):(O/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def ci(v):
    v=np.asarray(v,float);m=math.fsum(v)/len(v);half=1.96*math.sqrt(math.fsum((float(x)-m)**2 for x in v)/(len(v)*(len(v)-1)))
    return {'n_rounds':len(v),'mean':m,'ci95':[m-half,m+half],'scope':'paired post-hoc subset round-normal95%; original auditor estimator, not an equivalence test'}
def wilson(k,n):
    z=1.959963984540054;den=1+z*z/n;mid=(k/n+z*z/(2*n))/den;rad=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den;return [mid-rad,mid+rad]
def restore(o):
    trick=[];winners=[];won=[0]*4;voids=[0]*4
    for p,c in o['history']:
        if trick and c//13!=trick[0][1]//13:voids[p]|=1<<(trick[0][1]//13)
        trick.append((p,c))
        if len(trick)==4:
            seat,card=trick[0]
            for q,d in trick[1:]:
                if (d//13==card//13 and d>card) or (d//13!=card//13 and d//13==o['trump']):seat,card=q,d
            winners.append(seat);won[seat]+=1;trick=[]
    assert won==o['won'] and voids==o['voids']
    return InfoSnapshot(seat=o['seat'],n_players=4,dealer=o['dealer'],n_cards=o['n'],trump=o['trump'],hand=o['hand'],bids=tuple(map(tuple,o['bids'])),tricks_won=tuple(won),current_trick=tuple(trick),history=tuple(map(tuple,o['history'])),trick_winners=tuple(winners),voids=tuple(voids),phase=PLAY,cards_left_counts=tuple(o['left']),rules=frozen.RULES)
def high(info,cards,prop,y,seed):
    check();rng=np.random.default_rng(seed);hs=HandSampler(info);worlds=[hs.sample(rng) for _ in range(16384)];bw=BidInformedWorlds(frozen.LIK);lw=bw.log_weights(info,worlds);w=np.exp(lw-lw.max());ess=float(w.sum()**2/(w*w).sum());w/=w.sum();opps=[(info.seat+k)%4 for k in (1,2,3)];masks=np.array([[r[p] for p in opps] for r in worlds],np.uint64);marg=np.zeros((len(cards),4))
    for j,c in enumerate(cards):
        bits=((masks>>np.uint64(c))&np.uint64(1)).astype(bool);owner=np.where(bits.any(1),bits.argmax(1),3);marg[j]=np.bincount(owner,weights=w,minlength=4)
    q=.999*marg+.001*prop;loss=float(-np.log(q[np.arange(len(y)),y]).mean());return loss,ess
def main():
    frozen.check_sources();pred=dict(np.load(N1/'results/confirmation_predictions.npz'));records=[];ys=[];offset=[0]
    for start in range(0,3520,64):
        with np.load(N1/'results/confirmation'/f'{start:05d}.npz') as p:
            ys.extend(p['Y']);offset.extend((p['offset'][1:]+offset[-1]).tolist())
        records.extend(json.loads(s) for s in (N1/'results/confirmation'/f'{start:05d}.jsonl').read_text().splitlines())
    assert len(records)==3520 and len(offset)==3521;ys=np.array(ys);offset=np.array(offset);loss={}
    for name in ('bi','net_full','net_bids'):
        log=-np.log(pred[name][np.arange(len(ys)),ys]);loss[name]=np.array([log[a:b].mean() for a,b in zip(offset[:-1],offset[1:])])
    diagnostic=[]
    for n in range(3,14):
        rows=[r for r in records if r['n']==n];cap=sum(r['sampler']['M']>=2048 and r['sampler']['ess']<256 for r in rows);low=sum(r['sampler']['ess']<32 for r in rows)
        diagnostic.append({'n_cards':n,'n_rounds':len(rows),'max_bank_before_target_ESS':cap,'cap_fraction':cap/len(rows),'cap_fraction_Wilson95':wilson(cap,len(rows)),'ESS_lt32_rounds':low,'ESS_lt32_fraction':low/len(rows),'ESS_lt32_Wilson95':wilson(low,len(rows))})
    zero=(pred['uniform'][np.arange(len(ys)),ys]-.001*pred['prop'][np.arange(len(ys)),ys])<=1e-18
    # Count raw uniform zero marginals from saved original chunks, not the floor.
    zero_cards=0;zero_rounds=0
    for start in range(0,3520,64):
        with np.load(N1/'results/confirmation'/f'{start:05d}.npz') as p:
            zeros=p['uniform'][np.arange(len(p['Y'])),p['Y']]==0;zero_cards+=int(zeros.sum());zero_rounds+=sum(bool(zeros[a:b].any()) for a,b in zip(p['offset'][:-1],p['offset'][1:]))
    save('n1_degeneracy.json',{'by_n':diagnostic,'uniform_zero_counts':{'cards':zero_cards,'all_scored_cards':len(ys),'rounds':zero_rounds,'all_rounds':3520,'round_rate_Wilson95':wilson(zero_rounds,3520),'scope':'Card counts are exact observed totals, dependent within round; population inference uses rounds'}})
    tables={};replayed=[];ns=np.array([r['n'] for r in records])
    initial=np.load(A/'n1/bihigh_16384.npy');assert initial.shape==(100,8);expected=np.sort(np.random.default_rng(12345).choice(np.flatnonzero(ns>=8),100,replace=False));np.testing.assert_array_equal(initial[:,0],expected)
    for row in initial:
        g=int(row[0]);assert abs(row[4]-loss['bi'][g])<1e-12 and abs(row[6]-loss['net_full'][g])<1e-12 and abs(row[7]-loss['net_bids'][g])<1e-12
    tables['initial_100_n_ge8']={'BI_K256-minus-weighted16384':ci(initial[:,4]-initial[:,5]),'full-minus-weighted16384':ci(initial[:,6]-initial[:,5]),'bids-minus-weighted16384':ci(initial[:,7]-initial[:,5])}
    jobs=[(initial[i],False) for i in (0,len(initial)-1)]
    for lo,hi in ((8,13),(5,7)):
        rows=np.load(A/'n1_skeptic1'/f'bi2_{lo}_{hi}_16384.npy');assert rows.shape==(60,10);expected=np.sort(np.random.default_rng(424242).choice(np.flatnonzero((ns>=lo)&(ns<=hi)),60,replace=False));np.testing.assert_array_equal(rows[:,0],expected)
        for r in rows:
            g=int(r[0]);assert abs(r[3]-loss['bi'][g])<1e-12 and abs(r[4]-loss['net_full'][g])<1e-12 and abs(r[5]-loss['net_bids'][g])<1e-12
        tables[f'n{lo}_{hi}']={'full-minus-highA':ci(rows[:,4]-rows[:,6]),'full-minus-highB':ci(rows[:,4]-rows[:,8]),'bids-minus-highA':ci(rows[:,5]-rows[:,6]),'bids-minus-highB':ci(rows[:,5]-rows[:,8]),'highA-minus-highB':ci(rows[:,6]-rows[:,8]),'same_60_contexts_for_two_MC_replicates':True,'overlap_with_initial_100':len(set(rows[:,0])&set(initial[:,0]))}
        jobs.extend((rows[i],True) for i in (0,30,59))
    save('n1_high_budget_reaggregation.json',tables)
    for row,second in jobs:
        check();g=int(row[0]);r=records[g];info=restore(r['observation']);a,b=offset[g:g+2];prop=pred['prop'][a:b];y=ys[a:b]
        reps=((1,6,7),(2,8,9)) if second else ((0,5,3),)
        for rep,loss_col,ess_col in reps:
            seed=[31337,g,rep] if second else [777,g];value,ess=high(info,r['cards'],prop,y,seed)
            assert abs(value-row[loss_col])<1e-10,(g,rep,value,row[loss_col]);assert abs(ess-row[ess_col])<1e-7
            replayed.append({'g':g,'seed':seed,'M':16384,'loss_abs_error':abs(value-row[loss_col]),'ESS_abs_error':abs(ess-row[ess_col])});save('n1_high_replay_progress.json',replayed)
            print('replayed high-budget',g,rep,flush=True)
    save('n1_complete.json',{'status':'PASS within scope; no convergence/equivalence/runtime-budget claim','high_budget_subsets_reaggregated':tables,'n_exact_high_budget_record_replays':len(replayed),'numeric_replays':replayed,'notes':['Auditor A/B are two MC draws on the same60contexts per range, not independent chance replications.','16384counts proposals; originalK256counts resampled worlds after64..2048proposals. 64x is an output-count ratio, not measured compute.','High-budget diagnostic uses weighted marginals; original baseline uses finite SIR resampling.','No formal posterior convergence or prospective equivalence margin was tested.']})
if __name__=='__main__':main()
