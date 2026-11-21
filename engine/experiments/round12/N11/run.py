"""New selector only; unchanged already-checked N6 algorithms, no old writes."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,sys,time,hashlib
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;O=H/'results';sys.path.insert(0,str(R))
from common import save,sha,stamp,log
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
N6=module('n6_fixed_primary',R/'N6/run.py')
previous=sys.modules.get('run');sys.modules['run']=N6
V=module('n6_fixed_direct_verifier',R/'N6/verify.py')
if previous is None:del sys.modules['run']
else:sys.modules['run']=previous

def corpus(tag):
    folder=R/'N2/results'/tag;m=json.loads((folder/'posterior_manifest.json').read_text());assert sha(folder/'posterior_supports.npz')==m['posterior_sha256']
    with np.load(folder/'posterior_supports.npz') as z:return {k:z[k] for k in ('keys','reach_count','support_offset','support_deal_index')},m
def guard(sources):
    assert all(sha(Path(p))==v for p,v in sources.items())
    assert time.time()<float(os.environ['JUDGEMENT_JOB_DEADLINE'])
def main():
    O.mkdir(exist_ok=True);old=json.loads((R/'N6/results/complete.json').read_text())
    assert np.__version__==json.loads((R/'N2/results/source_freeze.json').read_text())['numpy']=='2.5.4'
    sources=old['sources']|{str(p):sha(p) for p in (Path(__file__),H/'PROTOCOL.md')};guard(sources)
    fp=O/'sources.json'
    if fp.exists():assert json.loads(fp.read_text())==sources
    else:save(fp,sources)
    summaries=[]
    for trump in (0,-1):
        tag=f'S3R4n3_t{trump}';g=N6.ToyGame(N=4,S=3,R=4,n=3,trump=trump);d,m=corpus(tag);folder=O/tag;folder.mkdir(exist_ok=True);strata={p:[] for p in range(4)}
        for i,k in enumerate(d['keys']):
            if d['reach_count'][i]<2:continue
            p,hand,hist=N6.decode(k)
            if len(set(hand)-set(hist[4:]))==3 and len(N6.legal(g,hand,hist))>=2:
                assert len(hist)-4==p;strata[p].append(i)
        chosen={p:ids if len(ids)<=64 else [ids[int(j)] for j in np.linspace(0,len(ids)-1,64,dtype=int)] for p,ids in strata.items()}
        selected=[i for p in range(4) for i in chosen[p]];assert len(selected)==len(set(selected)) and selected
        selection={'indices':selected,'stratum_eligible':{p:len(ids) for p,ids in strata.items()},'stratum_selected':{p:len(ids) for p,ids in chosen.items()},'posterior_sha256':m['posterior_sha256']}
        sp=folder/'selection.json'
        if sp.exists():assert json.loads(sp.read_text())==json.loads(json.dumps(selection))
        else:save(sp,selection)
        allworlds=g.deals();rows=[]
        for position,i in enumerate(selected):
            guard(sources);path=folder/f'{i:07d}.json'
            if path.exists():rows.append(json.loads(path.read_text()));continue
            p,hand,hist=N6.decode(d['keys'][i]);ids=d['support_deal_index'][int(d['support_offset'][i]):int(d['support_offset'][i+1])].astype(int);worlds=tuple(allworlds[j] for j in ids)
            assert len(worlds)>=2 and all(w[p]==hand for w in worlds)
            row=N6.solve(g,worlds,p,hist)|{'key_index':i,'key_hex':bytes(d['keys'][i]).ljust(20,b'\0').hex(),'support_ids_sha256':hashlib.sha256(ids.astype('<u4').tobytes()).hexdigest(),'utc':stamp()}
            save(path,row);rows.append(row);N6.clear_mid();save(O/'progress.json',{'stage':'primary','tag':tag,'complete':position+1,'target':len(selected),'utc':stamp()})
        summary={'tag':tag,'n_keys':len(rows),'reach_weight':sum(r['support_n'] for r in rows),'positive_first_action_loss_keys':sum(r['first_action_loss_sum']>0 for r in rows),'positive_max_value_gap_keys':sum(r['relaxation_gap_sum']>0 for r in rows),'positive_any_action_PI_gap_keys':sum(any(a>b for a,b in zip(r['PI_Q_sums'],r['BR_Q_sums'])) for r in rows),'first_action_loss_numerator':sum(r['first_action_loss_sum'] for r in rows),'relaxation_gap_numerator':sum(r['relaxation_gap_sum'] for r in rows),'scope':'Exact fixed stage-stratified diagnostic, not population estimate; no sampling CI'}
        summaries.append(summary);save(folder/'summary.json',summary);log(H,'Primary '+json.dumps(summary))
    complete={'summaries':summaries,'sources':sources}
    cp=O/'primary_complete.json'
    if cp.exists():assert json.loads(cp.read_text())==complete
    else:save(cp,complete)
    # Separate full pass after ALL primary results and selectors are frozen.
    checks=[]
    for s in summaries:
        tag=s['tag'];trump=0 if tag.endswith('t0') else -1;g=N6.ToyGame(N=4,S=3,R=4,n=3,trump=trump);d,_=corpus(tag);folder=O/tag;selected=json.loads((folder/'selection.json').read_text())['indices'];rows=[]
        for position,i in enumerate(selected):
            guard(sources);path=folder/f'{i:07d}.json';row=json.loads(path.read_text());p,hand,hist=N6.decode(d['keys'][i]);_,worlds=V.direct_posterior(g,p,hand,hist,V.myopic_bid,N6.ta_card)
            support=d['support_deal_index'][int(d['support_offset'][i]):int(d['support_offset'][i+1])].astype(int).tolist();assert sorted(V.deal_index(g,w) for w in worlds)==support
            pi,br=V.independent_values(g,worlds,p,hist,row['actions']);assert pi==row['PI_Q_sums'] and br==row['BR_Q_sums'];j=max(range(len(pi)),key=lambda j:pi[j]);assert max(br)-br[j]==row['first_action_loss_sum'] and max(pi)-max(br)==row['relaxation_gap_sum']
            rows.append(row);save(O/'verification_progress.json',{'tag':tag,'checked':position+1,'target':len(selected),'utc':stamp()});N6.clear_mid()
        assert sum(r['first_action_loss_sum'] for r in rows)==s['first_action_loss_numerator'] and sum(r['relaxation_gap_sum'] for r in rows)==s['relaxation_gap_numerator']
        checks.append({'tag':tag,'n_keys':len(rows),'all_supports_and_integer_values_match':True})
    save(O/'verification.json',{'utc':stamp(),'checks':checks,'primary_sha256':sha(cp),'sources':sources,'verdict':'self-checked Codex; pending Claude external'})
    log(H,'All N11 primary keys separately verified; no outcome-based extension; pending external audit.')
if __name__=='__main__':main()
