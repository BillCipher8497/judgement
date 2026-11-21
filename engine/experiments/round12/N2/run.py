"""Proposed N2 source-gated single-worker pipeline; no experimental CLI bypass.
Unexecuted prototype until synthetic/source review and pre-run commit pass.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,itertools,json,math,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;ENGINE=H.parents[2];ROOT=ENGINE.parent
sys.path[:0]=[str(H),str(R),str(ENGINE),str(ENGINE/'experiments/round9/D3')]
from judgement.cfr.toy import ToyGame
from judgement.cfr.heuristics import myopic_bid
from bounded_ta import ta_card,clear_mid
from export import deals,write_chunk,merge,digest
from corpus import Corpus
from fit import fit_models,save,check_deadline
from evaluate import evaluate
O=H/'results';CONFIGS=[(2,2,0,2520),(2,2,-1,2520),(3,3,0,369600),(3,3,-1,369600)]
SOURCES=sorted({H/name for name in ('run.py','export.py','features.py','joint.py','neural.py','corpus.py','fit.py','evaluate.py','reference.py','verify.py','bid_premise.py','determinism_check.py','self_check.py','seed_audit.py','PROTOCOL.md')}|{R/'learning.py',ENGINE/'experiments/round9/D3/bounded_ta.py'}|
 {Path(m.__file__).resolve() for name,m in list(sys.modules.items()) if name.startswith('judgement') and getattr(m,'__file__',None) and str(m.__file__).endswith('.py')})

def status(text):
    with (H/'STATUS.txt').open('a',encoding='utf-8') as f:f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' '+text+'\n')

def packet():
    raw={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in SOURCES};canonical={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in SOURCES}
    fingerprint=hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()
    return {'raw_sources':raw,'canonical_sources':canonical,'source_fingerprint':fingerprint,'numpy':np.__version__,'BLAS':str(np.__config__.CONFIG),'conditions':CONFIGS,'profile':'Four private/public deterministic toy TA policies, myopic bids, no pruning; every play decision recorded','fit_counts':{'train_key_cap':30000,'validation_key_cap':4000,'seeds_per_condition':5,'epochs':20},'primary':'Per-condition net-minus-BI joint KL on complete reached test-hand population; fit seed t95%CI only, chance fully enumerated'}

def source_check():
    freeze=json.loads((O/'source_freeze.json').read_text())
    for rel,h in freeze['raw_sources'].items():assert digest(ROOT/rel)==h,('Frozen source drift',rel)

def approved(p):
    external='N2_SOURCE_APPROVED '+p['source_fingerprint'] in (R/'REVIEW.md').read_text(encoding='utf-8')
    own=H/'SOURCE_REVIEW.md'
    independent=own.exists() and ('N2_CODEX_SOURCE_REVIEWED '+p['source_fingerprint'] in own.read_text(encoding='utf-8')) and (R/'INDEPENDENT_CONTINUATION.md').exists()
    assert external or independent,'Exact source review required; never manufacture Claude approval'
    authority='Claude external source approval' if external else 'self-reviewed(Codex), pending external audit; owner-authorized independent continuation'
    commit=os.environ['N2_APPROVED_COMMIT']
    for rel,h in p['canonical_sources'].items():
        raw=subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT);assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==h
    return commit,authority

def folder(S,n,tr):return O/f'S{S}R4n{n}_t{tr}'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args();p=packet()
    if args.preflight:
        from seed_audit import audit
        audit();save(H/'source_preflight.json',p);print(p['source_fingerprint']);return
    commit,authority=approved(p);deadline=float(os.environ['JUDGEMENT_N2_DEADLINE']);check_deadline(deadline)
    # Each authorized launch has both its absolute bound and max3elapsedhours.
    deadline=min(deadline,time.time()+10800);O.mkdir(exist_ok=True);p['pre_run_commit']=commit
    p['source_review_authority']=authority;freeze_path=O/'source_freeze.json'
    if freeze_path.exists():
        original=json.loads(freeze_path.read_text());current=json.loads(json.dumps(p))
        # A later external acknowledgement does not rewrite the origin record.
        assert {k:v for k,v in original.items() if k!='source_review_authority'}=={k:v for k,v in current.items() if k!='source_review_authority'}
    else:save(freeze_path,p)
    freeze=digest(freeze_path);timer=threading.Timer(deadline-time.time(),lambda:os._exit(124));timer.daemon=True;timer.start()
    try:
        from seed_audit import audit
        audit()
        # Resource pilots are fixed deterministic prefixes, retained as raw data.
        pilots=[]
        for S,n,tr,total in CONFIGS:
            check_deadline(deadline);source_check();g=ToyGame(N=4,S=S,R=4,n=n,trump=tr);f=folder(S,n,tr);f.mkdir(exist_ok=True)
            hands=list(itertools.islice(deals(tuple(g.deck),n),256));r=write_chunk(g,f/'raw',0,hands,myopic_bid,ta_card,freeze,deadline);pilots.append(r['wall_seconds']/256*total);clear_mid()
        resource_path=O/'resource_gate.json'
        if resource_path.exists():resource=json.loads(resource_path.read_text())
        else:
            projected=sum(pilots)+5400;resource={'pilot_deals_per_condition':256,'projected_full_traversal_seconds':sum(pilots),'fixed_fit_joint_audit_reserve_seconds':5400,'projected_total_seconds':projected,'remaining_seconds':deadline-time.time(),'accepted':projected<=deadline-time.time(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};save(resource_path,resource)
        if not resource['accepted']:
            save(O/'scope_failure.json',{'reason':'Predeclared resource-only gate','gate':resource,'no_complete_posterior_claim':True});status('N2resource-only scope failure; pilot chunks retained, no posterior/sample estimate promoted.');return
        results=[]
        for S,n,tr,total in CONFIGS:
            check_deadline(deadline);g=ToyGame(N=4,S=S,R=4,n=n,trump=tr);f=folder(S,n,tr);gen=deals(tuple(g.deck),n);list(itertools.islice(gen,256));start=256
            while start<total:
                check_deadline(deadline);source_check();hands=list(itertools.islice(gen,min(1024,total-start)));assert hands
                write_chunk(g,f/'raw',start,hands,myopic_bid,ta_card,freeze,deadline);start+=len(hands);status(f'Exacttrajectorysaved{start}/{total}deals:S{S}n{n}trump{tr}')
            source_check();manifest=merge(g,f/'raw',total,deadline);assert manifest['source_freeze_sha256']==freeze;status(f'Exactposteriorcomplete:S{S}n{n}trump{tr};{manifest["n_private_play_infosets"]}privatekeys')
            corpus=Corpus(g,f);fit_models(corpus,f/'models',freeze,deadline,source_check,status);result=evaluate(corpus,f/'models',f/'joint_KL',freeze,deadline,source_check,status,myopic_bid);results.append(result);clear_mid()
            save(O/'completed_conditions.json',{'conditions':[r['condition'] for r in results],'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        save(O/'summary.json',{'conditions':results,'source_freeze_sha256':freeze,'status':'Ready for separate verification; implementation/results pending external audit. M16 conditional mathematics externally checked by Claude17Nov.','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});status('AllN2fixedconditions complete; independent verification/external review required, no publication yet.')
    finally:timer.cancel()
if __name__=='__main__':main()
