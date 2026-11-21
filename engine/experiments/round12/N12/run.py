"""Exact one-step decision tables, then frozen N9 belief decoding."""
from lib12 import *

def main():
    O.mkdir(exist_ok=True);old=json.loads((R/'N9/results/sources.json').read_text());assert all(sha(Path(p))==v for p,v in old.items());assert json.loads((R/'N9/results/verification.json').read_text())['all_independent_forward_DP_normalizers_selection_splits_raw_and_CIs_pass']
    paths=[H/'lib12.py',H/'run.py',H/'verify.py',H/'PROTOCOL.md',R/'N9/results/evaluation.json',R/'N9/results/training_complete.json',R/'N9/results/verification.json',R/'N9/n9lib.py',R/'N2/reference.py',B.ENGINE/'experiments/round9/D3/bounded_ta.py',B.ENGINE/'judgement/cfr/toy.py',B.ENGINE/'judgement/cfr/heuristics.py']
    for tag in TAGS:
        paths.append(R/'N9/results'/tag/'data.npz')
        paths.extend((R/'N9/results'/tag/'models').glob('*/selected*.npz'));paths.extend((R/'N9/results'/tag/'models').glob('*/held_*.npz'))
    sources={str(p):sha(p) for p in paths};fp=O/'sources.json'
    if fp.exists():assert json.loads(fp.read_text())==sources
    else:save(fp,sources)
    sels={};tables={}
    for tag in TAGS:
        g=B.game(tag);test=[r for r in rows(tag) if r.split==2];selected=[r.kid for r in test if eligible(g,r)];assert selected;sels[tag]={'key_ids':selected,'n_keys':len(selected),'original_N9_test_keys':len(test),'rng':'NONE: deterministic derived benchmark, frozen2170Mmodels'}
    sel=O/'selection.json'
    if sel.exists():assert json.loads(sel.read_text())==sels
    else:save(sel,sels)
    for tag in TAGS:
        g=B.game(tag);folder=O/tag;folder.mkdir(exist_ok=True);tables[tag]={}
        for k,r in enumerate(r for r in rows(tag) if r.kid in set(sels[tag]['key_ids'])):
            check();path=folder/f'{r.kid:07d}.npz'
            if not path.exists():
                actor,hand,hist=B.decode(r.key);left,_,trick,_,_,_=g.replay(worlds(g,r)[0],hist);actions=g.legal_cards(left[actor],trick);assert len(actions)==2;ww=worlds(g,r);v=np.array([[rollout(g,w,actor,hist,a) for a in actions] for w in ww],np.int16);B.npz(path,Q=v,hands=np.array(ww,np.uint8),actions=np.array(actions),posterior=r.posterior,key=np.frombuffer(bytes(r.key).ljust(20,b'\0'),np.uint8));clear_mid()
            tables[tag][r.kid]=B.load(path)
            if k%64==0:save(O/'progress.json',{'stage':'tables','tag':tag,'complete':k+1,'target':sels[tag]['n_keys'],'utc':stamp()})
    results=[];controls={}
    for tag in TAGS:
        test=[r for r in rows(tag) if r.split==2];ids=[r.kid for r in test];reach=sum(r.count for r in test if r.kid in tables[tag]);uniform=0
        for r in test:
            if r.kid in tables[tag]:
                q=tables[tag][r.kid]['Q'];v=q[r.posterior].astype(np.int64).sum(0);uniform+=int(v.max()-v[q.astype(np.int64).sum(0).argmax()])
        controls[tag]={'n_keys':len(tables[tag]),'reach':reach,'uniform_F_private_regret_exact':str(Fraction(uniform,reach)),'oracle_R_private_regret_exact':'0','kind':'exact fixed selected support; no sampling CI'}
        for cfg in CONFIGS:
            for loss in LOSSES:
                for seed in range(5):
                    for budget in (20,80):
                        check();folder=R/'N9/results'/tag/'models'/f'{cfg}_{loss}_s{seed}';held=B.load(folder/f'held_{budget}.npz');assert held['key_ids'].tolist()==ids;offset=0;num=0;kl=0.;decisions=[];nearties=0
                        for j,r in enumerate(test):
                            z=held['logits'][offset:offset+len(r.x)];offset+=len(r.x)
                            if r.kid not in tables[tag]:continue
                            q=tables[tag][r.kid]['Q'];prob=probability(z,r);expected=prob@q;choice=int(np.flatnonzero(expected>=expected.max()-1e-10)[0]);v=q[r.posterior].astype(np.int64).sum(0);deficit=int(v.max()-v[choice]);num+=deficit;kl+=r.count*float(held['KL'][j]);nearties+=abs(expected[0]-expected[1])<=1e-10;decisions.append([r.kid,choice,deficit])
                        row={'tag':tag,'config':cfg,'loss':loss,'seed':seed,'budget':budget,'n_keys':len(tables[tag]),'reach':reach,'regret_numerator':num,'private_regret_exact':str(Fraction(num,reach)),'private_regret':num/reach,'joint_KL':kl/reach,'n_numeric_nearties':nearties,'selected_model_sha256':sha(folder/f'selected{budget}.npz'),'held_output_sha256':sha(folder/f'held_{budget}.npz')};results.append(row);save(O/tag/f'{cfg}_{loss}_s{seed}_b{budget}.json',row|{'decisions_key_choice_deficit':decisions});save(O/'progress.json',{'stage':'frozen_models','complete':len(results),'target':240,'utc':stamp()})
    assert all(sha(Path(p))==v for p,v in sources.items());save(O/'evaluation.json',summarize(results)|{'controls':controls,'selection':sels,'utc':stamp(),'source_sha256':sha(fp),'scope':'Known-hand deterministic two-card derived benchmark; exact private continuation, no future free hero choice. Optimizer CIs only; no fresh replication/fullgame benefit.'});log(H,'N12 fixed derived benchmark complete, separate audit next; pendingClaude/local.')
if __name__=='__main__':main()
