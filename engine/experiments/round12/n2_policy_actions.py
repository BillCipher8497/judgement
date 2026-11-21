"""Post-outcome diagnostic: replay fixed TA choice at every private key.

The frozen original verifier checks all raw actions for legality/consistency
and direct policy-conditioned reach at all small/64large contexts. This extra
check confirms the declared policy choice everywhere, reusing its definition.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
from pathlib import Path
H=Path(__file__).resolve().parent/'N2';sys.path.insert(0,str(H))
import run as science
from export import decode,digest
from reference import public,legal
import numpy as np
def main():
    deadline=float(os.environ['JUDGEMENT_N2_AUDIT_DEADLINE']);results={}
    for S,n,tr,total in science.CONFIGS:
        g=science.ToyGame(N=4,S=S,R=4,n=n,trump=tr);folder=science.folder(S,n,tr);corpus=science.Corpus(g,folder);actions=np.full(len(corpus.keys),255,np.uint8);free=0
        for start in range(0,len(actions),1024):
            assert time.time()<deadline
            for i in range(start,min(start+1024,len(actions))):
                p,h,hist=decode(corpus.keys[i]);played,voids,actor=public(g,hist);assert p==actor;tl=(len(hist)-4)%4;trick=[((p-tl+k)%4,c) for k,c in enumerate(hist[-tl:])] if tl else [];choices=legal(g,[c for c in h if c not in hist[4:]],trick)
                if len(choices)==1:actions[i]=choices[0]
                else:actions[i]=science.ta_card(g,p,h,hist);free+=1
                assert actions[i] in choices
            science.clear_mid()
        count=0
        for path in sorted((folder/'raw').glob('*.npz')):
            assert time.time()<deadline
            with np.load(path) as a:
                ids=np.searchsorted(corpus.keys,a['keys']);np.testing.assert_array_equal(actions[ids],a['chosen_card']);count+=len(ids)
        results[folder.name]={'all_private_keys':len(actions),'nonforced_TA_choices_recomputed':free,'all_raw_decision_world_actions_match':count,'policy_function':'Frozen bounded_ta.ta_card reused as fixed profile definition; this is action/input replay, not a second independently coded TA heuristic'};(science.O/'policy_action_verification_progress.json').write_text(json.dumps({'conditions':results},indent=2));science.clear_mid()
    result={'conditions':results,'status':'pass, self-checked(Codex), external implementation/results review pending','verifier_sha256':digest(__file__),'scope':'Post-outcome diagnostic, no new chance/fit samples, model or selected result changed','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};(science.O/'policy_action_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
