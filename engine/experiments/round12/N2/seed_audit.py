"""N2 namespace proposal: metadata-only state collision audit, no game draws."""
import json,subprocess,re
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[3];N1=0x4E31;N2=0x4E32

def audit():
    states={}
    def add(address,name):
        state=np.random.SeedSequence(address).generate_state(4).tobytes();assert state not in states,('Root-state collision',name,states.get(state));states[state]=name;return state.hex()
    for g in range(6000):add([2100000000,g],f'N0/train/{g}')
    for g in range(600):add([2100000000,10000000+g],f'N0/test/{g}')
    for stage,base,n in ((0,2100000000,176),(1,2100000000,30800),(2,2100000000,4400),(3,2110000000,3520)):
        for g in range(n):
            for tail in (3,4,5):add([base,N1,stage,g,tail],f'N1/{stage}/{g}/{tail}')
    for seed in range(5):
        add([2100000000,N1,20,seed],f'N1/init/{seed}')
        for epoch in range(1,51):add([2100000000,N1,21,seed,epoch],f'N1/batch/{seed}/{epoch}')
    add([2110000000,N1,90],'N1/calibration');add([2110000000,N1,91],'N1/predeclared-secondary-diagnostic')
    prior=len(states);new={}
    for seed in range(5):
        new[f'init/{seed}']=add([2100000000,N2,20,seed],f'N2/init/{seed}')
        for epoch in range(1,21):new[f'batch/{seed}/{epoch}']=add([2100000000,N2,21,seed,epoch],f'N2/batch/{seed}/{epoch}')
    assert prior==123545 and len(new)==105 and len(states)==123650
    result={'utc':datetime.now(timezone.utc).isoformat(),'N2_optimizer_base':2100000000,'N2_namespace':N2,'known_prior_root_states':prior,'new_distinct_root_states':105,'total_distinct_root_states':len(states),'collisions':0,'new_states_hex':new,
        'intentional_CRN':'Same N2seed index initializes all4condition models and uses the same per-epoch shuffle root; counted once, no claim of independent fits across conditions.',
        'limitation':'128-bit root-state collision check for declared known N0/N1/N2addresses, not a proof about child substreams or unknowable external uses. Hand/key splits use namespaced SHA256, no random seed stage. No2120M–2160Mdraws.'}
    target=H/'preflight';target.mkdir(exist_ok=True);path=target/'seed_audit.json'
    if path.exists():
        old=json.loads(path.read_text())
        assert {k:v for k,v in old.items() if k!='utc'}=={k:v for k,v in result.items() if k!='utc'}
        return old
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result
if __name__=='__main__':print(json.dumps(audit()))
