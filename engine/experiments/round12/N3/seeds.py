"""Declared known-root and child-state collision checks, before every draw."""
from common import np,BASE,DOMAIN,H,O,save,utc
import json
class Registry:
    def __init__(self):
        self.states={};self.addresses=set();self.prior_roots=0
        def prior(a):self.add(a,record=False);self.prior_roots+=1
        for g in range(6000):prior([2100000000,g])
        for g in range(600):prior([2100000000,10000000+g])
        for stage,base,n in ((0,2100000000,176),(1,2100000000,30800),(2,2100000000,4400),(3,2110000000,3520)):
            for g in range(n):
                for tail in (3,4,5):prior([base,0x4E31,stage,g,tail])
        for namespace,epochs in ((0x4E31,50),(0x4E32,20)):
            for seed in range(5):
                prior([2100000000,namespace,20,seed])
                for epoch in range(1,epochs+1):prior([2100000000,namespace,21,seed,epoch])
        for k in (90,91):prior([2110000000,0x4E31,k])
        assert self.prior_roots==123650
        self.path=O/'seed_addresses.jsonl';self.path.parent.mkdir(exist_ok=True,parents=True)
        if self.path.exists():
            for line in self.path.read_text().splitlines():self.add(json.loads(line)['address'],record=False)
    def add(self,address,record=True):
        a=tuple(map(int,address))
        if a in self.addresses:return
        root=np.random.SeedSequence(list(a));seq=[root]+root.spawn(5)
        states=[s.generate_state(4).tobytes() for s in seq]
        for k,state in enumerate(states):assert state not in self.states,('Seed-state collision',a,k,self.states.get(state))
        for k,state in enumerate(states):self.states[state]=(a,k)
        self.addresses.add(a)
        if record:
            with self.path.open('a',encoding='utf-8') as f:f.write(json.dumps({'address':a,'root_and_5children_hex':[s.hex() for s in states]})+'\n')
    def fit(self):
        for seed in range(5):
            self.add([BASE,DOMAIN,20,seed])
            for epoch in range(1,51):self.add([BASE,DOMAIN,21,seed,epoch])
    def receipt(self):
        return {'utc':utc(),'known_prior_roots':self.prior_roots,'total_root_addresses':len(self.addresses),'total_root_and_child_states':len(self.states),'collisions':0,'scope':'Declared N0/N1/N2 roots and five derived child states, plus every actual N3attempt/fit/game root and children. Does not certify unknown external uses or all historic RNG descendants. Namespaced SHA hand split is not random.'}
