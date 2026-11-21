"""Metadata-only seed collision audit. No fitting or chance outcomes."""
from n9lib import *
def main():
    known=set()
    def add(a):
        r=np.random.SeedSequence(a)
        for s in [r]+r.spawn(5):known.add(s.generate_state(4).tobytes())
    for g in range(6000):add([2100000000,g])
    for g in range(600):add([2100000000,10000000+g])
    for stage,base,n in ((0,2100000000,176),(1,2100000000,30800),(2,2100000000,4400),(3,2110000000,3520)):
        for g in range(n):
            for tail in (3,4,5):add([base,0x4e31,stage,g,tail])
    for ns,epochs in ((0x4e31,50),(0x4e32,20)):
        for s in range(5):
            add([2100000000,ns,20,s])
            for e in range(1,epochs+1):add([2100000000,ns,21,s,e])
    for k in (90,91):add([2110000000,0x4e31,k])
    for line in (R/'N3/results/seed_addresses.jsonl').read_text().splitlines():
        for h in json.loads(line)['root_and_5children_hex']:known.add(bytes.fromhex(h))
    for b,n in ((2150020000,4),(2150000000,300),(2150010000,300)):
        for g in range(n):add(b+g)
    for r in json.loads((R/'N5/preflight.json').read_text())['new']:known.add(bytes.fromhex(r['state']))
    prior=len(known);new=[]
    addresses=[[BASE,DOMAIN,99]]
    for ci in range(4):
        for ai in range(2):
            for si in range(5):addresses += [[BASE,DOMAIN,ci,ai,si,0]]+[[BASE,DOMAIN,ci,ai,si,1,e] for e in range(1,81)]
    for a in addresses:
        r=np.random.SeedSequence(a);states=[s.generate_state(4).tobytes() for s in [r]+r.spawn(5)]
        assert len(set(states))==6 and not set(states)&known;known.update(states);new.append({'address':a,'root_and5children':[s.hex() for s in states]})
    save(H/'preflight.json',{'utc':stamp(),'base':BASE,'domain':DOMAIN,'prior_known_states':prior,'new_roots':len(new),'new_root_and_child_states':6*len(new),'collisions':0,'roots':new,'intentional_CRN':'same initialization/shuffle across losses and matched MLPfeature controls; short20 is exact prefix of long80','scope':'Known N0-N5declared roots/children and N3actualjournal; N6/N11noRNG. No unknown external-use/all-history guarantee; metadata grep found no2170M-2220Mbase before reservation.'});print('N9 seed preflight passed',prior,len(new),flush=True)
if __name__=='__main__':main()
