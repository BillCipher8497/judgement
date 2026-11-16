"""Audit completed laws and independently recompute all saved bid summaries."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
checks={}
for seed in [4242,4243]:
    count=0
    cells={};context_rows=0
    with (HERE/'results'/f'raw_{seed}.jsonl').open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            r=json.loads(line);n=r['n'];legal=r['legal']
            assert r['i']==i and r['seed']==seed
            expected=set(range(n+1))
            if 'hand' in r:
                assert r['hand'].bit_count()==n
                if r['pos']==3:
                    expected.discard(n-sum(b for _,b in r['prior_bids']))
                assert set(legal)==expected and len(r['prior_bids'])==r['pos']
                context_rows+=1
            else:
                assert seed==4242  # Historical discovery rows predate context metadata.
                assert set(legal)<=expected
                assert len(legal) in ([n,n+1] if r['pos']==3 else [n+1])
            key=(n,r['trump'],r['pos'])
            cells[key]=cells.get(key,0)+1
            for name,p in r['policies'].items():
                laws={int(b):np.asarray(d) for b,d in p['laws'].items()}
                assert set(laws)==set(legal)
                for d in laws.values():
                    assert len(d)==n+1 and np.all(d>=0)
                    np.testing.assert_allclose(d.sum(),1,atol=1e-12)
                    np.testing.assert_allclose(d*300,np.rint(d*300),atol=1e-10)
                best=max(legal,key=lambda b:((10+b)*laws[b][b],-b))
                law=laws[best];mu=float(np.arange(n+1)@law)
                assert p['b']==best and p['mode']==int(law.argmax())
                assert abs(p['mu']-mu)<1e-12
                assert p['floor']==int(np.floor(mu)) and p['ceil']==int(np.ceil(mu))
                assert p['round']==int(np.floor(mu+.5))
                if name=='greedy':
                    assert all(np.array_equal(law,d) for d in laws.values())
            count+=1
    assert count==4160 and len(cells)==104 and set(cells.values())=={40}
    checks[str(seed)]={'situations':count,'cells':len(cells),'situations_per_cell':40,
                       'laws_counts_oracles_and_saved_moments_valid':True,
                       'rows_with_context_for_full_hook_audit':context_rows,
                       'limitation':'Legacy discovery prefix lacks hand/prior metadata; only legal-set shape checked there.' if context_rows<count else None}
(HERE/'results/verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
