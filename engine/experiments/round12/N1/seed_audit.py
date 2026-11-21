"""Registered seed inventory and entropy-state collision check, not game data."""
import json,re,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[3];DOMAIN=0x4E31
def audit():
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    hits=[];pattern=re.compile(r'(?<![\w.])(?:21[0-6]0000000|21[0-6]0M)(?![\w.])',re.I)
    for rel in paths:
        p=ROOT/rel
        if not p.is_file() or p.suffix not in ('.py','.md','.txt','.sh','.ps1') or p.stat().st_size>2_000_000:continue
        try:text=p.read_text(encoding='utf-8-sig').replace('_','')
        except UnicodeError:continue
        for line,s in enumerate(text.splitlines(),1):
            if pattern.search(s):hits.append({'path':rel,'line':line,'text':s[:350]})
    allowed=('round12/','CLAUDE_AUDIT_2026-11-17.md','AGENTS.md','LOCAL_REVIEW_LOGBOOK.md','HANDOFF_CODEX.md')
    unexpected=[h for h in hits if not any(x in h['path'].replace('\\','/') for x in allowed)]
    assert not unexpected,('Inspect prior seed registrations',unexpected)
    states={};count=0
    def add(address,name):
        nonlocal count
        value=np.random.SeedSequence(address).generate_state(4).tobytes()
        assert value not in states,('Entropy-state collision',name,states.get(value))
        states[value]=name;count+=1
    for g in range(6000):add([2100000000,g],f'N0/train/{g}')
    for g in range(600):add([2100000000,10000000+g],f'N0/test/{g}')
    for stage,base,n in ((0,2100000000,176),(1,2100000000,30800),(2,2100000000,4400),(3,2110000000,3520)):
        for g in range(n):
            for stream in (3,4,5):add([base,DOMAIN,stage,g,stream],f'N1/{stage}/{g}/{stream}')
    for seed in range(5):
        add([2100000000,DOMAIN,20,seed],f'N1/init/{seed}')
        for epoch in range(1,51):add([2100000000,DOMAIN,21,seed,epoch],f'N1/batch/{seed}/{epoch}')
    add([2110000000,DOMAIN,90],'N1/calibration-bootstrap')
    result={'utc':datetime.now(timezone.utc).isoformat(),'reserved_bases_M':list(range(2100,2170,10)),
        'registered_text_hits':hits,'unexpected':unexpected,'checked_distinct128bit_root_states':count,
        'collisions':0,'N0_untouched':'N0 two-integer streams checked; N1 has explicit disjoint namespace/stage/tail.',
        'limitation':'Tracked protocol/source text<=2MB, not a claim about unknowable external uses; no game labels generated.'}
    out=H/'preflight';out.mkdir(exist_ok=True);(out/'seed_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
