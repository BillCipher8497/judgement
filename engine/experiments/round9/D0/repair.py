"""Remove duplicated handoff summaries and make the old updater idempotent."""
from pathlib import Path
HERE=Path(__file__).resolve().parent;ENGINE=HERE.parents[2];PROJECT=ENGINE.parent
root=PROJECT/'AGENTS.md';inner=ENGINE/'AGENTS.md'
doc=root.read_text(encoding='utf-8');assert doc==inner.read_text(encoding='utf-8')
for marker in [' Top pool (3×Hybrid-BI K32c6),',' Round 8: none of 143']:
    lines=doc.splitlines()
    for i,line in enumerate(lines):
        if marker in line:
            parts=line.split(marker)
            if len(parts)>2:
                assert len(set(parts[1:]))==1,'Repeated summaries must match exactly'
                lines[i]=marker.join(parts[:2])
    doc='\n'.join(lines)+'\n'
root.write_text(doc,encoding='utf-8');inner.write_text(doc,encoding='utf-8')
old=ENGINE/'experiments/round8/T7/handoff.py'
source=old.read_text(encoding='utf-8')
source=source.replace("line.split(' Top-pool equal-depth')[0]+", "line.split(' Top-pool equal-depth')[0].split(' Top pool (3×Hybrid-BI K32c6),')[0]+")
source=source.replace("lines[i]=line+' Round 8: none of 143", "lines[i]=line.split(' Round 8: none of 143')[0]+' Round 8: none of 143")
old.write_text(source,encoding='utf-8')
assert root.read_bytes()==inner.read_bytes()
assert doc.count(' Top pool (3×Hybrid-BI K32c6),')==1
assert doc.count(' Round 8: none of 143')==1
print('Both handoffs identical, each requested summary occurs once; old updater guarded.')
