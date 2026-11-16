"""Check final documentation provenance without rerunning simulations."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ENGINE=HERE.parents[2];PROJECT=ENGINE.parent
manifest=json.loads((HERE/'results/evidence_sources.json').read_text(encoding='utf-8'))
for path,digest in manifest['sources_sha256'].items():
    assert hashlib.sha256((ENGINE/path).read_bytes()).hexdigest()==digest,path
evidence=ENGINE/'paper/EVIDENCE.md'
assert hashlib.sha256(evidence.read_bytes()).hexdigest()==manifest['evidence_sha256']
assert (PROJECT/'AGENTS.md').read_bytes()==(ENGINE/'AGENTS.md').read_bytes()
text=evidence.read_text(encoding='utf-8')
pool=json.loads((HERE.parent/'T2/results/summary.json').read_text())
for key in ['rb6','rb7','rb7_minus_rb6']:
    point,half=pool[key]['score_mean_half95']
    formatted=f'{point:+.2f} ± {half:.2f}'.replace('-', '−').replace('+-','−')
    assert formatted in text,(key,formatted)
assert '256 passed, 1 skipped' in (HERE/'results/final_pytest.txt').read_text()
out={'source_hashes_checked':len(manifest['sources_sha256']),
     'both_AGENTS_identical':True,'evidence_hash_and_top_pool_numbers_match':True,
     'final_suite_log':'256 passed, 1 optional OpenSpiel skip',
     'output_sha256':{str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [evidence,PROJECT/'AGENTS.md',ENGINE/'AGENTS.md']}}
(HERE/'results/document_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
