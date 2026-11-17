"""Separate source/scope reconciliation audit; not a numerical reanalysis."""
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[2]
ROOT = ENGINE.parent


def main():
    ledger_path = ENGINE/'experiments/round10/R0/claim_ledger.json'
    ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
    paper_path = ENGINE/'paper/MANUSCRIPT_DRAFT.md'
    paper = paper_path.read_text(encoding='utf-8')
    evidence_path = ENGINE/ledger['evidence_source']
    evidence = evidence_path.read_text(encoding='utf-8')
    notes = (ENGINE/'math/NOTES.md').read_text(encoding='utf-8')
    original = json.loads(subprocess.check_output([
        'git', 'show', 'a99c9577334719a04e1b6d6484c158081a71f17d:engine/experiments/round10/R0/claim_ledger.json'], cwd=ROOT))
    old_ids = [c['id'] for c in original['claims']]
    ids = [c['id'] for c in ledger['claims']]
    assert old_ids == ['C%02d' % i for i in range(1, 10)]
    assert ids == ['C%02d' % i for i in range(1, 12)]
    assert set(ids)-set(old_ids) == {'C10', 'C11'}
    assert ledger['publication_reconciliation']['local_only_claim_groups_after_reconciliation'] == []
    assert ledger['external_gate_ids'] == []
    assert 'pending external audit' not in paper
    assert 'externally pending' not in paper.lower()
    assert 'strict information cost and nothing about its magnitude' in paper
    assert 'strict information cost and nothing about its magnitude' in evidence
    assert '0.070' in paper and '0.014' in paper and 'material cost' in paper
    assert 'fixed TA+v2 hero' in paper and 'K16 per active decision' in paper
    assert 'does not identify a policy-independent information effect' in paper
    assert 'not a strategic-adversary team-maxmin' in paper
    for section in ['M13', 'M14']:
        text = notes.split('## '+section+' ', 1)[1].split('\n## ', 1)[0]
        assert 'proved and externally validated (Claude, 2026-11-16)' in text
        assert 'awaiting external validation' not in text
    checks, hashes = [], {}
    for claim in ledger['claims']:
        assert '['+claim['id']+']' in paper
        for text in claim.get('draft_strings', []): assert text in paper, (claim['id'], text)
        for text in claim.get('evidence_strings', []): assert text in evidence, (claim['id'], text)
        primary = ''
        for rel in claim['primary_sources']:
            path = ENGINE/rel
            assert path.exists(), rel
            raw = path.read_bytes(); hashes[rel] = hashlib.sha256(raw).hexdigest()
            primary += raw.decode('utf-8-sig')
        for text in claim.get('primary_strings', []): assert text in primary, (claim['id'], text)
        checks.append({'id': claim['id'], 'pass': True, 'status': claim['status']})
    for caveat in ledger['mandatory_caveats']: assert caveat in paper, caveat
    links = []
    for target in re.findall(r'\]\(([^)]+)\)', paper):
        if target.startswith(('https://', 'http://')): continue
        assert (paper_path.parent/target).exists(), target
        links.append(target)
    # Exact arithmetic/scope checks, not new empirical intervals.
    assert Fraction(446669, 73920)-Fraction(367921, 61600) == Fraction(25819, 369600)
    assert Fraction(213259, 30800)-Fraction(106419, 15400) == Fraction(421, 30800)
    assert round(float(Fraction(25819, 369600)), 3) == .070
    assert round(float(Fraction(421, 30800)), 3) == .014
    e5_verified = ENGINE/'experiments/round11/E5/results/verification.json'
    upper_done = e5_verified.exists() and json.loads(e5_verified.read_text())['pass']
    result = {'pass': True, 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'claim_count': len(checks), 'checks': checks,
              'historical_public_ids': old_ids, 'newly_cleared': ['C10', 'C11'],
              'current_local_only_quantitative_groups': [],
              'held': 'D5; R4/E3 packet pending publication/reproduction cross-check; mixed private history',
              'upper_replay': 'E5 fresh forward replay passed, self-validated(Codex); not externally replayed by Claude' if upper_done else 'separate E5 gate; not externally replayed by Claude',
              'links_checked': links, 'primary_source_sha256': hashes,
              'document_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [paper_path, evidence_path, ledger_path, ENGINE/'math/NOTES.md']},
              'limits': 'Source/citation/status/wording reconciliation only; no new raw empirical estimate or external audit'}
    (HERE/'results').mkdir(exist_ok=True)
    (HERE/'results/verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'pass': True, 'claim_count': len(checks), 'newly_cleared': ['C10', 'C11']}))


if __name__ == '__main__': main()
