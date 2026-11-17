"""Integrate only the passed exact replay, preserving external attribution."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ENGINE = HERE.parents[2]


def change(path, before, after):
    text = path.read_text(encoding='utf-8-sig')
    if after in text and before not in text: return
    assert before in text, (str(path), before)
    path.write_text(text.replace(before, after), encoding='utf-8')


def main():
    verified = json.loads((HERE/'results/verification.json').read_text())
    assert verified['pass'] and all(x['pass'] and x['n_exact_deals'] == 369600 for x in verified['checks'].values())
    notes = ENGINE/'math/NOTES.md'
    change(notes, 'Feasible upper scores were not independently\nreplayed; only their ordering against J1 was checked.',
           'Claude checked only the feasible uppers\' ordering against J1.\nRound11/E5 subsequently confirmed both saved uppers through a fresh forward\nreplay, separately self-validated (Codex), not an external replay verdict.')
    change(notes, 'feasible-upper forward replay is separately tracked as round11/E5.',
           'round11/E5 separately confirms the feasible uppers by fresh forward replay, with unchanged fixed hero/bidder definitions and no optimizer.')
    addition = ('\nRound11/E5 fresh forward replay gives integer hero-score sums2296988(trump)\n'
                'and2633016(NT), each over all369600ordered deals, hence exactly574247/92400\n'
                'and109709/15400. Portable private-key policies and per-deal score vectors are\n'
                'saved under experiments/round11/E5/results/. Independent integer-vector,\n'
                'checkpoint/source and map checks pass. This is same-investigator\n'
                'self-validation, not Claude replaying the uppers, extra optimization, a\n'
                'solved hidden optimum or a sampling confidence interval.\n\n')
    text = notes.read_text(encoding='utf-8')
    if addition not in text: notes.write_text(text.replace('## M14 ', addition+'## M14 ', 1), encoding='utf-8')
    d3 = ENGINE/'experiments/round9/D3'
    change(d3/'ROOT_BOUND.md', 'Feasible upper-policy independent replay remains task E5.',
           'Feasible uppers were separately confirmed by round11/E5 fresh forward replay, self-validated (Codex); Claude did not replay them.')
    change(d3/'README.md', 'Feasible upper-policy independent replay remains E5; original Codex replay is retained separately.',
           'Round11/E5 now independently forward-replays the feasible uppers on all369600deals/regime:574247/92400 and109709/15400, with portable private policies and raw score vectors. This is separate self-validation(Codex), not an external upper-replay verdict; the original cached replay is retained.')
    paper = ENGINE/'paper/MANUSCRIPT_DRAFT.md'
    change(paper, 'External D1/M13/M14 review is complete; the distinct feasible-upper forward replay is tracked as E5.',
           'External D1/M13/M14 review is complete; a fresh feasible-upper forward replay also passed as a separate Codex self-check (E5).')
    change(paper, 'Feasible upper policies have a separate forward-replay check in E5; Claude checked their ordering only.',
           'The feasible uppers were also confirmed by a fresh forward replay in E5, self-validated (Codex); Claude checked their ordering only.')
    change(paper, 'upper endpoints are integer-replayed feasible policies.',
           'upper endpoints are feasible saved policies, confirmed by a fresh forward replay in E5 (self-validated, Codex).')
    evidence = ENGINE/'paper/EVIDENCE.md'
    change(evidence, 'Feasible-upper forward replay is a separate E5 gate.',
           'Fresh feasible-upper forward replay has now passed separately in E5, self-validated (Codex).')
    change(evidence, 'the feasible-upper forward replay is separately tracked as E5.',
           'the feasible uppers are separately confirmed by E5 fresh forward replay, self-validated (Codex).')
    change(evidence, 'E5 addresses that distinct numerical check.',
           'E5 subsequently completed that distinct numerical check: all369600deals/regime, exact integer score vectors reproduce574247/92400 and109709/15400, with portable private policies. This is same-investigator self-validation, not a new Claude upper-replay verdict.')
    ledger_path = ENGINE/'experiments/round10/R0/claim_ledger.json'
    ledger = json.loads(ledger_path.read_text(encoding='utf-8-sig'))
    for claim in ledger['claims']:
        if claim['id'] != 'C11': continue
        claim['status'] = 'M13 proof/lower bounds externally validated(Claude,2026-11-16); feasible uppers freshly forward-replayed and self-validated(Codex,round11/E5)'
        for p in ['experiments/round11/E5/results/replay_0.json', 'experiments/round11/E5/results/replay_-1.json', 'experiments/round11/E5/results/verification.json']:
            if p not in claim['primary_sources']: claim['primary_sources'].append(p)
    ledger['publication_reconciliation']['newly_cleared']['C11'] = 'M13 proof/lower bounds externalClaude; feasible uppers fresh forward replay self-validatedCodex(E5), not externally replayed'
    ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


if __name__ == '__main__': main()
