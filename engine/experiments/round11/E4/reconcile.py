"""Apply the external-audit wording corrections without changing numeric evidence."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[2]
ROOT = ENGINE.parent


def replace(path, before, after):
    text = path.read_text(encoding='utf-8-sig')
    if after in text and before not in text: return
    assert before in text, f'Expected old wording missing: {path}: {before[:70]}'
    path.write_text(text.replace(before, after), encoding='utf-8')


def main():
    paper = ENGINE/'paper/MANUSCRIPT_DRAFT.md'
    replace(paper,
        'Round9 additions are separately self-validated by Codex, the same investigator who implemented them.',
        'D1/D2 and the M13/M14 proofs/lower-bound enumeration are externally validated (Claude, 2026-11-16); other additions retain their separate validation labels. The fresh E0 full-suite gate passed locally; external validation did not include a test-suite rerun.')
    replace(paper,
        'The round9 information intervention and larger-toy bound remain in the externally pending appendix below. They are not used to strengthen this section\'s causal or optimality claims.',
        'The externally audited round9 information intervention and larger-toy bound are detailed in Appendix A. D1 supports the implemented information intervention only; M13 certifies a strict toy information cost, with no material-magnitude claim or solved hidden-team optimum.')
    replace(paper, 'Round9 separately self-validated analysis holds guide play fixed',
                  'Round9 analysis, independently recomputed and externally validated by Claude, holds guide play fixed')
    replace(paper,
        'Before submission, extend the bounded primary-source related-work audit, complete external D1/M13 review, and perform a claim-by-claim reproduction pass.',
        'Before submission, extend the bounded primary-source related-work audit and perform a claim-by-claim reproduction pass. External D1/M13/M14 review is complete; the distinct feasible-upper forward replay is tracked as E5.')
    replace(paper,
        'The particle-MCMC experiment is unfinished and supplies no result to this draft.',
        'The particle-MCMC experiment supplies no validated improvement to this draft and remains outside the publication scope.')
    replace(paper, '## Appendix A. Externally pending round9 material',
                  '## Appendix A. Externally audited information intervention and finite-toy bounds')
    replace(paper,
        '**Audit gate:** these are separately self-validated (Codex), not externally validated. No strong information-causality claim is made. M13 remains **proved (pending external audit)**.',
        '**External status:** D1 and the M13/M14 proofs/lower-bound enumeration were independently checked and externally validated by Claude on 2026-11-16. Feasible upper policies have a separate forward-replay check in E5; Claude checked their ordering only. Validation is limited to the stated interventions and finite games.')
    replace(paper,
        'Scope is n=5, equal S/NT and the implemented policies. [C10]',
        'The supported causal statement is only that revealing the hero hand to this implemented sampler changes its punishment performance: n=5, equal S/NT, fixed TA+v2 hero, fixed bids and K16 per active decision. It does not identify a policy-independent information effect, optimal coalition punishment, human performance or a general effect across the full 25-round ladder using the standard 52-card deck. [C10]')
    replace(paper,
        'New proof and full reconstruction require external audit; review reused derived tree caches plus separate numerical/manual checks. No 52-card extrapolation. [C11]',
        'Claude independently reconstructed every deal and exactly reproduced the PI/J1 lower values, and rechecked M13. The certified J1−PI gaps are 25819/369600 ≈ 0.070 and 421/30800 ≈ 0.014 hero points, respectively. They certify a strict information cost and nothing about its magnitude: these lower bounds do not establish a material cost, nor estimate the actual hidden-team optimum minus PI. No 52-card extrapolation. [C11]\n\nM14, also externally validated, shows that team randomization cannot improve the global minimum against fixed non-team policies: conditioning on tapes gives a mixture of feasible pure private-information tuples, whose mean cost cannot beat the best tuple. Team tapes may be shared, but must be independent of chance and non-team tapes. The fixed-hero premise and full own/public histories are essential; fixing the team tapes must not change hero observations outside those histories. This is not a strategic-adversary team-maxmin or CFR convergence theorem. No novelty claim is made.')
    references = '\n- External audit: [Claude verdicts and retained limits](../../CLAUDE_AUDIT_2026-11-16.md); [M14 proof](../experiments/round10/R3/PROOF.md).\n'
    text = paper.read_text(encoding='utf-8')
    if references not in text: paper.write_text(text+references, encoding='utf-8')

    evidence = ENGINE/'paper/EVIDENCE.md'
    replace(evidence,
        'These are working drafts, with broader literature/novelty review and external D1/M13 audit still pending. M14\'s separate fixed-hero pure-team clarification is proved(pending external audit), not a new observed card-game value. R4 is an unexecuted future NT-bidder protocol with unused seed reservations; it supplies no performance evidence.',
        'These remain working drafts with broader literature/novelty review outstanding. Claude externally validated D1/D2 and M13/M14 on2026-11-16; M14 is a fixed-hero pure-team clarification, not a new observed card-game value. R4 execution and any resource shortfall are tracked in the local round11/E3 packet; no unexecuted or unvalidated stage supplies performance evidence. E0 freshly passed262tests/1optional OpenSpiel skip (49.07s pytest,53.910273s startup-inclusive output interval).')
    replace(evidence, '## Round 9 — separately self-validated (Codex), external audit pending',
                      '## Round 9 — separate self-review and completed external-audit updates')
    replace(evidence,
        'Codex is now head of research. These additions passed separate reviews in `experiments/round9/REVIEW.md`; the implementer and reviewer are the same investigator. Fresh raw-data/evaluator scripts and audit packets are under `round9/reviews/`. This is not external validation. **Do not use D1 for strong causal paper language before external review.** M13 remains **proved (pending external audit)**. No unfinished D5 result is included.',
        'Codex is head of research. The original separate reviews in `experiments/round9/REVIEW.md` were by the same investigator; Claude subsequently independently recomputed D1/D2 and rebuilt the complete D3 lower-bound enumeration, and externally validated M13/M14 (root CLAUDE_AUDIT_2026-11-16.md). D1 wording is confined to its audited intervention. D4 retains its original self-review label. Feasible-upper forward replay is a separate E5 gate. D5 remains outside this evidence map/publication scope.')
    replace(evidence,
        'Separate self-review is complete; causal headline wording still awaits external D1 audit.',
        'Claude independently recomputed and externally validated the intervention. Causal wording is limited to revealing the hero hand to this particular K16 sampler at n5 with fixed TA+v2 hero/bids; it does not generalize to other heroes, optimal coalitions, the full 25-round ladder or humans. K is fixed per active decision, not total wall time.')
    replace(evidence, 'conditional on M13\'s pending external audit.',
                      'with M13\'s proof/lower bounds externally validated by Claude; the feasible-upper forward replay is separately tracked as E5.')
    replace(evidence,
        'Integer J1−PI gaps are25819/369600 and421/30800.',
        'Integer J1−PI gaps are25819/369600 ≈ 0.070 and421/30800 ≈ 0.014 hero points. They certify a strict information cost and nothing about its magnitude; they neither establish a material effect nor estimate the actual optimal hidden-team cost minus PI.')
    replace(evidence, 'The proof is **proved (pending external audit)**.',
                      'The proof is **proved and externally validated (Claude,2026-11-16)**.')
    replace(evidence,
        'External full reconstruction remains requested.',
        'Claude subsequently independently reconstructed all369600deals/regime and reproduced exact PI/J1 values; Claude checked only the feasible uppers\' ordering, not their replay. E5 addresses that distinct numerical check.')

    notes = ENGINE/'math/NOTES.md'
    replace(notes, '## M13 First-decision private-information relaxation (round 9 candidate)',
                   '## M13 First-decision private-information relaxation (round 9)')
    replace(notes, '**Exact enumeration awaiting external validation:**',
                   '**Exact enumeration externally validated (Claude,2026-11-16):**')
    replace(notes,
        'These are integer full-population sums, so sampling CIs do not apply. If the\nproof and implementation pass audit, the positive differences certify a strict\ninformation cost in these toys, without relying on CFR convergence. They do\nnot quantify the 52-card information cost.',
        'These are integer full-population sums, so sampling CIs do not apply. The\ncertified gaps are approximately 0.070 (trump) and 0.014 (NT) hero points. They\ncertify a strict information cost and nothing about its magnitude: these lower\nbounds neither establish a material cost nor estimate the actual hidden-team\noptimum minus PI. No CFR convergence or 52-card information-cost claim follows.')
    replace(notes,
        'independent full construction remains a useful external-audit step.',
        'Claude subsequently independently reconstructed the complete lower-bound populations; feasible-upper forward replay is separately tracked as round11/E5.')
    readme = ENGINE/'experiments/round9/D3/README.md'
    replace(readme, 'The first-decision proof remains pending external audit.',
                    'The first-decision proof and exact PI/J1 enumeration are now externally validated by Claude (2026-11-16).')

    ledger = ENGINE/'experiments/round10/R0/claim_ledger.json'
    data = json.loads(ledger.read_text(encoding='utf-8'))
    data['publication_reconciliation'] = {
        'historical_original_public_groups': ['C%02d' % i for i in range(1, 10)],
        'current_cleared_groups': ['C%02d' % i for i in range(1, 12)],
        'newly_cleared': {'C10': 'audited D1 intervention only', 'C11': 'M13 proof/lower bounds externally validated; feasible-upper forward replay E5'},
        'local_only_claim_groups_after_reconciliation': [],
        'withheld_research': 'D5; completed R4/E3 packet pending publication/reproduction cross-check; mixed private history. None is another manuscript quantitative group',
        'why_original_difference': 'C10 strong D1 wording and C11 new proof were externally gated at original9-group publication; those gates are now cleared, not silently lost or relabelled'
    }
    data['mandatory_caveats'] = ['particle-MCMC experiment supplies no validated improvement'
                               if v == 'particle-MCMC experiment is unfinished' else v
                               for v in data['mandatory_caveats']]
    for claim in data['claims']:
        if claim['id'] == 'C09':
            claim['status'] = 'externally validated (Claude,2026-11-16), whole-bidder intervention only; multiplicity caveat retained'
        if claim['id'] in ['C09', 'C10', 'C11']:
            if '../CLAUDE_AUDIT_2026-11-16.md' not in claim['primary_sources']:
                claim['primary_sources'].append('../CLAUDE_AUDIT_2026-11-16.md')
    ledger.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


if __name__ == '__main__': main()
