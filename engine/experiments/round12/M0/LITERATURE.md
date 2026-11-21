# Bounded primary-source map — inspected 2026-11-17

This is a positioning audit, not a completed novelty review. A workshop-quality
target requires a precise question, defensible controls, reusable artifacts and
honest limits; it does not imply acceptance or priority. No submission made.

| Primary work | Inspected scope and connection | Boundary for this project |
|---|---|---|
| [Cai, Liu, Oikonomou, Zhang, NeurIPS2024: Provable Partially Observable Reinforcement Learning with Privileged Information](https://proceedings.neurips.cc/paper_files/paper/2024/hash/74d188c51d97fcfbc0269f584d6a53b7-Abstract-Conference.html) | Official abstract and introduction of the linked paper. Privileged distillation can fail; positive guarantees require structural assumptions; belief-weighted asymmetric methods studied. | Privilege-induced partial-observability failure is established prior work. No transfer of their POMDP assumptions or guarantees to our card benchmark. |
| [Song, Rohatgi, Singh, Bagnell, NeurIPS2025: To Distill or Decide?](https://proceedings.neurips.cc/paper_files/paper/2025/hash/81343f8ebe529de8d0bb654af7523184-Abstract-Conference.html) | Official abstract only; large PDF fetch failed. Compares distillation and ordinary RL under partial observability/privilege. | Closely related motivation; full technical comparison still needed. No claim that our benchmark resolves their algorithmic trade-off. |
| [Elmachtoub and Grigas: Smart Predict, then Optimize](https://arxiv.org/abs/1710.08005) | Abstract/source record; decision cost rather than prediction error, SPO/SPO+ framework. | Our direct finite-action risk is not a newly invented decision-focused objective or an implementation of SPO+. No inherited consistency theorem. |
| [Rusu et al.: Policy Distillation](https://arxiv.org/abs/1511.06295) | Abstract; compressing and combining RL policies. | Teacher/student imitation is established. Different information sets and exact lower certificates are the particular focus here. |
| [Cour, Sapp, Taskar, JMLR2011: Learning from Partial Labels](https://jmlr.org/papers/v12/cour11a.html) | Official abstract; candidate label sets with one correct label. | Related ambiguity-aware supervision. Here several actions can genuinely be optimal; semantics and consistency assumptions differ. |
| [Chen et al., CoRL2019/PMLR2020: Learning by Cheating](https://proceedings.mlr.press/v100/chen20a.html) | Official abstract and initial introduction. Privileged scene teacher distills to visual driving student. | Established privileged-teacher engineering paradigm, not a card-game guarantee. |
| [Foerster et al., AAAI2018: Counterfactual Multi-Agent Policy Gradients](https://ojs.aaai.org/index.php/AAAI/article/view/11794) | Official abstract. Centralized critic/decentralized actors; counterfactual credit baseline. | Our root-action exact Q supervision is not COMA and supplies no new multi-agent credit-assignment theorem. |
| [Solinas et al., NeurIPS2023, direct Oh Hell/MCMC precedent](https://proceedings.neurips.cc/paper_files/paper/2023/hash/87ee1bbac4635e7c948f3eea83c1f262-Abstract-Conference.html) | Existing round10 primary-source map retained; direct game overlap. | Never claim first AI, ML or MCMC for Oh Hell. D5 is a completed primary null with open external proof gate. |

## Working contribution hypothesis, pending results and external review

A compact, exactly enumerable privileged-teacher/private-student benchmark with
zero-gap controls, strictly positive certified information floors, observable
hand-group splits, explicit teacher ties and decomposed student regret. Pair it
with a separate, genuine52-card deployable residual bidder experiment. The toy
root relaxation does not establish full-team feasibility or a52-card gap.

Theoretical decomposition is elementary finite decision theory; the binary-payoff
tie-label observations need external audit and related loss literature review.
Potential empirical value is knowing exactly which part of error is information,
representation, supervision or finite fitting, rather than claiming a new general
RL algorithm. Null learner comparisons and failed full-game improvements are
legitimate outcomes. Deep architectures alone do not make this workshop-ready.

## Required before a submission

Read the closest2024/2025 papers in full, broaden ambiguous/set-valued label and
decision-focused distillation references, obtain external proof/implementation
review, test artifact reproduction in a clean public checkout, and identify an
appropriate live workshop call. No current deadline or suitability is assumed.
