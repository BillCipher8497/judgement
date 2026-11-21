# Targeted follow-up, 2026-11-17

The original bounded map remains a checkpoint, not the end of the review.
Additional primary sources materially narrow the positioning:

* [Ross and Bagnell2014, Reinforcement and Imitation Learning via Interactive
  No-Regret Learning](https://arxiv.org/abs/1406.5979): abstract inspected.
  Cost-sensitive imitation using action cost-to-go is established. Our exact
  finite-action supervision is not a new use of oracle action values.
* [Sun et al., ICML2017, Deeply AggreVaTeD](https://proceedings.mlr.press/v70/sun17d.html):
  official abstract and paper introduction inspected. Differentiable neural
  policies trained with cost-to-go oracles, including privileged training
  information, directly precede our direct-cost arm. No claim of introducing
  neural cost distillation or inheriting their sequential guarantees.
* [Warrington et al., ICML2021, Robust Asymmetric Learning in POMDPs](https://proceedings.mlr.press/v139/warrington21a.html):
  official abstract and introduction inspected. Expert/trainee information
  mismatch and adapting an expert to the trainee are explicit prior work.
  Our teacher stays fixed; we audit tie conventions under exact conditional risk.
* [Swamy et al., NeurIPS2022, Sequence Model Imitation Learning with Unobserved
  Contexts](https://proceedings.neurips.cc/paper_files/paper/2022/file/708e58b0b99e3e62d42022b4564bad7a-Paper-Conference.pdf):
  abstract/initial introduction inspected. On-policy recovery and offline
  action-history latching are distinct sequential issues. The first-free-root
  task has no student-induced pre-root occupancy shift; it does not test their
  on-policy/off-policy theorem or replace a sequential benchmark.
* [Chang et al., ICML2015, Learning to Search Better than Your Teacher](https://proceedings.mlr.press/v37/changb15.html):
  official abstract inspected. LOLS/local deviation guarantees exist. Our
  fixed-policy toy bounds give no learned full-team local/global optimality.
* [Walsman et al., ICLR2023, Impossibly Good Experts and How to Follow Them](https://openreview.net/forum?id=sciA_xgYofB):
  accepted primary PDF abstract/introduction inspected. Privileged experts can
  exceed every executable policy, and naive imitation can underperform the best
  reduced-information policy. This is a close conceptual antecedent; no claim
  of discovering the distinction between information and imitation error.

Read Cai2024§3.1/Proposition3.1 and deterministic-filter definition as well as
the initial sections: even one-step expert distillation failure is established.
Read targeted Song2025HTML related-work/conditional-belief passages after the
official large PDF fetch failed; [author arXivHTML](https://arxiv.org/html/2510.03207v1)
is accessible. Their full proof/experiment comparison is still incomplete.
No full-paper review is implied by these targeted reads.

Multiple genuinely optimal demonstrations are also established in robotics:
[Oh et al.,2022preprint/NeuralNetworks2023, Bayesian Disturbance Injection](https://arxiv.org/abs/2211.03393),
abstract inspected. Their variational multimodal/risk-sensitive algorithm is
different from our exact categorical-tie risk audit; no general multi-optimal
imitation novelty claim. A complete set-valued decision-loss review is still open.

The defensible possible contribution is a **certified, reproducible card-game
audit benchmark** with exact information floors and explicit tie-dependent risk,
plus controlled finite learners and a separate deployable full-game experiment.
Whether this merits a workshop depends on external audit, artifact quality and
comparison with existing benchmarks, not the mere use of neural networks.
