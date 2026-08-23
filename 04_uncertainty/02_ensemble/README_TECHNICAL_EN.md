# Probabilistic Ensemble: Epistemic and Aleatoric Uncertainty

Status: completed on 2026-08-22. This is a small PETS-inspired bootstrap ensemble and trajectory-sampling experiment, not a full PETS reproduction.

## Purpose

Add model disagreement to the probabilistic transition head so uncertainty can be decomposed into expected per-model noise (aleatoric) and disagreement among model means (epistemic). Implement PETS-style particle propagation with fixed or resampled model identities.

## Problem

A single probabilistic network can predict transition noise, but outside its training support it has no reliable signal that its parameters are underdetermined. Its variance head may incorrectly absorb model error as aleatoric noise. Multiple independently trained models provide a measurable disagreement signal.

## Previous Model

`01_probabilistic_dynamics` learned a diagonal Gaussian with strong in-distribution calibration. Its predicted standard deviation diverged from the known noise curve outside `x in [-0.8,0.8]`, motivating a separate epistemic mechanism.

## Hypothesis

Bootstrap-trained models should agree in dense training regions and disagree more outside them. The ensemble moment decomposition should retain learned aleatoric noise while exposing extra epistemic variance. Particle rollout should preserve model hypotheses differently under TS∞ and TS1.

## Architecture

```text
same (s_t,a_t)
   |       |       |       |       |
 Model 0 Model 1 Model 2 Model 3 Model 4
 bootstrap datasets; independent initialization/Adam
   |       |       |       |       |
 (mu_0,var_0) ...                 (mu_4,var_4)
             |
 mean prediction       = mean_m(mu_m)
 aleatoric variance    = mean_m(var_m)
 epistemic variance    = variance_m(mu_m)
 total variance        = aleatoric + epistemic
```

Each member is the bounded diagonal-Gaussian model from the previous experiment. No member or earlier folder is overwritten.

## Data Flow

```text
base transition dataset
  -> five bootstrap index samples (with replacement)
  -> independent member training by Gaussian NLL
  -> joint prediction and variance decomposition
  -> moment metrics / OOD map
  -> particle rollout:
       TS-infinity: one member per particle for entire trajectory
       TS1:         resample member identity at every step
```

## Tensor Shapes

For ensemble `E=5`, batch `B`, particles `P=128`, horizon `T=12`, state `D_s=2`:

| Tensor | Shape | Meaning |
|---|---|---|
| states/actions | `[B,2]`, `[B,4]` | shared model inputs |
| member means/variances | `[E,B,2]` | five Gaussian predictions |
| ensemble mean | `[B,2]` | mean of member means |
| aleatoric/epistemic/total variance | `[B,2]` | moment decomposition |
| rollout particles | `[B,P,T,2]` | sampled trajectories |
| selected model IDs | `[B,P,T]` | member assignment history |
| bootstrap indices | `[E,N]` | resampled training rows |

## Mathematics

For ensemble member `m`:

```text
p_m(y|x)=N(mu_m(x),Sigma_m(x)).
```

The law of total variance gives the implemented moment decomposition:

```text
mu_bar = (1/E) sum_m mu_m
Sigma_aleatoric = (1/E) sum_m Sigma_m
Sigma_epistemic = (1/E) sum_m (mu_m-mu_bar)^2
Sigma_total = Sigma_aleatoric + Sigma_epistemic.
```

The first term is average within-model variance; the second is between-model mean disagreement. This is a finite-ensemble approximation, not an exact Bayesian posterior.

Bootstrap member `m` trains on `N` indices drawn with replacement from the original `N` rows. Different data multiplicities plus initialization create diverse functions compatible with available evidence.

## Code Mapping

| Concept | File / symbol |
|---|---|
| bootstrap resampling | `ensemble_dataset.py::bootstrap_indices` |
| five probabilistic members | `probabilistic_ensemble.py::ProbabilisticEnsemble` |
| variance decomposition | `ProbabilisticEnsemble.decompose` |
| TS∞ / TS1 propagation | `ProbabilisticEnsemble.rollout` |
| independent optimizers | `train.py::train` |
| ID/OOD metrics | `evaluate.py::evaluate` |
| epistemic heatmap | `outputs/epistemic_map.png` generation in `evaluate.py` |
| reused Gaussian/NLL core | `../01_probabilistic_dynamics/probabilistic_dynamics.py`, `probabilistic_losses.py` |

## Training

```bash
.venv/bin/python 04_uncertainty/02_ensemble/train.py
.venv/bin/python 04_uncertainty/02_ensemble/evaluate.py
.venv/bin/python -m pytest -q \
  04_uncertainty/01_probabilistic_dynamics/tests \
  04_uncertainty/02_ensemble/tests
```

| Reproducibility item | Value |
|---|---|
| seed / bootstrap seed | 41 / 42 |
| dataset | `heteroscedastic-point-v1`, 1024 train / 256 validation |
| ensemble | 5 independent 4,872-parameter Gaussian MLPs |
| total parameters | 24,360 |
| optimizer | independent Adam per member, `1e-3` |
| epochs / steps per member | 60 / 960 |
| checkpoint | format 1, gitignored |
| evaluation | `python 04_uncertainty/02_ensemble/evaluate.py` |

## Losses

Each member independently uses the previous experiment's Gaussian NLL and variance-bound regularizer. There is no explicit diversity loss: diversity comes from bootstrap samples and initialization. Ensemble uncertainty is computed at evaluation/rollout time, not added as a supervised target.

## Evaluation Interface

Evaluation reports moment-matched NLL, mean RMSE, total-variance coverage, aleatoric correlation to known noise, ID/OOD epistemic std and ratio, particles/horizon, propagation modes, and parameter count. OOD states have `|x| in [1.1,1.5]`, outside training support.

## Smoke Test Results

All 15 uncertainty tests passed. Ensemble tests cover shapes/finite values, exact total-variance identity, zero epistemic variance for identical means, bootstrap diversity, gradients through all members, TS∞ identity persistence, TS1 switching, and mode validation.

| Metric | Result |
|---|---:|
| train member-mean NLL `epoch 1 -> 60` | `1.30675 -> -3.56282` |
| validation member-mean NLL | -3.68912 |
| moment-matched held-out NLL | -3.58230 |
| held-out RMSE | 0.06051 |
| 1σ / 2σ total coverage | 0.7324 / 0.9668 |
| aleatoric predicted/true std correlation | 0.9526 |
| ID epistemic std | 0.01066 |
| OOD epistemic std | 0.01640 |
| OOD / ID epistemic ratio | 1.5383 |

## Failure Cases

- OOD epistemic disagreement rises, but only modestly. The learned member aleatoric variances grow much more and dominate total uncertainty outside support.
- Total coverage is slightly conservative; adding epistemic variance improves safety margin but can reduce sharpness.
- Bootstrap neural ensembles are not guaranteed to cover every plausible model or detect every OOD input.
- Moment matching compresses a mixture of Gaussians into one Gaussian and can hide multimodality.
- TS1 changes model identity each step and can create trajectories inconsistent with any one learned dynamics hypothesis; TS∞ can preserve a bad member for the entire horizon.

## Findings

- Ensemble disagreement is spatially lowest inside most training support and increases toward/OOD beyond its boundary.
- Aleatoric variance remains strongly correlated with the true process noise after ensembling.
- The two uncertainty types can be computed separately, but learned models do not guarantee a perfectly clean semantic decomposition.
- Propagation choice is part of the world-model assumption, not an implementation detail.

## Limitations

- Five members, one ensemble seed, tiny synthetic dynamics.
- OOD split is deliberately geometric and easy to define.
- No elite selection, input/output normalization, PETS benchmark, CEM controller, reward model, or receding-horizon control.
- Sequential Python member execution; no vectorized ensemble layers.

## Compare Later

- Single Gaussian versus ensembles of 3/5/10 members.
- Bootstrap versus identical data, initialization-only diversity, or Bayesian approximations.
- Metrics: ID/OOD NLL, coverage, sharpness, error-detection AUROC, epistemic/data-size response, rollout coverage, compute/memory.
- Expected advantage: data-support sensitivity and multiple dynamics hypotheses.
- Expected weakness: linear parameter/training cost and unreliable disagreement under shared bias.
- Ablations: no bootstrap, shared initialization, deterministic members, fixed member variance, TS1 versus TS∞, moment matching versus mixture scoring.

## Final Model Candidate

```text
Candidate:
Yes for high-stakes rollout/planning experiments; size/propagation Undecided.

Reason:
It adds an empirically distinct OOD disagreement signal and supports PETS-style particles.

Advantages:
- separate within-member and between-member variance
- easy bootstrap implementation
- trajectory particles retain model hypotheses

Disadvantages:
- roughly E-times parameter/training cost
- disagreement can remain small under shared extrapolation bias
- aleatoric heads can absorb OOD error

Possible conflicts:
- RSSM stochasticity and ensemble particles may multiply rollout cost
- Transformer/video models make full ensembles expensive
- planning must choose a propagation and risk objective
```

## Next Questions

1. Does epistemic disagreement shrink with more in-region data while aleatoric variance stays fixed?
2. Which PETS propagation scheme gives calibrated long-horizon coverage?
3. Should planning penalize epistemic and aleatoric uncertainty differently?
4. Can latent ensembles share an encoder without collapsing diversity?

## References

### Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS)

Authors: Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine. Year: 2018. Paper: https://arxiv.org/abs/1805.12114.

Used for: probabilistic ensembles, bootstrap member training, decomposition-aware motivation, and trajectory sampling with TS1/TS∞ model assignment. Corresponding code: `probabilistic_ensemble.py`, `train.py`, `evaluate.py`. This experiment omits CEM planning, reward optimization, elite selection, and PETS control benchmarks.

### What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

Authors: Alex Kendall, Yarin Gal. Year: 2017. Paper: https://arxiv.org/abs/1703.04977.

Used for: aleatoric versus epistemic interpretation. Corresponding code: `ProbabilisticEnsemble.decompose` and ID/OOD evaluation.

### Provenance statement

The point environment, geometric OOD split, plots, and compact evaluation are **independent educational implementations**. The ensemble and trajectory-sampling mechanisms are **simplified PETS-inspired implementations**, not paper-result reproduction.
