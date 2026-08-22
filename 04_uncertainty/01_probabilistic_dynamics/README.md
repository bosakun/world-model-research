# Probabilistic Dynamics: Learning Aleatoric Uncertainty

Status: completed on 2026-08-22. This is an independent heteroscedastic point-dynamics experiment informed by PETS and uncertainty literature, not a PETS reproduction.

## Purpose

Change a deterministic next-state prediction into a Gaussian distribution and learn input-dependent process noise. This isolates **aleatoric uncertainty**: randomness inherent in a transition that remains even with abundant data.

## Problem

MSE produces a point estimate. It can learn the conditional mean but cannot distinguish a reliably predictable transition from one whose outcomes vary. Planning with only a mean trajectory can therefore be overconfident.

## Previous Model

The Memory experiments output deterministic images/latents, except that RSSM contains a stochastic latent variable. They did not evaluate whether a predicted standard deviation matches known transition noise. This experiment uses a simple continuous state so uncertainty has an exact ground truth.

## Hypothesis

A dynamics network trained by Gaussian negative log-likelihood will learn both mean motion and the known heteroscedastic noise scale. Sampled rollouts should spread over horizon while deterministic mean rollout stays repeatable.

## Architecture

```text
state s_t [2] + one-hot action a_t [4]
                  |
                MLP
             /         \
    mean delta [2]   raw log variance [2]
          |               |
  mu_{t+1}=s_t+delta   bounded logvar
             \           /
         N(mu_{t+1}, diag(sigma^2_{t+1}))
                    |
           mean or sampled next state
```

The data generator applies known action deltas and input-dependent Gaussian noise. True noise standard deviation is stored for evaluation only.

## Data Flow

```text
(s_t,a_t) -> stochastic environment -> sampled s_{t+1}
     |                                  |
     +-> probabilistic model -----------+
                  |
           Gaussian NLL training

rollout: sampled s_{t+1} becomes the next model input
```

## Tensor Shapes

| Tensor | Shape | Meaning |
|---|---|---|
| states | `[B,2]` | continuous `(x,y)` |
| one-hot actions | `[B,4]` | left/right/down/up |
| next states | `[B,2]` | one noisy transition sample |
| predicted mean/log variance/std | `[B,2]` | diagonal Gaussian parameters |
| sequence states | `[B,T+1,2]` | stochastic ground-truth rollout |
| sequence actions | `[B,T,4]` | action controls |
| model rollout states/means/stds | `[B,T,2]` | predicted trajectory distribution |

## Mathematics

The synthetic transition is

```text
s_{t+1} = s_t + delta(a_t) + epsilon_t,
epsilon_t ~ N(0, diag(sigma_true(s_t,a_t)^2)).
```

Horizontal noise grows sigmoidally with `x`; vertical actions have larger vertical noise. The model predicts

```text
p_theta(s_{t+1}|s_t,a_t) = N(mu_theta, diag(sigma_theta^2)).
```

For state dimension `j`, Gaussian NLL is

```text
L_j = 1/2 [log(sigma_j^2) + (target_j-mu_j)^2/sigma_j^2 + log(2pi)].
```

The squared-error term rewards correct means, while `log(sigma^2)` prevents the model from making variance arbitrarily large. Learned soft upper/lower log-variance bounds prevent numerical extremes.

Sampling uses

```text
s_hat_{t+1} = mu + sigma * epsilon, epsilon~N(0,I).
```

## Code Mapping

| Concept | File / symbol |
|---|---|
| known stochastic environment | `stochastic_dataset.py::stochastic_transition` |
| heteroscedastic `sigma_true` | `transition_noise_std` |
| transition and sequence datasets | `HeteroscedasticTransitionDataset`, `StochasticPointSequenceDataset` |
| Gaussian network | `probabilistic_dynamics.py::ProbabilisticDynamics` |
| bounded log variance | `ProbabilisticDynamics.forward` |
| reparameterized state sample | `GaussianPrediction.sample` |
| Gaussian NLL | `probabilistic_losses.py::diagonal_gaussian_nll` |
| sampled rollout | `ProbabilisticDynamics.rollout` |
| coverage/calibration plots | `evaluate.py::evaluate` |

## Training

```bash
.venv/bin/python 04_uncertainty/01_probabilistic_dynamics/train.py
.venv/bin/python 04_uncertainty/01_probabilistic_dynamics/evaluate.py
.venv/bin/python -m pytest -q 04_uncertainty/01_probabilistic_dynamics/tests
```

| Reproducibility item | Value |
|---|---|
| seed / dataset | 37 / `heteroscedastic-point-v1` |
| train / validation | 1024 / 256 transitions |
| model | two 64-unit SiLU hidden layers, diagonal Gaussian |
| optimizer / rate | Adam / `1e-3` |
| batch / epochs / steps | 64 / 80 / 1280 |
| parameters | 4,872 |
| checkpoint | format 1, `outputs/checkpoint.pt`, gitignored |
| evaluation | `python 04_uncertainty/01_probabilistic_dynamics/evaluate.py` |

## Losses

- Gaussian NLL jointly trains transition mean and aleatoric variance.
- A `1e-4` bound regularizer discourages excessively wide trainable log-variance bounds.
- No ensemble disagreement appears here; this model does not estimate epistemic uncertainty.

## Evaluation Interface

Evaluation reports next-state RMSE, Gaussian NLL, empirical 1σ/2σ coverage, correlation between predicted and known noise std, mean predicted/true std, sample count, horizon, and parameter count. It also visualizes an uncertainty curve and 64 sampled rollouts.

## Smoke Test Results

All eight experiment tests passed: dataset/sequence alignment, known heteroscedasticity, positive finite variance, analytic standard-normal NLL, reparameterization gradients, mean/variance gradient flow, rollout shapes, and stochastic versus deterministic behavior.

| Metric | Result |
|---|---:|
| train NLL `epoch 1 -> 80` | `1.29975 -> -3.65436` |
| validation NLL | -3.73237 |
| held-out next-state RMSE | 0.05686 |
| held-out Gaussian NLL | -3.59307 |
| within 1σ coverage | 0.6953 |
| within 2σ coverage | 0.9531 |
| predicted/true std correlation | 0.9376 |
| mean predicted std `(x,y)` | `(0.0532,0.0453)` |
| mean true std `(x,y)` | `(0.0539,0.0438)` |

Negative NLL is valid for continuous densities: density values can exceed one when a distribution is narrow. It is not a negative probability.

## Failure Cases

- `aleatoric_std.png` shows predicted std diverging above the true curve outside/near the edge of the training range. A single network has no separate “I lack data here” channel, so model error can contaminate its variance estimate.
- Diagonal Gaussian cannot express correlated noise or multiple separated outcome modes.
- Calibration on held-out in-distribution samples does not imply OOD calibration.
- Repeated sampling spreads trajectories rapidly; mean rollout hides this risk.

## Findings

- NLL recovered the known input-dependent noise closely in the data-supported region.
- Coverage is close to Gaussian reference rates (~0.68 and ~0.95).
- Mean accuracy and uncertainty calibration are separate evaluation axes.
- OOD divergence motivates an ensemble that can measure parameter/model disagreement separately.

## Limitations

- Two-dimensional synthetic state, not image/latent dynamics.
- Gaussian noise is exactly the model family assumed by the learner.
- One seed and no formal baseline comparison.
- No epistemic uncertainty, ensemble, trajectory-particle assignment, reward, or planning.

## Compare Later

- Compare deterministic MSE versus probabilistic NLL on mean RMSE and calibration.
- Compare single probabilistic model versus ensemble in- and out-of-distribution.
- Metrics: NLL, RMSE, coverage, calibration curve, sharpness, epistemic/aleatoric decomposition, rollout coverage, parameters, latency.
- Expected advantage: state-dependent risk and sampleable futures.
- Expected weakness: distributional misspecification and variance absorbing model error.
- Ablations: fixed variance, homoscedastic variance, unbounded variance, mean-only rollout, sampled rollout.

## Final Model Candidate

```text
Candidate:
Yes for a probabilistic output head; exact Gaussian form Undecided.

Reason:
Planning should know outcome spread, and the smoke task verifies calibrated input-dependent variance.

Advantages:
- separates mean prediction from transition noise
- supports NLL, coverage, and trajectory sampling
- small additional output cost

Disadvantages:
- diagonal unimodal assumption
- single model confounds OOD/model error with predicted noise

Possible conflicts:
- RSSM stochastic state also represents uncertainty but at a different latent level
- ensemble variance must be combined without double-counting aleatoric variance
```

## Next Questions

1. Can an ensemble make epistemic uncertainty rise outside the training state region?
2. Does more data reduce ensemble disagreement while leaving aleatoric variance intact?
3. How should mixture/multimodal output distributions replace a diagonal Gaussian?
4. How should particles propagate both uncertainty types through long rollout?

## References

### Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS)

Authors: Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine. Year: 2018. Paper: https://arxiv.org/abs/1805.12114.

Used for: probabilistic neural dynamics, bounded log variance, and trajectory sampling motivation. Corresponding code: `probabilistic_dynamics.py`, `probabilistic_losses.py`, `evaluate.py`. Ensembles, bootstrap training, TS1/TS∞ propagation, CEM, and control benchmarks are not implemented in this subexperiment.

### What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

Authors: Alex Kendall, Yarin Gal. Year: 2017. Paper: https://arxiv.org/abs/1703.04977.

Used for: epistemic versus aleatoric distinction and input-dependent/heteroscedastic uncertainty framing. Corresponding implementation: `transition_noise_std` and the learned variance head.

### Provenance statement

The synthetic continuous environment and its noise law are an **independent educational implementation**. The probabilistic-head and sampling concepts are paper-informed but do not reproduce PETS experiments.
