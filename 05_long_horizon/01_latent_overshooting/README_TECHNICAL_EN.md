# Latent Overshooting: Training Beyond One-Step Prediction

Status: completed on 2026-08-22. This is a deterministic educational analogue of PlaNet latent overshooting, not PlaNet's variational multi-step objective.

## Purpose

Expose compounding error and train a latent dynamics model on its own recursive predictions for distances 1–5. Evaluate whether excellent teacher-forced one-step accuracy is sufficient for a 30-step rollout.

## Problem

One-step training always starts from a true state. Open-loop imagination starts correctly only once and then feeds predictions back into the model. Small errors move the input distribution, create new errors, and compound over time.

## Previous Model

Earlier experiments checked short rollouts, generally six steps. Their low average pixel error was heavily influenced by static backgrounds. This experiment uses an exact two-dimensional nonlinear state so long-horizon dynamics error is directly visible.

## Hypothesis

Multi-step overshooting should make the model locally robust to its own prediction errors and reduce short/mid-horizon drift relative to relying on the one-step term alone. A matched baseline comparison is deferred; this phase verifies the mechanism and measures its remaining horizon limit.

## Architecture

```text
state s_t=(position,velocity) [2] + action a_t [4]
                         |
                     MLP delta
                         |
                  predicted s_{t+1}
                         |
              feed prediction back for k steps
                         |
          compare to s_{t+1},...,s_{t+k}, k<=5
```

## Data Flow

```text
deterministic controlled oscillator -> sequences [s_0,a_0,...,s_30]
   -> one-step teacher-forced prediction at every t
   -> for every start t, recursively rollout up to five available steps
   -> one-step + overshooting loss
   -> evaluation: one true s_0, 30 predicted transitions
```

## Tensor Shapes

| Tensor | Shape | Meaning |
|---|---|---|
| states | `[B,T+1,2] = [B,31,2]` | position and velocity |
| actions | `[B,T,4] = [B,30,4]` | one-hot acceleration |
| teacher-forced predictions | `[B,T,2]` | each starts from true `s_t` |
| autoregressive rollout | `[B,T,2]` | each starts from previous prediction |
| loss by overshoot distance | `[5]` | mean MSE at distances 1–5 |

## Mathematics

The known nonlinear transition is

```text
v_{t+1}=0.97v_t+accel(a_t)-0.02 sin(3p_t)
p_{t+1}=p_t+v_{t+1}.
```

The learned residual transition is

```text
s_hat_{t+1}=s_t+f_theta(s_t,a_t).
```

For every start `t`, recursive prediction is

```text
s_hat_{t+d}=f_theta^d(s_t,a_t,...,a_{t+d-1}), d=1...K.
```

The educational objective is

```text
L = L_one_step + 0.5 * mean_{t,d<=5} ||s_hat_{t+d}-s_{t+d}||^2.
```

PlaNet overshooting instead applies a multi-step variational objective to stochastic latent distributions. There is no prior/posterior/KL here.

## Code Mapping

| Concept | File / symbol |
|---|---|
| nonlinear true transition | `sequence_dataset.py::oscillator_transition` |
| reproducible sequences | `ControlledOscillatorSequenceDataset` |
| residual latent transition | `latent_dynamics.py::LatentDynamics.forward` |
| open-loop rollout | `LatentDynamics.rollout` |
| all-start multi-distance unroll | `overshooting_losses.py::latent_overshooting_loss` |
| combined objective | `long_horizon_loss` |
| horizon metrics/plots | `evaluate.py::evaluate` |

## Training

```bash
.venv/bin/python 05_long_horizon/01_latent_overshooting/train.py
.venv/bin/python 05_long_horizon/01_latent_overshooting/evaluate.py
.venv/bin/python -m pytest -q 05_long_horizon/01_latent_overshooting/tests
```

| Reproducibility item | Value |
|---|---|
| seed / dataset | 47 / `controlled-oscillator-v1` |
| sequences | 256 train / 64 validation, 30 transitions |
| model | residual MLP, 2x64 Tanh |
| overshooting | max distance 5, weight 0.5 |
| optimizer | Adam `1e-3`, batch 64 |
| epochs / steps | 80 / 320 |
| parameters | 4,738 |
| checkpoint/evaluation | format 1 gitignored / `python 05_long_horizon/01_latent_overshooting/evaluate.py` |

## Losses

- One-step MSE learns local transition accuracy from true states.
- Overshooting MSE exposes the model to predicted states for distances 2–5 and backpropagates through the recursive computation.
- No uncertainty or image loss is used; this isolates deterministic compounding error.

## Evaluation Interface

Evaluation records teacher-forced one-step MSE, overshooting error by distance 1–5, open-loop error for every horizon 1–30, named 5/10/30-step values, parameter count, and a true-versus-rollout state plot.

## Smoke Test Results

Five tests passed: exact environment transition/dataset chronology, finite forward/rollout shapes, distance-1 identity with one-step MSE, gradients through recursive unroll, and independence from unused future true states.

| Metric | Result |
|---|---:|
| train total `epoch 1 -> 80` | `0.052593 -> 0.002689` |
| final validation total | 0.002114 |
| held-out one-step MSE | 0.0001605 |
| overshoot MSE distance 1/2/3/4/5 | 0.000161 / 0.000897 / 0.002774 / 0.006445 / 0.012545 |
| rollout MSE horizon 5 | 0.006229 |
| rollout MSE horizon 10 | 0.066463 |
| rollout MSE horizon 30 | 1.408694 |

## Failure Cases

- Very small one-step MSE coexists with severe 30-step divergence.
- Five-step overshooting does not constrain inputs reached after twenty or thirty recursive errors.
- Equal averaging over starts/distances may underweight the most difficult long offsets.
- A deterministic model cannot represent uncertainty growth or multiple futures.
- This run alone does not show improvement over a one-step-only baseline.

## Findings

- Compounding error is quantitatively large and monotonic on the held-out oscillator despite strong local prediction.
- Overshooting creates genuine recursive gradient paths but only across its configured distance.
- “Trained with multi-step loss” must always be accompanied by the exact training distance and longer evaluation horizons.
- This failure motivates both longer/multi-scale objectives and temporal abstraction.

## Limitations

- Fully observed two-dimensional deterministic state, not visual latent inference.
- No stochastic prior/posterior or PlaNet variational bound.
- Fixed overshooting distance and uniform weighting.
- One seed, no baseline/ablation until Phase 90.

## Compare Later

- One-step-only versus overshooting distances 3/5/10 and weights.
- Metrics: 1/5/10/20/30-step error, Jacobian stability, train cost, gradient norm, parameters.
- Expected advantage: robustness to predicted-state inputs within trained distance.
- Expected weakness: quadratic-like unroll cost over starts/distances and no guarantee beyond K.
- Ablations: detach intermediate predictions, start only at t=0, distance weighting, curriculum, scheduled sampling.

## Final Model Candidate

```text
Candidate:
Undecided

Reason:
The mechanism is correct and targets exposure mismatch, but this smoke result still diverges strongly beyond its five-step training window.

Advantages:
- directly trains recursive dynamics
- no new inference-time parameters
- distance-specific diagnostics

Disadvantages:
- higher training computation/memory
- sensitive to horizon and weighting
- does not solve temporal scale or uncertainty alone

Possible conflicts:
- RSSM variational overshooting needs KL/distribution handling
- Transformer long contexts and ensemble particles multiply unroll cost
```

## Next Questions

1. Can a macro-transition skip several primitive steps and reduce the number of recursive applications?
2. How should overshooting distance be scheduled or weighted?
3. Does uncertainty predict when rollout leaves the trained state region?
4. How does PlaNet's distribution-level overshooting differ empirically from deterministic state MSE?

## References

### Learning Latent Dynamics for Planning from Pixels (PlaNet)

Authors: Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson. Year: 2018. Paper: https://arxiv.org/abs/1811.04551.

Used for: latent overshooting motivation—training multi-step latent predictions rather than only adjacent transitions. Corresponding code: `overshooting_losses.py`. This experiment replaces PlaNet's stochastic variational objective with deterministic state MSE and omits observation inference, KL, reward, and planning.

### Provenance statement

The controlled oscillator and deterministic overshooting loss are an **independent simplified educational implementation**. Results are not PlaNet reproduction results.
