# Decoder-Free Latent Planning

Status: completed on 2026-08-22. Simplified educational implementation inspired by MuZero and TD-MPC2; it is not a reproduction of either system.

## Purpose

Learn a compact state in which actions can be rolled forward and evaluated without reconstructing observations. This connects representation, dynamics, reward/value prediction, and CEM planning through one executable interface.

## Problem

The first three planning experiments use an exact hand-written transition and reward function. A practical World Model must plan with learned predictions. Pixel reconstruction can also spend capacity on visually detailed information that is irrelevant to control.

## Previous Model

`01_random_shooting`, `02_cem`, and `03_mpc` isolate search behavior using `PointWorldModel`. The Phase 06 heads predict control targets, but do not own a learned latent transition. This experiment replaces the planner's exact transition with a jointly learned encoder and latent dynamics.

## Hypothesis

Reward/value supervision plus latent consistency should produce a task-oriented latent space in which CEM finds actions that reduce true Goal distance, despite having no observation decoder.

## Architecture

```text
observation s_t [4] --encoder--> z_t [16]
                                  |
                     action a_t --+--> latent dynamics --> z_hat_(t+1)
                                                        |          |
                                                   reward head   value head

CEM action sequences --> repeated latent dynamics --> predicted return --> elites
selected actions ----------------------------------> exact environment (evaluation only)
```

## Data Flow

Training encodes every observed state. Starting at `z_0`, the model recursively predicts the next latent and matches the stop-gradient encoded next observation. Reward and value heads are trained on the recursively predicted latent, so model error encountered during planning is represented during training. Planning encodes only the initial observation and never decodes a latent.

## Tensor Shapes

| Quantity | Shape |
|---|---|
| observations | `[B, T+1, 4]` |
| actions | `[B, T, 2]` |
| rewards | `[B, T]` |
| distance-value targets | `[B, T+1]` |
| encoded/predicted latent | `[B, 16]` per step |
| CEM candidates | `[512, 10, 2]` |
| imagined latents | `[512, 10, 16]` |
| candidate scores | `[512]` |

Here `B=64` and `T=10` in the smoke run. Observation coordinates and actions are continuous and bounded; the exact environment applies `0.2*tanh(action)`.

## Mathematics

Representation and latent transition:

```text
z_t = e_theta(o_t)
z_hat_(t+1) = tanh(z_hat_t + f_theta(z_hat_t, a_t))
```

The residual transition retains a state-like path while allowing an action-conditioned change.

Decoder-free training objective:

```text
L_cons = mean_t ||z_hat_(t+1) - sg(e_theta(o_(t+1)))||^2
L_r    = mean_t (r_hat(z_hat_(t+1)) - r_(t+1))^2
L_v    = mean_t (v_hat(z_hat_t) - [-distance_t])^2
L      = L_cons + 2 L_r + L_v
```

`sg` means stop-gradient. Reward/value prediction prevents a consistency-only representation from being useful merely by collapsing to a constant. It does not mathematically guarantee a non-collapsed or sufficient representation.

The planner scores an action sequence by

```text
J = sum_(t=0)^(H-1) gamma^t r_hat(z_hat_(t+1)) + gamma^H v_hat(z_hat_H).
```

## Code Mapping

| Concept | File / symbol |
|---|---|
| mixed random/goal-directed sequences | `dataset.py::LatentPlanningSequenceDataset` |
| representation | `model.py::TaskOrientedLatentModel.encoder` |
| recurrent latent transition | `model.py::TaskOrientedLatentModel.transition` |
| reward/value prediction | `model.py::TaskOrientedLatentModel.reward`, `value` |
| recursive consistency and task losses | `losses.py::latent_model_loss` |
| decoder-free rollout/scoring | `model.py::rollout`, `evaluate_action_sequences` |
| elite action optimization | `planner.py::LatentCEMPlanner` |
| exact-world transfer check | `evaluate.py::evaluate` |

## Training

- seed: 101
- dataset: `point-world-latent-planning-v1`, 512 train / 128 validation sequences
- optimizer: Adam, learning rate `1e-3`
- batch size: 64, epochs: 60, optimizer steps: 480
- parameters: 5,922
- checkpoint: dictionary format version 1 with model state, config, optimizer name, and step count

Run:

```bash
MPLCONFIGDIR=/tmp/world-model-mpl .venv/bin/python 07_planning/04_latent_planning/train.py
MPLCONFIGDIR=/tmp/world-model-mpl .venv/bin/python 07_planning/04_latent_planning/evaluate.py
```

## Losses

- consistency loss teaches action-conditioned temporal prediction in latent space;
- reward loss makes planned latents decision-relevant and represents the sparse Goal bonus;
- value loss supplies a horizon-end estimate and anchors the latent to Goal distance;
- there is deliberately no pixel/state reconstruction loss.

## Evaluation Interface

`evaluate.py` loads `outputs/checkpoint.pt`, calls latent CEM, applies the selected actions to the exact Point World only for external validation, and writes `evaluation_metrics.json` and `latent_plan.png`.

## Smoke Test Results

Four focused tests passed. Validation losses were: total `0.2961`, consistency `0.02193`, reward `0.12991`, and value `0.01430`. Learned-latent CEM reduced exact-world distance from `2.2672` to `0.9858` over ten open-loop actions, a `56.5%` reduction. The learned score (`-11.470`) differs from the exact score (`-12.498`), exposing model bias rather than hiding it.

## Failure Cases

- The sparse Goal bonus is hard to regress and reward MSE remains the largest component.
- CEM can exploit errors in reward/value/dynamics; high latent score need not imply high real return.
- The exact-world distance bottoms near step 8 and then rises slightly; the open-loop latent plan overshoots its best real state.
- Open-loop execution cannot correct accumulated model bias.
- A constant latent can minimize consistency alone; task heads are essential but not a formal anti-collapse guarantee.
- The learned plan is weaker than exact-model CEM in this one-seed smoke run; this is not yet a controlled comparison.

## Findings

A decoder is not required for a planning interface: task losses can organize a latent representation sufficiently for a short-horizon control smoke test. The gap between learned and exact score is itself an important diagnostic for Phase 90.

## Limitations

This is a small deterministic coordinate task, not pixels. It has no policy prior, Q-function ensemble, distributional targets, target encoder/EMA, temporal weighting, multi-task conditioning, normalization suite, MCTS, or online data collection. These omissions distinguish it from TD-MPC2 and MuZero.

## Compare Later

- compare exact-model CEM, decoder-trained latent planning, and this decoder-free model under matched candidate budgets;
- measure true return, model-predicted return, exploitation gap, success, multi-step latent error, latency, and parameter count;
- vary consistency/reward/value weights and remove each head;
- test open-loop planning versus MPC replanning and in-/out-of-distribution starts;
- expected advantage: control-relevant compactness; expected weakness: representation collapse/model exploitation.

## Final Model Candidate

```text
Candidate: Undecided
Reason: The complete learned latent planning path works, but only on one small deterministic smoke task.
Advantages: no decoder during planning; differentiable compact dynamics; CEM-compatible; task-oriented.
Disadvantages: model exploitation, imperfect sparse-reward fit, no calibrated uncertainty or policy prior.
Possible conflicts: reconstruction objectives may compete for latent capacity; ensemble planning multiplies compute.
```

## Next Questions

Can an actor learn inside this latent model, replacing repeated black-box action search with imagined policy/value learning? Can continuation and uncertainty reduce exploitation?

## References

### Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)

- Authors: Julian Schrittwieser et al.
- Year: 2020
- Paper: https://arxiv.org/abs/1911.08265 ; https://doi.org/10.1038/s41586-020-03051-4
- Used for: learning representation, dynamics, reward, and value for planning without requiring observation reconstruction.
- Implementation: conceptual lineage for `model.py`; this experiment uses CEM, not MuZero's MCTS, policy head, or training algorithm.

### TD-MPC2: Scalable, Robust World Models for Continuous Control

- Authors: Nicklas Hansen, Hao Su, Xiaolong Wang
- Year: 2023 preprint / ICLR 2024
- Paper: https://arxiv.org/abs/2310.16828
- Used for: decoder-free task-oriented latent consistency and local continuous-action trajectory optimization.
- Implementation: `model.py`, `losses.py`, `planner.py`.

Classification: **Simplified educational implementation** and **independent Point World adaptation**. No benchmark result or complete algorithm from either paper is claimed.
