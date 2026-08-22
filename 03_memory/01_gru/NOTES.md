# Research Notes: GRU Latent Dynamics

Date started: 2026-08-22

## Initial questions

- Is the remembered prior implementation actually present in this checkout?
- If the environment is fully observable, what information could a GRU retain that is not already in `z_t`?
- Should hidden state predict the next latent directly, or should prediction condition on both hidden and current latent?
- How do teacher-forced hidden updates differ from autoregressive rollout?

## Pre-implementation expectations

- GRU should train and roll out, but full observability should make memory redundant in principle.
- Multi-step decoded predictions should degrade with horizon because predicted latents feed themselves back.
- A reconstruction term is needed so the latent cannot solve dynamics by collapsing to a constant.

## Repository audit surprise

The directory contained only `.git` and had no commits. None of the recalled Grid World, autoencoder, or latent-dynamics code was available. This changed the implementation plan from “reuse existing Encoder/Decoder and latent_dim” to “build a minimal self-contained experimental substrate and state every new choice.” See the root audit report.

## Implementation decisions

- `latent_dim=16`: new experimental choice; there was no existing value to match.
- `hidden_dim=64`: requested initial scale.
- `GRUCell`: chosen to expose state timing explicitly.
- deterministic 5x5 full-observation RGB Grid World: enough to validate visual/temporal paths while preserving the central caveat that memory is not necessary.
- action one-hot order: up, down, left, right.
- teacher-forced dynamics training plus separate autoregressive rollout evaluation.
- stopped encoded target in dynamics loss plus image reconstruction to reduce representation-collapse pressure.
- Simple Dynamics retained but not given a fake random-weight “comparison result.” A controlled trained comparison belongs with partial observation / comparison phase.

## Paper-to-code boundary

- GRU gating: derived from the GRU family introduced by Cho et al.; exact computation is PyTorch `GRUCell`.
- recurrent world-model context: motivated by World Models.
- deterministic/stochastic state distinction: informed by PlaNet/Dreamer.
- independent educational changes: RGB Grid World, CNN architecture, latent MSE, target detach, loss weights, dataset, and evaluation protocol.
- explicitly absent: MDN-RNN, VAE, RSSM prior/posterior, KL, reward/value, actor, planning.

## Problems and errors encountered

- The host's default Python 3.14 environment had no PyTorch, NumPy, matplotlib, or pytest. The project declares Python `>=3.11,<3.14` so `uv` can create a compatible managed environment.
- First training run: total loss spiked during joint Encoder/dynamics training and eventually produced low pixel MSE while omitting the agent in decoded rollouts. Cause: the agent occupies few pixels, so plain image MSE rewarded a background/goal shortcut; unconstrained latent scale also made the detached-target joint objective less stable.
- Correction: bound Encoder output with `tanh`, weight high-intensity agent/goal pixels 11x in reconstruction, and add agent-cell accuracy. Pixel MSE alone is retained but is no longer treated as proof of state prediction.
- Second run: training became stable and dynamics MSE fell to about `6.7e-6`, yet agent-cell accuracy was `3.9%`, essentially the 1/25 chance level. This revealed a second shortcut: equal “active” weighting still let the fixed goal dominate while latents carried little moving-state information. Correction: detect simulator-known red agent and green goal pixels separately, weighting them 101x and 11x respectively. This is deliberately recorded as an educational supervised weighting choice, not a paper-faithful unsupervised representation objective.
- Third run: target-only 101x red weighting made the decoder paint nearly every cell red; incorrect red pixels were cheap relative to missing the true red pixels. Agent accuracy stayed at chance. Correction: return to ordinary full-frame MSE and add an exclusive 25-way agent-cell cross-entropy computed from decoded colors. This penalizes false positions and explicitly tests whether the latent carries agent location.
- Fourth run: even with the position objective, cross-entropy remained near `ln(25)` and accuracy at chance. The small transposed-convolution decoder did not learn a clean global latent-to-cell mapping. Correction: for this 20x20 educational environment, replace it with an explicit MLP decoder (`16 -> 256 -> 1200`) rather than expanding convolutional complexity unrelated to the GRU question.

## Before / After

```text
Before (conceptual baseline):
(z_t, a_t) -> MLP -> predicted z_{t+1}

After:
(z_t, a_t, h_t) -> GRUCell -> h_{t+1} -> head -> predicted z_{t+1}
```

## Results

- Final test run: 6/6 passed.
- Final 40-epoch run: train total `0.803508 -> 0.252070`; validation total `0.278304`.
- Validation reconstruction MSE `0.003758`; position CE `0.005982`.
- Held-out one-step: latent MSE `0.141365`, pixel MSE `0.005980`, agent-cell accuracy `83.59%`.
- Held-out 8-step autoregressive rollout: mean pixel MSE `0.008376`, mean agent-cell accuracy `55.86%`.
- Position accuracy by horizon: `75.0, 76.6, 64.1, 54.7, 51.6, 50.0, 37.5, 37.5%`.
- GRU hidden sequence was `[64,8,64]` for evaluation batch; Simple Dynamics was retained but not trained for an unfair comparison.
- Conclusion: implementation and rollout work, but error compounds and complete observation does not establish a causal benefit from memory.

## Failure cases

- Plain-MSE “success” omitted the agent.
- Target-only weighting painted many/all cells red.
- Final rollout begins near the correct trajectory but develops false colors and position errors at longer horizons.
- The final plotted sample is deterministic (seeded), not hand-selected from multiple outputs.

## Figures useful for a future article

- `outputs/loss_curve.png`: joint training behavior.
- `outputs/rollout_comparison.png`: truth vs autoregressive predicted observations.
- `outputs/rollout_error.png`: compounding error by horizon.
- A future diagram contrasting identical partial observations with different hidden histories.

## Potential article angles

- “A GRU running successfully does not prove memory is useful.”
- `z_t` is present perception; `h_t` is history-dependent belief/context.
- Teacher forcing can hide rollout failure.
- Why a fully observable Markov environment is a necessary implementation check but an insufficient memory experiment.
- Why GRU dynamics are not automatically an RSSM.

## Interpretation guardrails

- Do not call the implementation PlaNet or Dreamer.
- Do not infer memory benefit from GRU loss alone.
- Do not compare an optimized GRU to an untrained baseline.
- Distinguish smoke-run reproducibility from robust multi-seed evidence.
