# Recurrent State-Space Model: Deterministic Memory and Stochastic State

Status: completed on 2026-08-22. This is a simplified educational RSSM inspired primarily by PlaNet. It is not a full PlaNet or Dreamer reproduction.

## Purpose

Build the first probabilistic recurrent world model in this repository. The experiment separates persistent deterministic memory `h_t` from a stochastic latent state `z_t`, learns an observation-conditioned posterior for training, learns a predictive prior for imagination, and aligns them with KL divergence.

## Problem

The earlier GRU model emits one deterministic next latent. It does not explicitly represent multiple plausible hidden states, and it has no distinction between state inference with an observation and state prediction without one. Partial observation makes that distinction important: training can use `o_t` to infer hidden state, while future rollout must proceed from actions and model memory alone.

## Previous Model

`03_memory/01_gru` uses one GRU hidden vector and an MSE target in the encoder's latent space. `03_memory/02_partial_observation` supplies an image sequence in which the Goal can leave view. Neither experiment has a latent distribution, prior, posterior, variational objective, or observation reconstruction trained jointly with dynamics.

## Hypothesis

An RSSM can learn a compact recurrent belief-like state whose posterior explains observed frames and whose prior supports observation-free future rollout. KL alignment should make prior samples decodable, rather than leaving training and imagination in unrelated latent spaces.

This phase tests implementation validity and a small smoke task only. It does not claim superiority over GRU.

## Architecture

```text
observed filtering (training/evaluation)

o_t [3,20,20] -> CNN encoder -> e_t [64] ---------------------+
                                                                |
z_{t-1} [16] + a_{t-1} [4] -> GRUCell(h_{t-1}) -> h_t [64]     |
                                                     |          |
                                                     +-> prior p(z_t|h_t)
                                                     |          |
                                                     +----------+-> posterior q(z_t|h_t,e_t)
                                                                      |
                                                       reparameterized z_t [16]
                                                                      |
                                                    [h_t,z_t] [80]
                                                       /          \
                                         image decoder             Goal-state head
                                         o_hat_t [3,20,20]         logits [10]

prior imagination (no future image)

(h_t,z_t) + a_t -> GRUCell -> h_{t+1} -> p(z_{t+1}|h_{t+1})
                                      -> z_{t+1} -> decoder -> o_hat_{t+1}
```

The Goal-state head is a local experimental aid for this synthetic task, not a PlaNet component. It classifies nine visible local cells plus `not visible` from `[h_t,z_t]`.

## Data Flow

```text
true world sequence
  -> partial-observation function
  -> observations o_0...o_T and one-hot actions a_0...a_{T-1}
  -> encoder embeddings e_0...e_T
  -> posterior filtering q(z_t | h_t,e_t)
  -> reconstruction + semantic auxiliary loss
  -> KL(q || p) teaches predictive prior p(z_t | h_t)
  -> seed with posterior at t=0
  -> prior-only rollout using actions
  -> predicted future observations
```

`true_states` and `full_worlds` remain dataset evaluation metadata and are never inputs to the RSSM.

## Tensor Shapes

For batch `B=32`, action horizon `T=6`, image `C=3,H=W=20`, embedding `E=64`, deterministic dimension `D_h=64`, stochastic dimension `D_z=16`, and actions `D_a=4`:

| Tensor | Shape | Meaning |
|---|---|---|
| observations | `[B,T+1,C,H,W] = [32,7,3,20,20]` | partial image sequence |
| actions | `[B,T,D_a] = [32,6,4]` | one-hot transitions between frames |
| true states | `[B,T+1,4]` | evaluation-only world coordinates |
| embeddings | `[B,T+1,E] = [32,7,64]` | CNN observation features |
| deterministic states `h` | `[B,T+1,D_h] = [32,7,64]` | recurrent history |
| stochastic states `z` | `[B,T+1,D_z] = [32,7,16]` | sampled posterior state |
| prior mean/std | `[B,T+1,D_z]` | predictive distribution parameters |
| posterior mean/std | `[B,T+1,D_z]` | observation-conditioned parameters |
| reconstructions | `[B,T+1,C,H,W]` | decoded posterior states |
| imagined observations | `[B,T,C,H,W]` | future prior rollout |
| Goal logits | `[B,T+1,10]` or `[B,T,10]` | semantic auxiliary prediction |

## Mathematics

The deterministic transition is

```text
h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}]).
```

The predictive prior and observation-conditioned posterior are diagonal Gaussians:

```text
p(z_t | h_t)       = N(mu^p_t, (sigma^p_t)^2 I)
q(z_t | h_t, o_t)  = N(mu^q_t, (sigma^q_t)^2 I).
```

Sampling uses the reparameterization trick:

```text
epsilon ~ N(0,I),    z_t = mu^q_t + sigma^q_t * epsilon.
```

The decoder models the observation from the combined state:

```text
o_hat_t = Decoder([h_t,z_t]).
```

For each stochastic dimension, the analytic Gaussian divergence is

```text
KL(q||p) = log(sigma_p/sigma_q)
           + (sigma_q^2 + (mu_q-mu_p)^2)/(2 sigma_p^2) - 1/2.
```

The smoke objective is

```text
L = L_weighted_image + 0.1 L_goal + 0.001 max(KL(q||p), 1 nat).
```

The green image channel has weight 20 because the tiny Goal occupies few pixels. The Goal cross-entropy and channel weighting are independent experimental modifications; PlaNet does not define this synthetic classification task.

## Code Mapping

| Equation or concept | File / class / function |
|---|---|
| CNN `o_t -> e_t` | `rssm.py::ObservationEncoder.forward` |
| `h_t` recurrent transition | `rssm.py::RecurrentStateSpaceModel.transition`, `recurrent_transition` |
| diagonal Gaussian prior/posterior | `rssm.py::GaussianHead`, `prior`, `posterior` |
| reparameterization | `rssm.py::DiagonalGaussian.sample` |
| posterior filtering over a sequence | `rssm.py::RecurrentStateSpaceModel.observe` |
| prior-only future rollout | `rssm.py::RecurrentStateSpaceModel.imagine` |
| reconstruction model | `rssm.py::ObservationDecoder`, `decode` |
| semantic state auxiliary head | `rssm.py::goal_head`, `predict_goal` |
| analytic `KL(q||p)` | `rssm_losses.py::diagonal_gaussian_kl` |
| complete objective/free nats | `rssm_losses.py::rssm_loss` |
| partial sequence adapter | `rssm_dataset.py::build_rssm_dataset` |
| training/checkpoint metadata | `train.py::train` |
| reconstruction/prior rollout metrics | `evaluate.py::evaluate` |

`model.py`, `dataset.py`, and `losses.py` are conventional import wrappers; the named `rssm_*` files contain the implementation.

## Training

From repository root:

```bash
uv run python 03_memory/03_rssm/train.py
uv run python 03_memory/03_rssm/evaluate.py
uv run pytest -q 03_memory/01_gru/tests 03_memory/02_partial_observation/tests 03_memory/03_rssm/tests
```

Reproducibility record:

| Item | Value |
|---|---|
| seed | 23 |
| dataset | `partial-observation-v1`, 128 train / 32 validation sequences |
| sequence length | 6 actions / 7 observations |
| batch / epochs | 32 / 40 |
| optimizer / learning rate | Adam / `3e-3` |
| optimizer steps | 160 |
| model parameters | 428,330 |
| checkpoint | `outputs/checkpoint.pt`, format version 1; intentionally gitignored |
| input/output | observations `[B,7,3,20,20]`, actions `[B,6,4]`; posterior states/reconstructions and prior rollout |
| evaluation entry point | `python 03_memory/03_rssm/evaluate.py` |

## Losses

- Weighted image MSE teaches `[h_t,z_t]` to preserve decodable visual content. Green weighting counteracts the Goal's pixel imbalance.
- Goal-state cross-entropy checks that the learned state distinguishes nine visible Goal locations from `not visible`. It prevents a low pixel loss from hiding semantic failure.
- KL divergence makes the observation-free prior approximate the posterior used when the true observation exists. Without it, rollout begins from an untrained latent distribution.
- Free nats avoids spending optimization pressure on making an already-small KL even smaller. It is a practical Dreamer-lineage stabilization choice here.

## Evaluation Interface

`evaluate.py` loads checkpoint/config metadata, deterministically uses distribution means, and emits JSON plus plots. It reports plain pixel MSE separately from the weighted training reconstruction, semantic accuracy from the **state head**, per-horizon prior rollout error, tensor shapes, and parameter count. This interface is intentionally small so Phase 90 can wrap it without forcing every model into a premature framework.

## Smoke Test Results

All 21 tests across GRU, partial observation, and RSSM passed. RSSM-specific tests cover shape/finite checks, positive standard deviations, reparameterization gradients, analytic KL, gradients through encoder/GRU/prior/posterior/decoder/state head, posterior dependence on observation, prior independence from future observation, and prior-only imagination.

Final validation/training evidence:

| Metric | Result |
|---|---:|
| initial train total loss | 1.036060 |
| final train total loss | 0.016743 |
| final validation total loss | 0.010783 |
| final validation weighted reconstruction | 0.008861 |
| final validation raw KL | 1.756577 nats |
| posterior plain pixel MSE | 0.000746 |
| one-step prior plain pixel MSE | 0.000618 |
| mean 6-step prior rollout pixel MSE | 0.000628 |
| posterior/prior rollout state-head Goal accuracy | 1.000 / 1.000 |

See `outputs/loss_curve.png`, `reconstruction.png`, `latent_rollout.png`, and `rollout_error.png`.

## Failure Cases

- At the ambiguous first frame, the image decoder renders the correct right Goal strongly but also a faint second green candidate below the Agent. Pixel MSE permits this averaging; 100% state-head accuracy must not be misreported as perfect decoded imagery.
- Plain RGB MSE initially ignored the tiny Goal. Deriving the semantic class directly from decoded RGB then produced a shortcut: the decoder painted red artifacts so a `not visible` threshold would win. A separate state head removed that shortcut.
- Deterministic mean rollout hides sample diversity. Calibration and multi-sample evaluation are deferred to Phase 04.
- Low pixel error is easy on a mostly static, mostly dark 20x20 image and is not evidence of general world understanding.

## Findings

- The implementation cleanly separates observed inference (`posterior`) from unobserved prediction (`prior`).
- The prior can roll six actions without consuming future images and preserve the tested hidden Goal class in its learned state.
- A semantic probe exposed behavior that aggregate reconstruction MSE concealed.
- Stochastic state does not automatically prevent blurry conditional means; the observation model and objective still matter.

## Limitations

- Tiny deterministic educational Grid World; the stochastic latent is architectural, not proof that calibrated multimodality was learned.
- Continuous diagonal Gaussian only; no discrete DreamerV2 state.
- Weighted MSE stands in for an explicit learned observation likelihood.
- No reward/continuation model, actor/critic, planning, CEM, latent overshooting, KL balancing, or multi-step variational objective.
- No matched GRU/RSSM comparison or multiple seeds until Phase 90.

## Compare Later

- Compare with No Memory, GRU, and Transformer Memory on identical partial sequences.
- Metrics: posterior reconstruction, 1/5/10-step rollout, hidden-Goal probe, sample diversity/calibration, parameters, latency, memory, and stability across seeds.
- Expected advantage: explicit prior/posterior semantics and uncertainty-capable state for imagination.
- Expected weakness: harder optimization, KL sensitivity, sampling variance, and larger model.
- Ablations: deterministic-only `z=mu`, remove `h`, remove KL, remove free nats, remove Goal auxiliary loss, reset history, shuffle history, posterior rollout versus prior rollout.

## Final Model Candidate

```text
Candidate:
Undecided

Reason:
The core RSSM mechanisms work and enable prior imagination, but this smoke task does not establish calibrated uncertainty or an advantage over simpler memory.

Advantages:
- explicit observed posterior and predictive prior
- recurrent deterministic context plus stochastic state
- differentiable sampling and decodable imagination

Disadvantages:
- more objectives and tuning points than deterministic GRU
- current decoder shows a mild averaging artifact
- semantic auxiliary loss is task-specific

Possible conflicts:
- Transformer memory may replace the recurrent deterministic transition
- later uncertainty ensembles may duplicate or complement latent stochasticity
- planning requires reward/value interfaces not present here
```

## Next Questions

1. Does causal attention retain longer histories better than compressing all history into `h_t`?
2. Does RSSM stochasticity represent genuine alternative futures, or only inject noise on this deterministic dataset?
3. How should KL balancing, discrete latents, and overshooting change stability and long-horizon error?
4. Can reconstruction be made semantically sharp without a task-specific Goal head?

The next implementation is `03_memory/04_transformer_memory`; controlled performance comparison remains deferred to Phase 90.

## References

### Learning Latent Dynamics for Planning from Pixels (PlaNet)

Authors: Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson. Year: 2018 (ICML 2019). Paper: https://arxiv.org/abs/1811.04551.

Used for: RSSM factorization, deterministic recurrent state, stochastic state, learned prior/posterior, sequence latent dynamics, and observation-free latent planning context. Implementation: `rssm.py`, `rssm_losses.py`, `train.py`. This repository omits PlaNet's reward model, CEM planning, latent overshooting, and benchmark-scale likelihood model.

### Auto-Encoding Variational Bayes

Authors: Diederik P. Kingma, Max Welling. Year: 2013. Paper: https://arxiv.org/abs/1312.6114.

Used for: reparameterized stochastic sampling and variational KL regularization. Implementation: `rssm.py::DiagonalGaussian.sample`, `rssm_losses.py::diagonal_gaussian_kl`.

### Dream to Control: Learning Behaviors by Latent Imagination

Authors: Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi. Year: 2019. Paper: https://arxiv.org/abs/1912.01603.

Used for: RSSM-based latent imagination context and the practical free-nats lineage. Implementation relation: `rssm.py::imagine`, `rssm_losses.py::rssm_loss`. No actor, value model, reward model, or imagined behavior learning is implemented here.

### Provenance statement

The continuous-Gaussian RSSM core is a **simplified educational implementation** of PlaNet-family ideas. The partial Grid World, green-channel weighting, ten-class Goal-state head, smoke configuration, and evaluation plots are **independent experimental modifications**. They must not be described as paper-reproduction results.
