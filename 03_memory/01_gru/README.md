# GRU Latent Dynamics on a Fully Observable Grid World

Status: completed on 2026-08-22.

## Purpose

Introduce a recurrent hidden state into action-conditioned latent dynamics and verify one-step prediction, hidden-state propagation, and autoregressive multi-step rollout in an isolated PyTorch experiment.

## Hypothesis

GRU dynamics can carry information from earlier latent/action pairs in `h_t`. However, because this environment is deterministic and fully observable, the current observation is already a Markov state. A trained GRU is therefore expected to work, but memory is not expected to have a principled advantage over a matched memory-free model until partial observability creates state aliasing.

## Problem

A memory-free transition predicts only from the present:

```text
(z_t, a_t) -> predicted z_{t+1}
```

When two histories produce the same current observation but imply different hidden physical states, this mapping is ambiguous. A recurrent state allows the transition to condition on a learned summary of history.

## Previous Model

The audit found no previous implementation in this checkout. `baseline.py` therefore preserves the requested Simple Dynamics interface as a new, memory-free reference; it is not falsely labeled as recovered prior work. Likewise, no existing latent dimension could be reused, so this experiment explicitly chooses `latent_dim=16`.

## Architecture

```text
observation sequence [B,T+1,3,20,20]
             |
       shared CNN Encoder
             |
       z_0 ... z_T [B,T+1,16]
             |
 z_t [16] + one-hot a_t [4] + h_t [64]
             |
       GRUCell(20 -> 64)
             |
          h_{t+1} [64]
             |
       MLP prediction head
             |
    predicted z_{t+1} [16]
             |
       shared MLP Decoder
             |
 predicted observation [3,20,20]
```

Training uses teacher forcing: each recurrent step receives encoded ground-truth `z_t`. Evaluation rollout is autoregressive: after the seed `z_0`, each predicted latent becomes the next input while `h_t` is carried forward.

This is a deterministic recurrent latent model. It has no stochastic state, prior, posterior, KL divergence, reward model, actor, value model, or planner, and is not an RSSM/PlaNet/Dreamer reproduction.

## Tensor Shapes

Let `B=batch`, `T=8`, `C=3`, `H=W=20`, `D_z=16`, `D_a=4`, and `D_h=64`.

| Tensor | Shape | Meaning |
|---|---|---|
| observations | `[B,T+1,C,H,W]` | full RGB Grid World sequence |
| action indices | `[B,T]` | integers 0..3 |
| one-hot actions | `[B,T,D_a]` | GRU conditioning |
| encoded latents | `[B,T+1,D_z]` | current visual representations |
| GRU input at step t | `[B,D_z+D_a] = [B,20]` | concatenated latent/action |
| initial/final hidden | `[B,D_h] = [B,64]` | recurrent memory |
| hidden sequence | `[B,T,D_h]` | post-update hidden states |
| predicted next latents | `[B,T,D_z]` | one-step or rollout predictions |
| decoded predictions | `[B,T,C,H,W]` | predicted future observations |

## Mathematics

For `x_t = [z_t; a_t]`, PyTorch `GRUCell` computes:

```text
r_t = sigmoid(W_ir x_t + b_ir + W_hr h_t + b_hr)
u_t = sigmoid(W_iu x_t + b_iu + W_hu h_t + b_hu)
n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_t + b_hn))
h_{t+1} = (1-u_t) * n_t + u_t * h_t
predicted z_{t+1} = g_theta(h_{t+1})
```

`r_t` controls how much past state contributes to the candidate; `u_t` interpolates between the candidate and existing memory. Gate symbols vary across papers; these equations match the implemented PyTorch convention.

The total training loss is:

```text
L = lambda_rec * mean((Decoder(Encoder(o_t)) - o_t)^2)
  + lambda_pos * CE(cell_logits(Decoder(Encoder(o_t))), agent_cell_t)
  + lambda_dyn * mean((predicted z_{t+1} - stopgrad(Encoder(o_{t+1})))^2)
```

with `lambda_rec=1`, `lambda_pos=0.2`, and `lambda_dyn=2`. `cell_logits` pools decoded red-vs-other color evidence into the 25 mutually exclusive grid cells. This auxiliary term was added after aggregate MSE ignored the small moving agent and target-only pixel weighting encouraged red false positives everywhere. The stopped dynamics target prevents dynamics loss from training the encoder to collapse toward easy moving targets. Reconstruction keeps the latent grounded in the full image, and `tanh` bounds the jointly learned latent. Position supervision uses simulator-known colors/labels and is an independent educational modification, not the likelihood/ELBO objective of PlaNet or Dreamer.

## Code Mapping

| Concept/equation | Code |
|---|---|
| visual state `z_t = Encoder(o_t)` | `model.py::VisualEncoder` |
| reconstruction `Decoder(z_t)` | `model.py::VisualDecoder` |
| `x_t=[z_t;a_t]`, gates, hidden update | `model.py::GRUDynamics.step` / `torch.nn.GRUCell` |
| sequence teacher forcing | `model.py::GRUDynamics.forward` |
| autoregressive imagination path | `model.py::GRUDynamics.rollout` |
| `predicted z_{t+1}=g(h_{t+1})` | `model.py::GRUDynamics.prediction_head` |
| reconstruction/dynamics losses | `losses.py::world_model_loss` |
| memory-free reference | `baseline.py::SimpleDynamics` |
| deterministic transitions/rendering | `env.py::FullyObservableGridWorld` |
| fixed trajectory tensors | `dataset.py::GridSequenceDataset` |

## Training

From the repository root:

```bash
uv sync
uv run pytest
uv run python 03_memory/01_gru/train.py
uv run python 03_memory/01_gru/evaluate.py
```

The default smoke-scale run uses 256 training sequences, 64 validation sequences, length 8, batch 32, 40 epochs, Adam with learning rate `3e-3`, and seed 7. The autoencoder and GRU dynamics are jointly optimized.

## Losses

- Reconstruction MSE teaches Encoder/Decoder to preserve the full visual frame.
- Agent-cell cross-entropy teaches the decoded representation which single cell contains the small moving agent; removing it allowed low-MSE background shortcuts in this imbalanced image.
- Latent dynamics MSE teaches GRU + prediction head to map history, current latent, and action to the next encoded observation. Removing it leaves no future-prediction learning signal.
- There is deliberately no KL loss: there is no stochastic latent distribution, prior, or posterior in this experiment.

## Evaluation

- Held-out one-step latent MSE under teacher-forced hidden-state evolution.
- Held-out one-step decoded pixel MSE and agent-cell accuracy.
- Eight-step autoregressive rollout pixel MSE overall and by horizon.
- Visual comparison of truth and decoded rollout.
- Hidden shape, finite values, gradient flow, dataset consistency, environment boundaries, parameter count, and inference time.

The retained Simple Dynamics baseline is interface- and parameter-tested but is not yet trained as a performance comparator. A fair performance claim requires matched optimization and preferably the partial-observation experiment; a random-baseline number would be misleading.

## Results

The final seeded 40-epoch smoke run completed successfully:

| Metric | Result |
|---|---:|
| tests | 6 passed |
| trainable parameters (whole model) | 343,336 |
| GRU dynamics parameters | 21,712 |
| Simple Dynamics parameters | 6,544 |
| initial train total loss | 0.803508 |
| final train total loss | 0.252070 |
| final validation total loss | 0.278304 |
| validation reconstruction MSE | 0.003758 |
| validation position cross-entropy | 0.005982 |
| held-out one-step latent MSE | 0.141365 |
| held-out one-step pixel MSE | 0.005980 |
| held-out one-step agent-cell accuracy | 83.59% |
| 8-step rollout mean pixel MSE | 0.008376 |
| 8-step rollout mean agent-cell accuracy | 55.86% |
| mean GRU rollout time per sequence | 6.98 microseconds |

Rollout position accuracy by horizon was `75.0%, 76.6%, 64.1%, 54.7%, 51.6%, 50.0%, 37.5%, 37.5%`. Pixel MSE rose overall from `0.00633` at horizon 1 to `0.00989` at horizon 8 (not strictly monotonically for every sample/horizon). This confirms both a functioning autoregressive path and compounding long-horizon error.

The output figures and machine-readable metrics are in `outputs/`. Results are from one seed and do not support a GRU-vs-MLP superiority claim.

## Failure Cases

Observed failures:

- Initial plain-MSE run reached low aggregate error while omitting the small agent entirely.
- Equal active-pixel weighting still produced chance-level cell accuracy (~3.9%).
- Target-only high red weighting produced a red-everywhere false-positive shortcut.
- A small transposed-convolution Decoder kept position cross-entropy near random; the final MLP Decoder learned the global latent-to-cell mapping.
- Final autoregressive accuracy declined from 75.0% at horizon 1 to 37.5% at horizons 7–8, and decoded images accumulated color/position artifacts.

These failures show why pixel MSE and one-step metrics must be paired with semantic state accuracy and rollout visualization.

## Comparison

| Property | Simple Dynamics | This GRU Dynamics |
|---|---|---|
| Transition input | `z_t, a_t` | `z_t, a_t, h_t` |
| History | none | compressed into `h_t` |
| State carried during rollout | predicted `z_t` | predicted `z_t` and `h_t` |
| Necessary in this full-observation task | likely sufficient | likely redundant |
| Ambiguous partial observation | cannot disambiguate | potentially can disambiguate |

No empirical superiority claim is made before the matched comparison phase.

## Findings

- `z_t` and `h_t` are operationally distinct and their sequence/rollout shapes were verified: `[B,T,16]` predictions and `[B,T,64]` hidden states.
- Teacher-forced one-step position prediction succeeded at 83.6%, while autoregressive performance fell with horizon. One-step success did not guarantee long-rollout stability.
- Aggregate pixel MSE twice concealed semantically invalid representations; the agent-cell metric caught both failures.
- The final position auxiliary loss and MLP Decoder are educational modifications required by this tiny imbalanced renderer, not GRU or World Models paper mechanisms.
- Full observability means these results validate recurrent implementation but still do not demonstrate that memory was necessary.

## Limitations

- Fully observable deterministic dynamics do not test the central memory hypothesis.
- Small synthetic images and short sequences are a mechanism test, not a research-scale benchmark.
- Deterministic MSE prediction cannot express multiple plausible futures.
- Joint training does not include reward/value/policy/planning.
- Only one seed is used for the smoke run.
- The baseline is retained but controlled training comparison is deferred.

## Final Model Candidate

```text
Candidate:
Undecided

Reason:
The mechanism and rollout path can be validated here, but usefulness cannot be judged in a fully observable Markov environment without a matched baseline and a partial-observation task.

Advantages:
- fixed-size differentiable history summary
- explicit recurrent state carried through rollout
- low per-step cost relative to full attention over history

Disadvantages:
- sequential computation
- finite hidden bottleneck and possible forgetting
- extra state/reset semantics and parameters
- no uncertainty representation

Conflicts with other methods:
- a future Transformer may replace rather than complement GRU memory
- RSSM will combine recurrent deterministic state with stochastic state, changing objectives and state semantics
```

## Next Questions

- Does the trained model actually use `h_t`, or can current `z_t` explain every transition?
- Under identical training, does GRU beat Simple Dynamics on complete observations?
- What partial-observation construction yields identical frames with history-dependent next states?
- How long can useful information survive in `h_t`?
- Should the decoder reconstruct from `z_t`, `h_t`, or their combination in later models?
- When do deterministic dynamics require stochastic prior/posterior states?

The next experiment should be `03_memory/02_partial_observation` because it creates the missing causal test: hide enough state that current observation alone is ambiguous, then compare models under matched data and optimization.

## References

### Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation

Authors: Kyunghyun Cho, Bart van Merriënboer, Çağlar Gülçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio. Year: 2014. Paper: https://aclanthology.org/D14-1179/; DOI: https://doi.org/10.3115/v1/D14-1179.

Used for: the gated recurrent unit family and interpretation of reset/update gating. Implementation: `model.py::GRUDynamics` uses `torch.nn.GRUCell`, whose exact gate equations are documented above. The original paper addressed machine translation; the action-conditioned latent transition here is an independent educational adaptation.

### World Models / Recurrent World Models Facilitate Policy Evolution

Authors: David Ha, Jürgen Schmidhuber. Year: 2018. Papers: https://arxiv.org/abs/1803.10122 and https://arxiv.org/abs/1809.01999.

Used for: World Model context in which a visual representation is followed by a recurrent temporal model and rollouts. Implementation relation: `model.py` and `evaluate.py`. Difference: the paper's temporal model is an MDN-RNN over VAE latents; this experiment uses deterministic GRU latent MSE, no mixture density output, no VAE, and no controller.

### Learning Latent Dynamics for Planning from Pixels (PlaNet)

Authors: Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson. Year: 2018 (published at ICML 2019). Paper: https://arxiv.org/abs/1811.04551.

Used for: conceptual distinction between deterministic recurrent state and learned latent dynamics. Future implementation target: `03_memory/03_rssm/`. Difference: this code has no RSSM stochastic state, prior/posterior, variational objective, latent overshooting, reward prediction, or planning. It is not PlaNet.

### Dream to Control: Learning Behaviors by Latent Imagination

Authors: Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi. Year: 2019. Paper: https://arxiv.org/abs/1912.01603.

Used for: context for recurrent state-space rollouts and the later progression from a learned world model to latent imagination. Difference: this implementation only trains deterministic observation dynamics; it has no stochastic RSSM state, reward/value models, actor, or imagination-based policy optimization and is not Dreamer.
