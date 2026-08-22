# Transformer Memory: Causal Attention over Latent/Action History

Status: completed on 2026-08-22. This is a small educational causal world model inspired by TransDreamer and the Transformer architecture. It is not a TransDreamer reproduction.

## Purpose

Replace a single recurrent memory vector with explicit causal access to a latent/action sequence. The implementation makes tokenization, positional information, causal masking, next-latent prediction, autoregressive rollout, and attention visualization concrete.

## Problem

GRU and RSSM recurrent paths must continually compress history into one deterministic vector. That is efficient, but accessing one old event requires it to survive every intervening update. Self-attention offers a different mechanism: the current prediction can compute content-dependent weights over all retained history tokens.

## Previous Model

- `01_gru`: deterministic `h_t` updated recurrently.
- `03_rssm`: deterministic recurrent `h_t` plus stochastic `z_t`, predictive prior, observation posterior, and KL.

This experiment is deterministic and removes RSSM prior/posterior stochasticity to isolate the memory architecture question. It retains the same partial-observation sequence contract.

## Hypothesis

A causal Transformer can use explicit latent/action history to predict future latent states without a GRU hidden state. Positional information should preserve order, while the causal mask should prevent future leakage. Long-range advantages over recurrent memory are only hypotheses until Phase 90.

## Architecture

```text
o_t [3,20,20] -> CNN Encoder -> z_t [16]
a_t [4] ---------------------------+
                                      |
                 concat[z_t,a_t] [20]
                                      |
                         linear projection [64]
                                      + learned position p_t [64]
                                      |
                           token x_t [64]
                                      |
                    2 x causal Transformer block
                    (4-head attention + feed-forward)
                                      |
                         context c_t [64]
                         /                  \
              next-latent head          attention map
                    |                         |
              z_hat_{t+1} [16]          [layers,B,heads,T,T]
                    |
             decoder / Goal-state head
                    |
             o_hat_{t+1} / logits [10]
```

During teacher forcing, each token uses the encoded true `z_t`. During rollout, each predicted `z_hat_{t+1}` is appended as the next token input. No future observation is encoded.

## Data Flow

```text
partial observations o_0...o_T + actions a_0...a_{T-1}
    -> encode all observations for training targets
    -> tokens ([z_0,a_0]+p_0)...([z_{T-1},a_{T-1}]+p_{T-1})
    -> causal self-attention (token t sees only tokens <= t)
    -> predict z_1...z_T
    -> decode future images and semantic Goal state

rollout:
o_0 -> z_0 -> token(z_0,a_0) -> z_hat_1
                append token(z_hat_1,a_1) -> z_hat_2 -> ...
```

## Tensor Shapes

With `B=32`, `T=6`, latent `D_z=16`, action `D_a=4`, model width `D=64`, `H=4` heads, and `L=2` layers:

| Tensor | Shape | Meaning |
|---|---|---|
| observations | `[B,T+1,3,20,20] = [32,7,3,20,20]` | partial image sequence |
| actions | `[B,T,4] = [32,6,4]` | one-hot action sequence |
| encoded latents | `[B,T+1,16] = [32,7,16]` | frame representation |
| concatenated content | `[B,T,20]` | `[z_t,a_t]` |
| input/context tokens | `[B,T,64] = [32,6,64]` | projected content plus position |
| attention | `[L,B,H,T,T] = [2,32,4,6,6]` | query-to-key weights |
| predicted next latents | `[B,T,16]` | `z_hat_{t+1}` |
| future images | `[B,T,3,20,20]` | decoded predictions |
| Goal logits | `[B,T,10]` | nine visible locations plus not-visible |

## Mathematics

Token construction:

```text
x_t = W_x [z_t,a_t] + p_t.
```

For one attention head:

```text
Q=XW_Q, K=XW_K, V=XW_V
A = softmax(QK^T / sqrt(d_k) + M)
Attention(X) = AV,
```

where causal mask `M_ij=0` for `j<=i` and `-infinity` for `j>i`. Therefore the representation used to predict `z_{t+1}` cannot access token `t+1` or later.

Each pre-normalized residual block computes:

```text
Y = X + MultiHeadAttention(LayerNorm(X), M)
C = Y + FFN(LayerNorm(Y)).
```

The prediction and smoke objective are:

```text
z_hat_{t+1} = W_o c_t
L = L_reconstruct + L_future_image
  + 0.5 ||z_hat_{t+1} - stopgrad(z_{t+1})||^2
  + 0.1 L_goal.
```

Image terms use RGB channel weights `[1,20,1]` so the small green Goal is not dominated by background pixels. The Goal head and image weighting are independent modifications, not TransDreamer mechanisms.

## Code Mapping

| Concept | File / class / function |
|---|---|
| image representation | `transformer_memory.py::VisualEncoder`, `VisualDecoder` |
| latent/action tokenization | `TransformerMemoryDynamics.tokenize` |
| positional information | `position_embedding` |
| causal mask | `TransformerMemoryDynamics.causal_mask` |
| attention + residual FFN | `CausalTransformerBlock.forward` |
| teacher-forced next-latent sequence | `TransformerMemoryDynamics.forward` |
| autoregressive sequence rollout | `TransformerMemoryDynamics.rollout` |
| integrated image model | `TransformerMemoryWorldModel` |
| loss terms | `transformer_losses.py::transformer_world_model_loss` |
| partial dataset adapter | `transformer_dataset.py::build_transformer_dataset` |
| training/checkpoint | `train.py::train` |
| metrics/attention plots | `evaluate.py::evaluate` |

## Training

```bash
.venv/bin/python 03_memory/04_transformer_memory/train.py
.venv/bin/python 03_memory/04_transformer_memory/evaluate.py
.venv/bin/python -m pytest -q \
  03_memory/01_gru/tests \
  03_memory/02_partial_observation/tests \
  03_memory/03_rssm/tests \
  03_memory/04_transformer_memory/tests
```

Reproducibility record:

| Item | Value |
|---|---|
| seed / dataset | 29 / `partial-observation-v1` |
| data | 128 train, 32 validation; six actions/seven images |
| Transformer | width 64, 4 heads, 2 layers, FFN 128, max context 16, dropout 0 |
| optimizer | Adam, learning rate `3e-3` |
| batch / epochs / steps | 32 / 40 / 160 |
| parameters | 406,794 |
| checkpoint | `outputs/checkpoint.pt`, format 1, gitignored |
| evaluation | `python 03_memory/04_transformer_memory/evaluate.py` |

## Losses

- Autoencoder reconstruction keeps each encoded latent decodable.
- Future-image weighted MSE teaches causal contexts to predict visible consequences.
- Detached next-latent MSE gives a stable representation-space transition target without letting the target encoder chase the predictor in that term.
- Goal-state cross-entropy tests whether predicted latent state retains the small partial-observation semantic signal.
- There is no KL because this experiment is deterministic; uncertainty is handled in later phases.

## Evaluation Interface

`evaluate.py` reports autoencoder reconstruction, teacher-forced one-step prediction, fully autoregressive rollout by horizon, Goal **state-head** accuracy, latent/token/attention shapes, parameter count, and plots. The teacher-forced and rollout paths are named separately to prevent future-observation conditioning from being confused with open-loop prediction.

## Smoke Test Results

All 29 tests across the four Memory experiments passed. Transformer tests verify shapes, finite forward/backward values, gradient flow through encoder/decoder/token projection/position/attention/prediction/state head, exact zero future attention, causal invariance under changed future tokens, positional distinction, context bounds, autoregressive rollout, and dataset compatibility.

| Metric | Result |
|---|---:|
| train total `epoch 1 -> 40` | `1.794921 -> 0.043907` |
| final validation total | 0.035018 |
| final validation weighted future image | 0.011991 |
| final validation latent prediction | 0.000094 |
| autoencoder plain pixel MSE | 0.001481 |
| teacher-forced one-step plain pixel MSE | 0.000964 |
| mean six-step autoregressive pixel MSE | 0.000964 |
| one-step / rollout state-head Goal accuracy | 1.000 / 1.000 |

The final-layer attention image has an exactly masked upper triangle and nonzero weights over available history. This proves causality and access, not that every attended token is causally necessary.

## Failure Cases

- Rollout images contain faint green traces after the Goal is out of view. The semantic state head remains correct, so decoder fidelity and state fidelity must be distinguished.
- The tiny dataset has a six-token horizon; it cannot demonstrate a long-range Transformer advantage.
- Attention weights are not automatically explanations or causal importance scores.
- Learned absolute positions are limited to `max_context=16`; longer inference requires a design change or sliding-window interpretation.
- Autoregressive latent error can compound because rollout consumes its own predictions.

## Findings

- A no-GRU causal architecture can process the repository's latent/action sequence contract and roll forward without future observations.
- The mask is correct both numerically and visually.
- Explicit attention exposes which stored token positions are available, unlike a single opaque recurrent hidden state.
- On this smoke task, semantic prediction is easier than pixel-perfect decoding.

## Limitations

- Not TransDreamer's Transformer State-Space Model: no stochastic state, prior/posterior, reward model, actor/value, or shared Transformer policy.
- Dense attention costs quadratic time/memory in context length.
- Small deterministic Grid World, one seed, no matched recurrent comparison.
- The joint encoder can still choose a task-specific compact representation; no pretrained tokenizer.

## Compare Later

- Compare No Memory, GRU, RSSM, and Transformer Memory on matched splits/seeds.
- Metrics: one/5/10-step error, hidden-Goal probe, long-context recall, parameters, training/inference latency, peak memory, and stability.
- Expected advantage: direct content-dependent access to retained history and parallel teacher-forced training.
- Expected weakness: quadratic attention, fixed context, autoregressive recomputation, and greater data demand.
- Ablations: remove position, remove action token, disable causal mask only as a leakage control, truncate context, shuffle history, inspect per-head versus averaged attention, and replace autoregressive predictions with true latents.

## Final Model Candidate

```text
Candidate:
Undecided

Reason:
The causal memory mechanism works, but a six-step smoke sequence cannot justify replacing recurrent memory.

Advantages:
- explicit access to all retained history tokens
- causal teacher-forced sequence processing is parallelizable
- attention can be inspected by layer and head

Disadvantages:
- O(T^2) attention and repeated rollout computation
- fixed maximum context and position scheme
- deterministic latent lacks RSSM uncertainty semantics

Possible conflicts:
- may replace rather than complement RSSM's recurrent deterministic state
- may require cache-aware rollout before planning
- later video tokenization may dramatically increase sequence length
```

## Next Questions

1. Does explicit attention outperform recurrent compression only when the useful clue is much farther in the past?
2. Should a future model combine Transformer context with a stochastic prior/posterior rather than choose one mechanism?
3. How should KV caching and relative/rotary positions change rollout cost and context extrapolation?
4. Which attention patterns survive history-shuffle or token-drop ablations?

Memory mechanisms are now implemented; controlled comparison remains deferred to Phase 90. The next mechanism phase is probabilistic/ensemble uncertainty.

## References

### TransDreamer: Reinforcement Learning with Transformer World Models

Authors: Chang Chen, Yi-Fu Wu, Jaesik Yoon, Sungjin Ahn. Year: 2022. Paper: https://arxiv.org/abs/2202.09481.

Used for: motivation for replacing recurrent world-model dynamics with a Transformer State-Space Model and studying long-range memory access. Corresponding code: `transformer_memory.py::TransformerMemoryDynamics`. This experiment does not reproduce TransDreamer's stochastic state-space formulation or transformer policy.

### Attention Is All You Need

Authors: Ashish Vaswani et al. Year: 2017. Paper: https://arxiv.org/abs/1706.03762.

Used for: scaled dot-product multi-head self-attention, positional information, causal masking, residual connections, and feed-forward blocks. Corresponding code: `CausalTransformerBlock`, `tokenize`, and `causal_mask`.

### Provenance statement

The causal-attention mechanism is a **simplified educational implementation** informed by the papers above. Latent/action token grouping, the partial Grid World objective, Goal-state head, green-channel weighting, and evaluation interface are **independent experimental modifications**.
