# Temporal Abstraction: Learned Fixed-Horizon Macro Dynamics

Status: completed on 2026-08-22. This is an independent fixed-duration action-chunk model inspired by temporal abstraction; it is not a learned Options framework implementation.

## Purpose

Predict every fifth state from an ordered chunk of five primitive actions, reducing a 30-step imagination from 30 model applications to six macro transitions. Make the trade-off between fewer compositions and a harder/coarser transition explicit.

## Problem

Primitive-step world models repeatedly apply small transitions and accumulate error. A temporally abstract model can jump farther, but must summarize an action sequence and learn more nonlinear change per prediction.

## Previous Model

`01_latent_overshooting` applies the learned transition 30 times. Despite excellent one-step MSE and a five-step training objective, its horizon-30 error is large. This experiment changes temporal resolution rather than only increasing the training loss horizon.

## Hypothesis

Encoding five actions into one macro condition reduces recursive applications by 5x. It may reduce some compounding paths, but macro one-step error will be larger because each prediction covers more dynamics.

## Architecture

```text
action chunk [a_t,...,a_{t+4}] [5,4]
                   |
               GRU encoder
                   |
            chunk embedding [32]
                   + state s_t [2]
                   |
               residual MLP
                   |
             s_hat_{t+5} [2]

repeat 6 macro steps for primitive horizon 30
```

## Data Flow

```text
primitive oscillator sequence [B,31,2] and actions [B,30,4]
 -> boundary states at 0,5,10,...,30: [B,7,2]
 -> action chunks: [B,6,5,4]
 -> teacher-forced macro transition training
 -> autoregressive six-macro-step rollout
```

## Tensor Shapes

| Tensor | Shape | Meaning |
|---|---|---|
| primitive states/actions | `[B,31,2]`, `[B,30,4]` | unchanged environment trajectory |
| boundary states | `[B,7,2]` | every fifth state |
| action chunks | `[B,6,5,4]` | six ordered action sequences |
| chunk embeddings | `[B,6,32]` | GRU final hidden per chunk |
| teacher/rollout macro states | `[B,6,2]` | predictions for times 5...30 |

## Mathematics

For fixed duration `K=5`, the true macro transition is the composition

```text
s_{t+K}=F^K(s_t,a_t,...,a_{t+K-1}).
```

The learned approximation is

```text
c_t=GRU_actions(a_t,...,a_{t+K-1})
s_hat_{t+K}=s_t+g_theta([s_t,c_t])
L_macro=||s_hat_{t+K}-s_{t+K}||^2.
```

Action order matters because nonlinear state changes make the same action multiset produce different paths. The GRU encodes order.

## Code Mapping

| Concept | File / symbol |
|---|---|
| primitive-to-macro data transformation | `macro_dataset.py::chunk_sequences` |
| boundary/chunk dataset | `MacroSequenceDataset` |
| ordered chunk encoding | `macro_dynamics.py::ActionChunkEncoder` |
| macro transition | `MacroDynamics.forward` |
| six-step macro rollout | `MacroDynamics.rollout` |
| boundary metrics/plots | `evaluate.py::evaluate` |

## Training

```bash
.venv/bin/python 05_long_horizon/02_temporal_abstraction/train.py
.venv/bin/python 05_long_horizon/02_temporal_abstraction/evaluate.py
.venv/bin/python -m pytest -q 05_long_horizon/02_temporal_abstraction/tests
```

| Reproducibility item | Value |
|---|---|
| seed / dataset | 53 / `controlled-oscillator-v1-macro5` |
| train / validation | 256 / 64 sequences |
| chunk / horizon | 5 / 30 primitive steps = 6 macro steps |
| model | GRU action encoder 32 + residual MLP 2x64 |
| optimizer | Adam `1e-3`, batch 64 |
| epochs / steps | 100 / 400 |
| parameters | 10,178 |
| checkpoint/evaluation | format 1 gitignored / `python 05_long_horizon/02_temporal_abstraction/evaluate.py` |

## Losses

Teacher-forced macro MSE teaches only boundary-to-boundary prediction. There is no intermediate-state reconstruction, macro overshooting, uncertainty, reward, or termination loss. This isolates the fixed temporal abstraction.

## Evaluation Interface

Evaluation reports teacher-forced macro MSE and autoregressive error after primitive-equivalent horizons 5/10/15/20/25/30, plus model size and transition count. It visualizes only boundary states; it makes no claim about the unpredicted intermediate path.

## Smoke Test Results

Five tests passed: exact boundary/action reshape, invalid chunk rejection, chunk-encoder shape, full gradient flow, macro rollout shape, and no use of future boundary truth.

| Metric | Result |
|---|---:|
| train macro MSE `epoch 1 -> 100` | `0.135954 -> 0.015157` |
| validation macro MSE | 0.014978 |
| held-out teacher-forced macro MSE | 0.016751 |
| rollout MSE at primitive horizon 5/10/20/30 | 0.01131 / 0.10596 / 0.54136 / 1.18563 |
| macro applications for horizon 30 | 6 instead of 30 |

## Failure Cases

- One macro prediction is much less accurate than one primitive prediction.
- Boundary prediction does not reconstruct intermediate states, collisions, rewards, or termination inside a chunk.
- Fixed chunk size may cross meaningful events and cannot terminate early.
- Six recursive errors still compound to large horizon-30 error.
- Results use different training seed/model from overshooting; direct performance ranking is deferred.

## Findings

- Temporal abstraction reduces model applications, not automatically error.
- Ordered action encoding is necessary even for a fixed-duration macro.
- Coarse macro accuracy versus recursive depth is a concrete design trade-off.
- Planning later needs reward/termination summaries inside macro transitions, not boundary state alone.

## Limitations

- Open-loop fixed action chunks, not options with initiation set, internal policy, and termination condition.
- Fully observed deterministic state; no visual representation or uncertainty.
- Only `K=5`, one seed, no matched comparison.
- No hierarchy choosing between primitive and macro models.

## Compare Later

- Chunk sizes 1/2/5/10 under matched compute/data; primitive overshooting versus macro dynamics.
- Metrics: boundary error, intermediate reconstruction, transition applications, latency, parameter count, planning success.
- Expected advantage: fewer recursive world-model calls and shorter planning depth.
- Expected weakness: coarse error and hidden within-chunk events.
- Ablations: unordered action pooling, last action only, mean action counts, GRU/Transformer chunk encoder, macro overshooting.

## Final Model Candidate

```text
Candidate:
Undecided

Reason:
The mechanism reduces temporal depth, but coarse error and missing within-chunk events must be resolved before planning use.

Advantages:
- fivefold fewer transition applications
- ordered action-chunk representation
- compatible with hierarchical multi-scale models

Disadvantages:
- higher per-transition approximation error
- no intermediate predictions or early termination
- fixed duration is inflexible

Possible conflicts:
- reward/continuation models must aggregate within chunks
- primitive and macro dynamics can disagree at boundaries
- variable-duration options require a different interface
```

## Next Questions

1. How should reward and termination inside a macro be predicted?
2. Can the model choose its temporal scale dynamically?
3. Does macro overshooting improve six-step stability?
4. When is intermediate state reconstruction indispensable for physical safety?

## References

### Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning

Authors: Richard S. Sutton, Doina Precup, Satinder Singh. Year: 1999. DOI: https://doi.org/10.1016/S0004-3702(99)00052-1.

Used for: motivation for temporally extended actions and multi-timescale reasoning. This implementation has only fixed open-loop action chunks; it does not implement option initiation sets, closed-loop policies, termination conditions, SMDP value learning, or intra-option learning.

### Provenance statement

The action-chunk encoder and macro boundary predictor are an **independent simplified educational implementation**, not an Options paper reproduction.
