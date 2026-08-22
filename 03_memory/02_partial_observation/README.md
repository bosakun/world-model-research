# Partial Observation Grid World: Making Memory Necessary

Status: completed on 2026-08-22. This is an independent educational environment, not a reproduction of PlaNet, Dreamer, or World Models.

## Purpose

`01_gru` proved that a GRU transition can be implemented and rolled out, but its fully observable image almost identifies the whole state. This experiment deliberately hides the world outside an agent-centred 3x3 view, creating cases where the current frame alone cannot reveal the true Goal position.

Hypothesis: in these sequences, a model that carries observation/action history in a GRU hidden state can eventually outperform a memory-free `(z_t, a_t) -> z_{t+1}` model. This experiment creates and verifies the condition; it does **not** train or compare the two models yet.

## Full Observability vs Partial Observability

| Property | 01 fully observable world | This partial-observation world |
|---|---|---|
| Agent input | whole 5x5 world | only agent-centred 3x3 local view |
| Goal outside local view | impossible | rendered as unknown; no Goal pixel leaked |
| Agent position in observation | global image position | always centre of local view |
| Evaluation access | visible image | separate true state and full-world image |
| Current image sufficient for Goal location | usually yes | deliberately no |

The visible 3x3 patch is drawn into the middle of a fixed 5x5 / 20x20 canvas so it remains compatible with the completed visual encoder. Blue outer cells mean **unknown**, not empty world cells.

## Why Memory Matters

Two paired sequences start with the same Agent position but different, initially visible Goals: right `(2,3)` or down `(3,2)`. Both execute the same actions: `left`, `left`.

```text
t=0: goal right  OR  goal down  (both visible)
t=1: Agent moves left
t=2: both goals are outside the 3x3 view

partial_observation_A(t=2) == partial_observation_B(t=2)
true_goal_A != true_goal_B
```

The current frames at `t=2` are bitwise identical, as the test and generated `aliasing_pair.png` verify. A memory-free model gets the same current input for both worlds, so it cannot infer two different hidden Goal locations from that input alone. A GRU can, in principle, retain the earlier visual cue and intervening actions in `h_t`.

## POMDP intuition

In a fully observable MDP, the state given to an agent contains enough information for the next-state distribution and decision. In a partially observable MDP (POMDP), the environment has a true state `s_t`, but the agent receives an observation `o_t` sampled/generated from it:

```text
s_t -- observation function O --> o_t
```

Different `s_t` can produce the same `o_t`. The agent therefore needs a belief or a learned history summary rather than treating one frame as the whole world. Here, the true Goal coordinate remains in `s_t` but is omitted from `o_t` once outside the local field of view. This is a concrete visual POMDP-like construction, not a full POMDP planning benchmark.

## Data Flow

```text
true world state s_t = (agent_row, agent_col, goal_row, goal_col)
                         |
                         | evaluation only: full_world [3,20,20]
                         v
agent-centred observation function O(s_t)
                         |
                         v
partial observation o_t [3,20,20]
                         |
                    existing Encoder
                         |
                      z_t [16]
                         |
   action a_t [4] + history-dependent GRU hidden h_t [64]
                         |
                         v
                  predicted z_{t+1}
```

`SimpleDynamics` can instead use only `(z_t,a_t)`. Both are instantiated from unmodified `01_gru` code through `model_adapters.py`.

## Tensor Shapes

Let `B=batch`, `T=6`, `C=3`, `H=W=20`, `D_a=4`, `D_z=16`, `D_h=64`.

| Tensor | Shape | Meaning |
|---|---|---|
| partial observations | `[B,T+1,C,H,W]` | input available to the model |
| full worlds | `[B,T+1,C,H,W]` | evaluation/visualization only; never model input |
| action indices | `[B,T]` | `up/down/left/right` integers |
| one-hot actions | `[B,T,D_a]` | dynamics conditioning |
| true states | `[B,T+1,4]` | `(agent_row, agent_col, goal_row, goal_col)` |
| goal visibility | `[B,T+1]` | evaluation metadata |
| encoded latent | `[B,T+1,D_z]` | `01_gru` Encoder output |
| Simple predictions | `[B,T,D_z]` | memory-free path |
| GRU predictions | `[B,T,D_z]` | recurrent path |
| GRU hidden states | `[B,T,D_h]` | history representation |

## Mathematics

The environment state and observation function are:

```text
s_t = (p_t, g),                  p_t=(agent row, agent col), g=(goal row, goal col)
p_{t+1} = clip(p_t + delta(a_t))
o_t = O(s_t) = render_3x3_centred_on(p_t, g) + unknown_elsewhere
```

`g` stays fixed within an episode. If `|g_row-p_row| > 1` or `|g_col-p_col| > 1`, it is not rendered in the observation. The evaluator still retains it in `true_states`.

For an alias pair at time `t`, the required property is:

```text
s_t^A != s_t^B,  O(s_t^A) = O(s_t^B)
```

This repository tests equality exactly using `torch.equal`.

## Code Mapping

| Concept | Implementation |
|---|---|
| true state, transition, observation function | `partial_env.py::PartialObservationGridWorld` |
| conventional import entry point | `env.py` |
| paired sequence generation | `partial_dataset.py::PartialObservationSequenceDataset` |
| conventional dataset entry point | `dataset.py` |
| access to full truth | `true_states`, `full_worlds`, `goal_visible` dataset tensors |
| unmodified Simple/GRU compatibility | `model_adapters.py` |
| full vs local and aliasing images | `visualize.py` |

## Experiments

This phase validates environment and data properties, not learned performance:

1. Run the deterministic state transition and local observation function.
2. Verify that the Goal is visible at `t=0` and outside the view at `t=2`.
3. Generate paired histories with equal `o_2` but different `s_2` Goal coordinates.
4. Feed the tensors through the existing Simple Dynamics and GRU world-model interfaces without changing `01_gru`.

Run from repository root:

```bash
uv run pytest -q 03_memory/01_gru/tests 03_memory/02_partial_observation/tests
uv run python 03_memory/02_partial_observation/visualize.py
```

## Results

All 13 tests passed: six original `01_gru` tests plus seven new partial-observation tests. The completed Simple Dynamics and GRU paths both accepted the partial sequence tensors:

```text
encoded latents:     [2, 7, 16]
Simple predictions:  [2, 6, 16]
GRU predictions:     [2, 6, 16]
GRU hidden states:   [2, 6, 64]
```

The aliasing pair at `t=2` was exactly identical (`True`) while its true Goal coordinates differed. Generated evidence:

- `outputs/full_world.png`
- `outputs/partial_observation.png`
- `outputs/observation_sequence.png`
- `outputs/aliasing_pair.png`

## Failure Cases

- A local observation does not by itself prove a trained GRU uses memory; a controlled comparison is still required.
- If the Goal re-enters the 3x3 view, the present frame can resolve its location again.
- The current world has no obstacles, stochastic dynamics, distractors, or reward/value objective.
- Unknown border cells communicate that information is unavailable; later work should avoid mistaking the unknown encoding itself for an absolute-position signal.
- `true_states` and `full_worlds` must never be supplied as learning inputs, or the partial-observation experiment leaks its answer.

## Comparison

The comparison is intentionally deferred. It must train `SimpleDynamics` and `GRUDynamics` under matching Encoder, data, optimization, and parameter-accounting conditions. The causal ablations should include hidden-state reset and history shuffling.

## Findings

The environment now has a checked information gap: at `t=2`, one current image corresponds to two different true Goal coordinates. It therefore supplies a valid minimal setting in which recurrent history can have information unavailable to a memory-free transition.

## Limitations

- This is a tiny deterministic synthetic environment and a hand-designed aliasing construction.
- The Goal is static; only visibility, not goal dynamics, is hidden.
- It does not contain a learned belief state, uncertainty distribution, RSSM prior/posterior, KL loss, planning, reward model, or policy.
- Compatibility confirms tensor interfaces only; it is not a performance result.

## Final Model Candidate

```text
Candidate:
Yes, as an evaluation environment; Undecided, as an integrated-model mechanism.

Reason:
It makes the missing-information condition observable and testable, but does not yet establish a GRU advantage.

Advantages:
- exact true-state access for evaluation
- no visual leakage outside the 3x3 view
- deterministic paired aliasing case
- unchanged compatibility with the previous visual models

Disadvantages:
- deliberately narrow and synthetic
- color-coded, hand-authored semantics
- does not test uncertainty or planning

Conflicts with other methods:
- none; it is an environment/dataset, not a replacement memory architecture
```

## Next Question

**Is GRU really superior to a memory-free model under partial observation?**

The next permitted research step is `03_memory/05_comparison/`: train `No Memory / Simple Dynamics` and `GRU` with the same partial-observation sequences and evaluate hidden-goal prediction, one-step/rollout error, horizon, parameter count, runtime, history shuffling, and hidden reset. Do not attribute an advantage to memory until that controlled test is complete.

## References

### Planning and Acting in Partially Observable Stochastic Domains

Authors: Leslie Pack Kaelbling, Michael L. Littman, Anthony R. Cassandra. Year: 1998. Paper: https://doi.org/10.1016/S0004-3702(98)00023-X.

Used for: POMDP framing, distinction between state and observation, finite-memory-controller context. Implementation: `partial_env.py`, `partial_dataset.py`. This environment is an independent learning example, not an implementation of the paper's planning algorithms.

### World Models

Authors: David Ha, Jürgen Schmidhuber. Year: 2018. Paper: https://arxiv.org/abs/1803.10122.

Used for: visual representation followed by recurrent temporal-model context. This experiment does not reproduce its VAE/MDN-RNN/controller design.

### Learning Latent Dynamics for Planning from Pixels (PlaNet)

Authors: Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson. Year: 2018 (ICML 2019). Paper: https://arxiv.org/abs/1811.04551.

Used for: relation between partial observability and recurrent latent state. This experiment has no stochastic state, prior/posterior, KL, overshooting, reward prediction, or planning; it is not PlaNet.

### Dream to Control: Learning Behaviors by Latent Imagination

Authors: Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi. Year: 2019. Paper: https://arxiv.org/abs/1912.01603.

Used for: recurrent world-model and later imagination context. This experiment has no Dreamer actor, value model, reward model, stochastic RSSM state, or imagined behavior optimization.

