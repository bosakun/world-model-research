# Understanding Partial Observability and Memory

## What problem does this solve?

It separates “a GRU was added” from “memory had information to use.” In the full Grid World, a frame reveals the Goal. Here the Goal can disappear from the local camera view while remaining part of the real world.

## State and observation are different

- **True state `s_t`** is the complete environment information needed by the simulator: Agent row/column and Goal row/column.
- **Observation `o_t`** is only what the Agent receives: a 3x3 agent-centred image with unknown outside cells.

The environment knows `s_t`; the model is allowed to receive only `o_t` and actions. Evaluation may inspect `s_t` afterwards to determine whether a prediction was correct.

## Before

In a full observation, the visible frame contains Agent and Goal. A memory-free model can often use current `z_t` and `a_t` alone.

```text
full image -> z_t -> f(z_t, a_t) -> predicted z_{t+1}
```

## After

The observation function hides content outside the local view.

```text
true state s_t -> local camera O -> partial image o_t -> z_t
                                                 |
past z/action history -> GRU hidden h_t --------+
```

`h_t` is expected to retain evidence such as “the green Goal was right of the Agent earlier” and how later actions changed the relative relationship.

## Partial Observability

Partial observability means an observation is not a complete state description. A single picture can be compatible with several true worlds. The key aliasing condition is:

```text
s_t^A != s_t^B but o_t^A = o_t^B
```

In this dataset, Goal-right and Goal-down histories have identical `o_2` after two left actions, yet their hidden Goal coordinates differ. The same present image therefore has different meanings depending on history.

## POMDP intuition

A POMDP formalizes the situation as hidden state, actions, transition, observations, and often rewards. You do not need all formal solution machinery to understand the key point here: the Agent cannot read all of `s_t`, so it needs to combine evidence over time. A Bayesian belief state would explicitly represent possible worlds; a GRU hidden state is a learned vector that may approximate useful historical information without explicit probabilities.

## Data Flow

1. The simulator keeps `s_t=(agent_row,agent_col,goal_row,goal_col)`.
2. The action updates only Agent position with boundary clipping.
3. `O(s_t)` maps the local 3x3 field into the central region of a 20x20 RGB image.
4. Goal pixels are drawn only when they lie within the local field; blue means unknown.
5. The Encoder maps `o_t` to `z_t`.
6. Simple Dynamics sees `(z_t,a_t)`; GRU additionally carries `h_t` into its next update.

## Mathematics

### True transition

```text
p_{t+1} = clip(p_t + delta(a_t))
g_{t+1} = g_t
s_t = (p_t, g_t)
```

`p_t` is Agent position, `g_t` is the static Goal coordinate, and `clip` prevents leaving the 5x5 grid. This determines real world state even when the Goal is hidden.

### Observation function

```text
o_t = O(s_t)
Goal is rendered iff max(|g_row-p_row|, |g_col-p_col|) <= 1
```

Why: this makes the camera local. No image operation writes the true Goal into an unknown cell outside this condition.

### Recurrent belief-like update

```text
h_{t+1} = GRUCell([z_t; a_t], h_t)
```

Why: `h_t` can carry earlier evidence absent from `z_t`. It is not automatically a probabilistically correct belief state and this experiment does not claim it is.

## Tensor Shapes

```text
observations: [B,T+1,3,20,20]
actions:      [B,T,4]
true_states:  [B,T+1,4]
full_worlds:  [B,T+1,3,20,20]
z:            [B,T+1,16]
h:            [B,T,64]
```

There are `T+1` observations/states because each of the `T` actions produces the following observation/state.

## Code Mapping

| Question | Code |
|---|---|
| Where is the real state stored? | `partial_env.py::WorldState` |
| What does the Agent get? | `PartialObservationGridWorld.render_partial_observation` |
| How is hidden Goal leakage prevented? | local-coordinate loop and `goal_is_visible` in `partial_env.py` |
| How are alias pairs guaranteed? | `partial_dataset.py::PartialObservationSequenceDataset` |
| How do existing models consume data? | `model_adapters.py` |
| How do I see the distinction? | `outputs/observation_sequence.png`, `outputs/aliasing_pair.png` |

## Important Components

### True state access only for evaluation

Without it, it would be impossible to prove that two equal observations hide different Goals. Giving it to the model would invalidate the POMDP setup.

### Agent-centred view

It removes a global-map shortcut and makes observations express relative local contents. Removing this and showing the whole map turns the task back toward full observability.

### Unknown representation

Unknown means “not observed,” whereas black local cells mean “observed empty.” Merging the two would make it harder to reason about what information was actually available.

### Paired aliases

Random local crops might rarely create a clear same-image/different-state case. The paired construction guarantees one, making a memory ablation falsifiable.

### Sequence dataset

A single transition contains no earlier Goal cue after it leaves view. The model must receive `o_0,a_0,...,o_t` so recurrence has evidence to retain. This is why transition-only data cannot test the intended memory mechanism.

## What happens if we remove it?

| Remove | Consequence |
|---|---|
| local observation function | current frame exposes Goal; memory question weakens |
| true-state evaluation tensor | cannot establish aliasing or hidden-goal accuracy |
| paired alias construction | memory-critical cases may be absent by chance |
| action history | the model cannot update remembered relative relations |
| GRU hidden carry | recurrent model reduces to no-memory processing |
| sequence structure | only one-step visual mapping remains; no history can be used |

## What I Should Be Able to Explain

- Can I state the difference between `s_t` and `o_t` for this exact environment?
- Which four values comprise true state, and which are hidden after `t=2`?
- Why can equal current observations have different true Goal positions?
- Why is the blue unknown region different from an observed empty cell?
- Why is a sequence, rather than isolated transitions, necessary for GRU memory?
- What could `h_t` preserve after the green Goal disappears?
- Why does compatibility with GRU not prove that GRU is better?
- Why must `full_worlds` and `true_states` stay out of model inputs?
- Why is this not an RSSM, PlaNet, or Dreamer implementation?

## Questions

- Does a trained GRU retain the early Goal direction after it disappears?
- Does it outperform a matched Simple Dynamics baseline on hidden-goal prediction?
- Does history shuffling or resetting `h_t` erase that advantage?
- What changes when multiple Goals, moving Goals, obstacles, or observation noise introduce uncertainty?
- When is a deterministic GRU insufficient and an RSSM stochastic state necessary?

