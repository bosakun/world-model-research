# Research Notes: Partial Observation

Date started: 2026-08-22

## Initial question

`01_gru` could carry history, but the whole image already revealed the Goal. The unresolved question was not “can GRU run?” but “does the dataset contain information that only memory can provide?”

## Key design decision

The local view alone was not considered enough. The dataset therefore includes paired histories: Goal-right vs Goal-down, same Agent start, same `left,left` action prefix, and exactly equal local observation at `t=2` after both Goals leave view. This makes the missing information a checked property instead of an intuition.

## Complete observation vs partial observation

- Full image: current frame nearly tells the model all relevant positions.
- Local image: current frame tells the model only a local neighbourhood; a previously seen Goal can disappear.
- Important nuance: “unknown” is not “empty.” Blue outer cells encode absence of observation, while dark central cells encode observed empty local space.

## Concrete memory example

At `t=0`, the green Goal is visible to the right. The Agent moves left twice. At `t=2`, the current local image contains no Goal. A hidden state can potentially preserve the old direction plus actions; a memory-free current-frame model cannot recover an unrendered Goal coordinate from pixels that are identical across two true worlds.

## Implementation issue found

Running `01_gru` and `02_partial_observation` tests together initially exposed Python module-name collisions (`env`, `dataset`) because both experiments are standalone folders. `02` now uses internally unique names (`partial_env.py`, `partial_dataset.py`) while retaining thin conventional `env.py` and `dataset.py` exports for readability. `01_gru` was not modified.

## Results

- 13 tests passed: 6 existing GRU tests and 7 new tests.
- Goal visibility: true at `t=0`, false by `t=2` in the reference sequence.
- Aliasing: two `t=2` observations are bitwise equal while `true_states[...,2:]` differs.
- Existing Simple Dynamics and GRU interfaces accept the new `[B,T+1,3,20,20]` observations and one-hot actions.
- Visual outputs clearly show true full world versus partial local view.

## Article material

- Figure pair: full world and local camera at the same time.
- Sequence figure: Goal visibly leaves the Agent's information set, not the world.
- Aliasing figure: identical present input, different hidden truth.
- Framing: a GRU is not useful merely because it has a hidden state; the environment must make history predictive.

## Guardrails for the next phase

- Do not feed `true_states` or `full_worlds` into either model.
- Do not call environment validation a GRU performance result.
- Compare matched models, not an optimized GRU against an untrained Simple model.
- Preserve the alias pair and add history-shuffle/hidden-reset ablations.
- Do not implement RSSM or Transformer memory before the controlled comparison.

