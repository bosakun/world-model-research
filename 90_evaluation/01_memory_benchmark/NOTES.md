# Research Notes

## Hypothesis and first attempt

- Expected all memory methods to exceed No Memory.
- Initial result: RSSM 3/3, Transformer 1/3, GRU 0/3.
- Cause audit found Goal heads for GRU/Transformer read the 16-D state simultaneously forced to equal the encoder of an identical local image. RSSM alone read `h+z`.

## Fairness correction

- Moved GRU Goal head to hidden state and Transformer Goal head to context token. Kept image decoder on the visual latent. RSSM already used its full state.
- Reran all 12 trainings rather than patching metrics.
- Final: No Memory 0.5; GRU 0.5/1/1; RSSM and Transformer 1/1/1.
- Reset/context-1 ablations all 0.5 after RSSM ablation correctly reset both `h` and `z`.

## Unexpected result

- No Memory image MSE is competitive. Most future local-view pixels are predictable without the hidden Goal.
- RSSM perfect semantic memory did not produce best h10 image MSE.
- GRU's single-seed collapse is precisely why three seeds mattered.

## Article material

- `memory_comparison.png` has horizon, alias, and ablation panels.
- Strong explanation: “Where the probe reads from is part of benchmark fairness.”
- Raw `per_seed_results.csv` supports a stability table.

## Later work

Equal parameter budgets, frozen probes, more seeds, stochastic transitions, and planning-based memory utility.
