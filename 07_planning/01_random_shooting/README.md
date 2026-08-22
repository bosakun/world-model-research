# Random Shooting Planning

Status: completed on 2026-08-22. Planner-mechanism smoke test with an exact compact model; no learned-model performance claim.

## Purpose

Turn world-model rollout and reward/value scoring into action selection by uniformly sampling complete action sequences and selecting the highest predicted return.

## Problem

A world model predicts consequences but does not choose controls. Random shooting provides the smallest explicit planning baseline.

## Previous Model

Phase 06 produced reward/value/continuation outputs but no optimizer over actions.

## Hypothesis

With enough uniform candidates, the selected sequence should reduce Goal distance. Sampling efficiency should be weak in a multi-step continuous space, motivating CEM.

## Architecture

```text
current state -> sample N action sequences [N,H,2]
              -> model rollout all candidates
              -> discounted rewards + terminal value
              -> argmax score -> execute/return best sequence
```

## Data Flow

The planner constructs the whole candidate tensor, the model evaluates it in parallel, and `argmax` selects one sequence.

## Tensor Shapes

`state [4]`, candidates `[4096,10,2]`, predicted states `[4096,10,4]`, scores `[4096]`, selected actions `[10,2]`.

## Mathematics

```text
J(a_0:H-1)=sum_t gamma^t r(s_t,a_t)+gamma^H c_H V(s_H)
a*=argmax_{a in sampled candidates} J(a).
```

## Code Mapping

Exact model/objective: `../planning_core.py::PointWorldModel`. Sampling/argmax: `random_shooting.py::RandomShootingPlanner.plan`. Plot/metrics: `evaluate.py`.

## Training

No trainable planner parameters. The seed fixes sampling. The exact model intentionally isolates planning mechanics. Learned-model integration is deferred to Phase 90.

## Losses

There is no supervised loss; candidate returns are online optimization scores.

## Evaluation Interface

`python 07_planning/01_random_shooting/evaluate.py` records seed, candidates, horizon, discount, score distribution, and distance reduction. No checkpoint is needed.

## Smoke Test Results

Four tests passed. With 4,096 candidates and horizon 10, predicted Goal distance changed `2.2672 -> 1.3089` (42.3% reduction); selected score `-15.742` versus candidate mean `-20.855`.

## Failure Cases

- The sampled best path did not reach the Goal and did not halve distance in the deterministic test seed.
- Continuous search volume grows exponentially with action dimension/horizon.
- Open-loop execution cannot correct model/environment disturbance.
- Exact-model success would not imply learned-model robustness.

## Findings

Random shooting is a transparent baseline and parallelizes candidate evaluation, but wastes most samples.

## Limitations

This folder has no learned dynamics, uncertainty, constraints, or replanning.

## Compare Later

Compare against CEM/MPC with matched model evaluations. Metrics: achieved return/success, model calls, wall time, sensitivity to horizon/action dimension. Ablate candidate count, proposal distribution, terminal value.

## Final Model Candidate

```text
Candidate: Yes as a baseline; No as the default high-dimensional planner.
Reason: Minimal and auditable, but sample inefficient.
Advantages: simple, parallel, derivative-free.
Disadvantages: poor scaling, open-loop.
Possible conflicts: model-evaluation budget grows rapidly with ensembles/video models.
```

## Next Questions

Can elite-based proposal refitting concentrate samples on good action regions? Can replanning correct open-loop errors?

## References

### PETS

Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine. 2018. https://arxiv.org/abs/1805.12114. Used for sampling-based model predictive control context and as the lineage motivating CEM after shooting baselines. This exact-model random sampler is an independent educational implementation.
