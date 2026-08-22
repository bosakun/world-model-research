# Cross-Entropy Method (CEM) Planning

Status: completed on 2026-08-22. Exact compact-model planner smoke test; not a PlaNet/PETS benchmark reproduction.

## Purpose

Concentrate a Gaussian action-sequence proposal around elite predicted returns instead of spending every sample uniformly.

## Problem

Random shooting wastes samples in low-return regions and returned a limited improvement. CEM iteratively uses good candidates to update where the next candidates come from.

## Previous Model

`01_random_shooting` uses a proposal that never learns from candidate scores during a planning call.

## Hypothesis

Five elite-refit iterations should contract action standard deviation and find a stronger path with fewer candidates per iteration than the random-shooting smoke run.

## Architecture

```text
initialize mu=0,sigma=1
 -> sample 512 sequences
 -> exact model return
 -> top 64 elites
 -> refit/smooth mu,sigma
 -> repeat 5 times
 -> return best sequence seen
```

## Data Flow

Each iteration converts predicted returns into elite indices and then into the next proposal mean and standard deviation.

## Tensor Shapes

Samples `[512,10,2]`, scores `[512]`, elites `[64,10,2]`, mean/std `[10,2]`, iteration best `[5]`.

## Mathematics

```text
E=TopK({a_n},J(a_n)); mu_new=mean(E); sigma_new=std(E)
mu <- alpha mu +(1-alpha)mu_new; sigma similarly, alpha=0.1.
```

## Code Mapping

Distribution iteration: `cem.py::CEMPlanner.plan`; rollout objective: `../planning_core.py`; evidence: `evaluate.py`.

## Training

CEM has no learned weights. Its optimization iterations refit a proposal distribution online.

## Losses

There is no supervised loss; predicted return is the objective maximized by elite selection.

## Evaluation Interface

`python 07_planning/02_cem/evaluate.py` emits hyperparameters, per-iteration best scores, final proposal std, distance reduction, and plot.

## Smoke Test Results

Four tests passed. Distance `2.2672 -> 0.6069` (73.2% reduction), selected score `-11.866`, mean action std contracted to `0.272`. Raw iteration best was not monotonic (`-13.651` then `-13.721`) because each finite sample set differs; global best is retained.

## Failure Cases

- Gaussian proposal is unimodal and may collapse to one mode.
- Finite-sample iteration quality need not improve monotonically.
- Hard action clipping distorts the fitted Gaussian near bounds.
- Open-loop CEM still cannot correct execution/model error.

## Findings

Elite refitting concentrates search effectively on this smooth exact task.

## Limitations

There is no learned-model uncertainty, warm start, constraints, or comparison-controlled model-call accounting yet.

## Compare Later

Matched random shooting/CEM budgets; elite fraction, iterations, momentum, horizon, initialization/policy prior. Metrics: return, success, evaluations, latency, sensitivity to model bias.

## Final Model Candidate

```text
Candidate: Yes for continuous-action planning.
Reason: Strong derivative-free refinement with a simple world-model interface.
Advantages: sample concentration, bounded compute, no model gradients required.
Disadvantages: unimodal proposal, hyperparameters, local convergence.
Possible conflicts: ensemble particles multiply every candidate evaluation.
```

## Next Questions

How much does executing only the first action and replanning improve robustness? How should uncertainty/risk modify elite scores?

## References

PlaNet: Danijar Hafner et al., 2018, https://arxiv.org/abs/1811.04551. PETS: Kurtland Chua et al., 2018, https://arxiv.org/abs/1805.12114. Used for online CEM planning/trajectory-sampling context. This is a simplified exact-model implementation without learned latent dynamics or benchmark reproduction.
