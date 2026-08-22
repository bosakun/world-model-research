# Understanding CEM Planning

## What problem does this solve?

CEM uses high-scoring samples to learn where to sample next within one planning call.

## Before / After / Core Idea

Random shooting has a fixed uniform proposal. CEM repeatedly refits a Gaussian to elite action sequences, converting sparse successful samples into a denser next search distribution.

## Data Flow / Mathematics

```text
q_i(a)=N(mu_i,sigma_i) -> samples -> returns -> elites -> q_{i+1}.
```

Momentum smooths refits; a std floor prevents immediate zero exploration.

## Code Mapping

`cem.py::CEMPlanner` implements sampling/top-k/refit/global-best retention. `planning_core.py` scores sequences.

## Important Components

Elite fraction controls selectivity; multiple iterations create refinement; action bounds keep controls valid; global-best retention protects against a worse later random batch.

## What happens if we remove it?

No refit becomes random shooting. Too few elites causes noisy collapse; too many dilute selection. Zero variance floor can freeze search. No global best can return a worse final sample.

## What I Should Be Able to Explain

- What distribution is being optimized?
- Why can raw best score decrease between iterations?
- Why is CEM derivative-free?
- What does elite fraction trade off?
- Why does Gaussian unimodality matter?

## Questions

Can mixture proposals, policy priors, or uncertainty-aware scores prevent bad local convergence?
