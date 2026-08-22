# Understanding Random Shooting

## What problem does this solve?

It converts model predictions into a finite optimization problem over sampled action sequences.

## Before / After / Core Idea

Before: model answers “what if this action sequence?” After: sample many “what ifs,” score them, and choose the best observed candidate. There is no gradient and no guarantee of global optimality.

## Data Flow / Mathematics

```text
a^(n)_0:H ~ Uniform -> rollout model -> J_n -> n*=argmax J_n.
```

`J` includes discounted rewards and a terminal value so a short horizon values progress beyond its last explicit reward.

## Code Mapping

Sampling/selection: `random_shooting.py`. Rollout/score: `planning_core.py`. Evidence: `evaluate.py`.

## Important Components

- candidate diversity explores alternatives;
- terminal value reduces horizon truncation bias;
- continuation stops post-terminal accumulation;
- fixed seed makes smoke evidence reproducible.

## What happens if we remove it?

Without reward/value, sequences cannot be ranked. Without action bounds, samples may be invalid. Without enough candidates, good narrow regions are missed. Without replanning, errors persist.

## What I Should Be Able to Explain

- Why does random shooting scale poorly with horizon?
- What is scored, and why include terminal value?
- Why is selected score better than mean candidate score?
- Why does this smoke model use exact dynamics?
- Why is one seed not a comparison?

## Questions

How should sampling distributions incorporate policy priors, uncertainty penalties, and hard constraints?
