# Understanding Probabilistic Ensembles

## What problem does this solve?

A probabilistic head can represent random outcome variation, but it cannot by itself state how uncertain its learned function is. An ensemble approximates multiple plausible dynamics functions and measures how much their predictions disagree.

## Before

One model outputs one Gaussian. Its variance mixes true process randomness with any errors caused by sparse data or extrapolation.

## After

Five bootstrap-trained models output five Gaussians. Average within-model variance estimates aleatoric uncertainty; variance of their means estimates epistemic uncertainty.

## Core Idea

- Aleatoric: “Even if I knew the correct model, the next outcome is random.”
- Epistemic: “I do not know which dynamics model is correct because evidence is incomplete.”

More data in a region should reduce epistemic disagreement. It should not remove irreducible aleatoric transition noise.

## Data Flow

```text
dataset D
 -> bootstrap D_0...D_4
 -> train model_0...model_4 independently
 -> same query to all models
 -> collect means and variances
 -> decompose predictive variance
 -> choose member identities for trajectory particles
```

## Mathematics

### Bootstrap

```text
D_m = N samples drawn with replacement from D of size N.
```

Why needed: each model sees a different empirical dataset, approximating uncertainty about which fitted function the finite data support.

### Ensemble predictive moments

```text
mu_bar = E_m[mu_m]
Var(y|x,D) approx E_m[Var_m(y|x)] + Var_m(mu_m(x)).
```

- first term: average member Gaussian variance = aleatoric estimate;
- second term: member-mean disagreement = epistemic estimate.

Why needed: law of total variance separates within-hypothesis randomness from between-hypothesis uncertainty.

### TS-infinity

```text
m_p ~ Uniform({1...E}) once per particle p
s_{t+1}^p ~ p_{m_p}(.|s_t^p,a_t) for all t.
```

Why needed: each particle follows one coherent sampled dynamics hypothesis through the whole rollout.

### TS1

```text
m_{p,t} ~ Uniform({1...E}) at every step t
s_{t+1}^p ~ p_{m_{p,t}}(.|s_t^p,a_t).
```

Why needed: it repeatedly mixes ensemble hypotheses. It propagates uncertainty differently and may average model identity over time.

## Code Mapping

| Concept | Code |
|---|---|
| bootstrap indices | `ensemble_dataset.py::bootstrap_indices` |
| member collection | `probabilistic_ensemble.py::ProbabilisticEnsemble.members` |
| total-variance identity | `ProbabilisticEnsemble.decompose` |
| TS∞/TS1 selection | `ProbabilisticEnsemble.rollout` |
| independent optimizers | `train.py::train` |
| OOD probes/decomposition | `evaluate.py::evaluate` |

## Important Components

### Independent initialization and optimization

Why necessary: perfectly tied models cannot disagree. Bootstrap data alone is not useful if all parameters are shared.

### Member mean disagreement

Why necessary: averaging only member variances discards epistemic information. Averaging only means hides all uncertainty.

### In-distribution versus OOD evaluation

Why necessary: an epistemic mechanism should be tested where data support changes. In-distribution NLL alone cannot establish OOD sensitivity.

### Particle model identity

Why necessary: a mixture distribution over dynamics must be propagated through nonlinear multi-step transitions; mean/variance at one step is insufficient.

## What happens if we remove it?

- Remove bootstrap and use identical initialization/data order: members may collapse to nearly identical functions.
- Use one member: epistemic variance is exactly zero by definition.
- Average variances but not mean disagreement: reports only aleatoric uncertainty.
- Average means and discard all variance: returns to overconfident point rollout.
- Use TS∞ only: may overrepresent persistent extreme models.
- Use TS1 only: may switch between mutually inconsistent hypotheses.
- Moment-match every mixture: can erase separated modes.
- Assume disagreement is perfect OOD detection: shared architecture/training bias can make all models confidently wrong together.

## Smoke-result interpretation

OOD epistemic std is 1.54x ID, so disagreement responds in the intended direction. However, the `uncertainty_decomposition.png` curve shows member aleatoric variance growing much faster OOD. Therefore:

1. the ensemble adds a distinct epistemic signal;
2. the learned decomposition is not semantically pure merely because the formula has two terms;
3. later calibration must assess both terms under changing data volume and distribution shift.

## What I Should Be Able to Explain

- Why is member-mean variance called epistemic?
- Why is average member variance called aleatoric?
- How does the law of total variance produce their sum?
- Why does bootstrap use replacement?
- Why can all ensemble members be wrong in the same way?
- What does an OOD/ID disagreement ratio establish and not establish?
- How do TS1 and TS∞ differ in model identity over a trajectory?
- Why can moment matching hide multimodality?
- Why did aleatoric variance dominate the OOD plot in this experiment?
- Why is an ensemble more expensive than one stochastic latent model?

## Questions

- Does increasing data density reduce only the epistemic term empirically?
- How many members are enough for stable disagreement?
- Should variance heads be regularized/frozen OOD to stop aleatoric absorption?
- Would randomized priors or diversity objectives improve shared-bias detection?
- How should risk-sensitive planning value epistemic versus aleatoric spread?
