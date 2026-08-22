# Understanding Aleatoric Probabilistic Dynamics

## What problem does this solve?

The same state and action can genuinely produce different next states because of process noise, sensor noise, or unobserved factors. A point prediction says only where outcomes average. A probabilistic prediction also says how widely they vary.

## Before

```text
(s_t,a_t) -> one next-state estimate
```

MSE encourages the conditional mean. It supplies no calibrated transition distribution to sample or score.

## After

```text
(s_t,a_t) -> mean mu_{t+1} and variance sigma^2_{t+1}
          -> distribution p(s_{t+1}|s_t,a_t)
```

The variance changes with input, so the model can say one region/action is intrinsically noisier than another.

## Core Idea

Aleatoric uncertainty belongs to the data-generating process. Even a model with perfect parameters cannot predict the exact random noise draw before it happens. The best it can do is learn the conditional outcome distribution.

Epistemic uncertainty instead describes uncertainty about the model because data are limited. This single probabilistic network is designed to learn aleatoric uncertainty; it cannot reliably separate epistemic uncertainty by itself.

## Data Flow

```text
known state/action
  -> deterministic action displacement
  -> unknown Gaussian process-noise draw
  -> observed next state

model(state,action)
  -> mean + log variance
  -> NLL against observed next state
  -> calibrated Gaussian prediction
  -> mean rollout OR sampled rollout
```

## Mathematics

### Environment transition

```text
s_{t+1}=s_t+delta(a_t)+epsilon_t
epsilon_t~N(0,diag(sigma_true(s_t,a_t)^2)).
```

`sigma_true` increases horizontally as `x` increases, and vertical actions have greater vertical noise. Why needed: known heteroscedastic ground truth makes it possible to test whether learned uncertainty means what we claim.

### Predicted distribution

```text
p_theta(s_{t+1}|s_t,a_t)=N(mu_theta,diag(exp(l_theta)))
```

- `mu_theta`: learned next-state mean.
- `l_theta=log sigma^2`: learned log variance.

Why log variance: it maps a wide positive variance range to unconstrained real outputs and makes NLL numerically convenient.

### Soft variance bounds

```text
l_upper = l_max - softplus(l_max-l_raw)
l = l_min + softplus(l_upper-l_min).
```

Why needed: it maintains differentiable upper/lower bounds and avoids variance underflow/overflow. The bound values themselves remain trainable.

### Gaussian negative log-likelihood

```text
NLL = 1/2 sum_j [l_j + (y_j-mu_j)^2/exp(l_j) + log(2pi)].
```

- residual term divided by variance: being wrong is penalized more when claiming confidence.
- log-variance term: claiming large uncertainty everywhere is penalized.

Why needed: these opposing pressures jointly learn accuracy and appropriate spread.

### Reparameterized sample

```text
epsilon~N(0,I)
y_hat=mu+exp(l/2) epsilon.
```

Why needed: it creates trajectory particles and remains differentiable when later objectives backpropagate through samples.

### Coverage

```text
coverage(k) = fraction(|y-mu| <= k sigma).
```

For a calibrated Gaussian, reference values are about 68.3% at `k=1` and 95.4% at `k=2`. Coverage alone is insufficient: an extremely wide distribution can cover everything but be uninformative, so sharpness/NLL also matter.

## Code Mapping

| Concept | Code |
|---|---|
| true input-dependent noise | `stochastic_dataset.py::transition_noise_std` |
| stochastic environment | `stochastic_transition` |
| mean/logvar prediction | `probabilistic_dynamics.py::ProbabilisticDynamics.forward` |
| positive std/variance | `GaussianPrediction.std`, `.variance` |
| sampling | `GaussianPrediction.sample` |
| NLL | `probabilistic_losses.py::diagonal_gaussian_nll` |
| sequence propagation | `ProbabilisticDynamics.rollout` |
| coverage/correlation | `evaluate.py::evaluate` |

## Important Components

### Separate mean and variance heads

Why necessary: transition location and transition randomness are different quantities. One deterministic head cannot state both.

### Input-dependent variance

Why necessary: homoscedastic variance would report the same risk everywhere and miss state/action-specific noise.

### NLL instead of variance regression

Why necessary: in real tasks true variance labels are unavailable. NLL learns variance from repeated residual statistics. Here true variance is used only to verify the result.

### Sampled rollout

Why necessary: applying only conditional means hides distribution growth through time. Particles reveal possible trajectory spread.

## What happens if we remove it?

- Remove variance head: returns to point dynamics; no likelihood or risk-aware samples.
- Fix one global variance: cannot learn heteroscedastic regions/actions.
- Remove `log variance` term from NLL: model can inflate variance without cost.
- Remove residual/variance scaling: variance no longer controls claimed confidence.
- Remove bounds: optimization can encounter extremely tiny/large variance and unstable division/exponentiation.
- Always use mean rollout: aleatoric risk disappears from imagined trajectories.
- Call variance “epistemic”: conceptually wrong; extra data cannot remove the actual transition noise.

## What I Should Be Able to Explain

- What is the difference between aleatoric and epistemic uncertainty?
- Why can repeated identical inputs have different next states?
- Why does MSE learn a mean but not a distribution?
- How does each Gaussian NLL term prevent cheating?
- Why predict log variance instead of raw standard deviation?
- Why can continuous Gaussian NLL be negative?
- What does 1σ coverage measure, and what can it hide?
- Why is a predicted/true std correlation possible only because this environment is synthetic?
- Why does sampled rollout spread while mean rollout is repeatable?
- Why does the curve's OOD divergence motivate an ensemble rather than merely a larger variance head?

## Questions

- How should non-Gaussian or multimodal transition noise be modeled?
- Does diagonal covariance miss important coupled physical motion?
- How many repeated outcomes are needed to distinguish noise from mean-model error?
- How should aleatoric variance combine with ensemble disagreement mathematically?
- Which trajectory-sampling scheme preserves the two uncertainty types over long horizons?
