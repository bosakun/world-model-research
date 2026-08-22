# Understanding Actor–Critic Learning from Imagination

## What problem does this solve?

It turns a predictive world model into a reusable action policy without requiring every actor update to execute actions in the physical environment.

## Before

CEM searched from scratch on each planning call. The learned world predicted futures, but no network had learned “given this latent, act this way.”

## After

An actor generates differentiable actions inside the frozen world model. A critic bootstraps long-horizon predicted return. Only after behavior training is finished is the policy audited in exact dynamics.

## Core Idea

If the model is differentiable, predicted consequences provide a training signal for action parameters. This cheaply creates many trajectories—but the policy optimizes the model, not reality. Any systematic model error becomes a potential strategy.

## Data Flow

```text
o_0 -> frozen encoder -> z_0
z_t -> actor -> a_t -> frozen dynamics -> z_(t+1) -> frozen reward
                              repeated H times
trajectory -> target critic + lambda return -> actor and critic objectives
```

## Mathematics

### Reparameterized action

`u=mu_phi(z)+sigma_phi(z)epsilon`, `a=tanh(u)`, `epsilon~N(0,I)`.

- The stochastic policy explores.
- `rsample` expresses the sample as differentiable noise transformation.
- `tanh` enforces action bounds.

### Imagination

`z_(t+1)=f_theta(z_t,a_t)`, `r_hat_t=r_theta(z_(t+1))`.

- `theta` is frozen world-model state.
- Gradients can pass through a frozen module to its inputs even though its parameters do not update.
- This lets actor parameters learn how their actions alter predicted futures.

### Lambda return

`G_t^lambda=r_hat_t+gamma[(1-lambda)V_bar(z_(t+1))+lambda G_(t+1)^lambda]`.

- `lambda=0`: one-step target with heavy critic bootstrap.
- `lambda=1`: long sampled return with terminal bootstrap.
- intermediate λ balances bias from the critic and accumulated model/reward error.

### Objectives

`L_actor=-E[G^lambda]-eta H(pi)` and `L_critic=E[(V(z_t)-sg(G_t^lambda))^2]`.

Entropy resists premature policy collapse. Stop-gradient makes λ-return a critic target instead of allowing the critic to move the target itself.

### Target critic

`bar_psi <- rho bar_psi + (1-rho)psi`.

It changes more slowly than the online critic, reducing feedback between value predictions and their own bootstrapped targets.

## Code Mapping

- `behavior.py::GaussianActor`: distribution, reparameterization, bounded action
- `behavior.py::Critic`: scalar value
- `imagination.py::imagine`: latent trajectory loop
- `imagination.py::lambda_returns`: backward recursion
- `train.py`: separated actor/critic optimizer steps and target EMA
- `world_model.py::freeze`: no world parameter updates
- `evaluate.py`: exact-world audit

## Important Components

Frozen world parameters isolate behavior learning. Differentiability still permits dynamics gradients. The critic extends effective horizon; the target critic stabilizes it; λ selects a bias/error tradeoff; entropy preserves exploration; exact evaluation detects exploitation.

## What happens if we remove it?

- Actor: online search remains necessary.
- Critic: returns truncate at imagination horizon.
- Target critic: bootstrapped targets chase online estimates more quickly.
- Reparameterization: ordinary samples do not provide a pathwise action gradient.
- Entropy: the Gaussian may collapse early around a spurious model optimum.
- Frozen world: actor updates could distort the model to report higher rewards instead of improving actions.
- Exact audit: optimistic imagined performance could be mistaken for real success.
- Continuation (already absent): imagination cannot represent episode termination, one reason this experiment is incomplete.

## What I Should Be Able to Explain

- How can gradients pass through a frozen world model?
- Why does `rsample` matter?
- What changes between λ=0 and λ=1?
- Why train a separate target critic?
- Why did imagined return improve while exact success remained false?
- How does actor learning amortize CEM?
- Which parts are Dreamer-like, and which Dreamer mechanisms are absent?

## Questions

- Should actor objectives be penalized by ensemble epistemic uncertainty?
- How should learned continuation change discounts after predicted termination?
- Would posterior RSSM start states prevent actor dependence on encoder point estimates?
- How much real replay refresh is needed to correct exploited regions?
