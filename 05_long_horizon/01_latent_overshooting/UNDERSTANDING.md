# Understanding Latent Overshooting and Compounding Error

## What problem does this solve?

Training inputs are often true states; rollout inputs are model predictions. This distribution mismatch lets tiny local errors accumulate into states the model never saw during one-step training.

## Before

```text
train: true s_t -> predict s_{t+1}
test: predicted s_hat_t -> predict s_hat_{t+1}
```

## After

Training also begins at many true states and recursively applies the model for up to five actions before comparing against the corresponding future truth.

## Core Idea

One-step accuracy is a local property. Rollout stability is a property of repeated composition. Overshooting differentiates through that composition so early errors are penalized for their downstream consequences.

## Data Flow

```text
s_t --a_t--> s_hat_{t+1} --a_{t+1}--> ... --a_{t+d-1}--> s_hat_{t+d}
 |                 |                                      |
true sequence s_{t+1} ...                              s_{t+d}
 |                 |                                      |
 +---------------- multi-distance losses ------------------+
```

## Mathematics

### One-step objective

```text
L_1=E_t ||f_theta(s_t,a_t)-s_{t+1}||^2.
```

Why needed: learns the local transition from well-supported true inputs.

### Recursive prediction

```text
s_hat_{t+1}=f_theta(s_t,a_t)
s_hat_{t+d}=f_theta(s_hat_{t+d-1},a_{t+d-1}).
```

Why needed: matches the information flow during imagination; after the seed there is no true state correction.

### Overshooting objective

```text
L_over=mean_{t,d=1...K} ||s_hat_{t+d}-s_{t+d}||^2
L=L_1+lambda L_over, K=5, lambda=0.5.
```

Why needed: teaches parameters that a small early error is bad when it causes larger later errors. Gradients pass through all earlier applications of `f_theta`.

### Compounding error

If a transition has local error `e_t` and local Jacobian `J_t`, a rough linearized propagation is

```text
e_{t+1} approximately J_t e_t + new model error.
```

When relevant Jacobian directions repeatedly amplify errors, long-horizon drift can grow even with tiny one-step MSE.

## Code Mapping

| Concept | Code |
|---|---|
| true nonlinear dynamics | `sequence_dataset.py::oscillator_transition` |
| one-step residual model | `latent_dynamics.py::LatentDynamics.forward` |
| evaluation rollout | `LatentDynamics.rollout` |
| recursive all-start losses | `overshooting_losses.py::latent_overshooting_loss` |
| combined loss | `long_horizon_loss` |
| compounding curve | `evaluate.py` |

## Important Components

- Multiple start times: teaches recovery throughout the state distribution, not one fixed initial condition.
- Predicted intermediate states: creates the actual exposure mismatch; replacing them with truth reduces to repeated one-step losses.
- Distance-wise logging: reveals which horizon begins to fail.
- Evaluation beyond training K: prevents claiming long-horizon success only inside the optimized window.

## What happens if we remove it?

- Remove overshooting: only local true-state transitions are optimized.
- Detach intermediate prediction: later losses cannot teach earlier transitions about downstream effects.
- Feed true intermediate states: no recursive rollout is trained.
- Evaluate only one step: the 30-step failure remains invisible.
- Set very large K immediately: optimization cost and unstable gradients may increase.

## What I Should Be Able to Explain

- Why is teacher forcing easier than open-loop rollout?
- How can MSE `1.6e-4` lead to horizon-30 MSE `1.4`?
- Which prediction is input at overshooting distance 3?
- Where do recursive gradients flow in code?
- Why is distance-1 overshooting equal to the one-step MSE?
- Why does K=5 not guarantee horizon 30?
- How is this implementation different from PlaNet's variational latent overshooting?
- What computation/memory cost grows with K?

## Questions

- How do scheduled sampling and overshooting differ?
- Can Jacobian/spectral regularization improve stability?
- Should distant errors receive more or less weight?
- Can temporal abstraction reduce the number of recursively composed transitions?
- How should probabilistic particles and overshooting be combined without exploding cost?
