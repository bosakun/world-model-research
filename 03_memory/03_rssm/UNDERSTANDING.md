# Understanding the RSSM Experiment

## What problem does this solve?

A recurrent deterministic predictor can remember history, but it still maps one internal state to one next latent and does not say how an observation should correct a prediction. An RSSM gives these two operations different distributions:

- the **prior** predicts state before seeing the current image;
- the **posterior** infers state after seeing the image.

This distinction lets the model learn from observed sequences and then imagine without future observations.

## Before

The GRU experiment had an encoded latent `z_t`, a hidden state `h_t`, and a deterministic prediction head. `z_t` was simply the encoder output. There was no learned distribution over `z_t`, no uncertainty parameters, and no KL term connecting observation-conditioned training to open-loop rollout.

## After

The RSSM has two internal state parts:

- `h_t in R^64`: deterministic recurrent memory, updated from the previous sampled state and action;
- `z_t in R^16`: stochastic state sampled from a diagonal Gaussian.

At each observed step, the posterior corrects the prior using image evidence. During future imagination, only the prior is available.

## Core Idea

`h_t` and `z_t` are complementary, not two names for the same latent:

- `h_t` is a deterministic summary of the model's previous states/actions. Repeating the same inputs yields the same `h_t`.
- `z_t` represents the current uncertain latent realization. Its distribution can express residual information that is not fixed by the recurrent path.

An observation changes `q(z_t|h_t,o_t)` but does not directly change `p(z_t|h_t)`. KL divergence trains the prior to predict the posterior, so the prior remains useful when the observation disappears.

## Data Flow

### Learning with observations: filtering

```text
previous (h_{t-1}, z_{t-1}) and a_{t-1}
              |
              v
h_t = GRU(h_{t-1}, [z_{t-1},a_{t-1}])
       |                         |
       v                         v
p(z_t|h_t)             o_t -> Encoder -> e_t
       |                         |
       +-------------+-----------+
                     v
              q(z_t|h_t,e_t)
                     |
       z_t = mu_q + sigma_q * epsilon
                     |
                 [h_t,z_t]
                     |
             reconstruct o_t
```

### Future rollout without observations: imagination

```text
posterior seed (h_t,z_t)
  -> action a_t
  -> h_{t+1}
  -> prior p(z_{t+1}|h_{t+1})
  -> sample/mean z_{t+1}
  -> decode prediction
  -> repeat
```

No encoder or posterior may consume `o_{t+1}` in this second path. Otherwise evaluation leaks the answer.

## Tensor Shapes

For the smoke configuration:

```text
observations:          [B, T+1, 3, 20, 20]
actions:               [B, T, 4]
observation embeddings:[B, T+1, 64]
h sequence:            [B, T+1, 64]
z sequence:            [B, T+1, 16]
prior mean/std:         [B, T+1, 16]
posterior mean/std:     [B, T+1, 16]
reconstructions:        [B, T+1, 3, 20, 20]
imagined states/images: [B, T, 16] / [B, T, 3, 20, 20]
```

There are `T+1` observations because `T` actions connect adjacent observations.

## Mathematics

### Deterministic recurrent transition

```text
h_t = f_GRU(h_{t-1}, z_{t-1}, a_{t-1})
```

- `h_{t-1}`: previous deterministic memory.
- `z_{t-1}`: previous stochastic realization.
- `a_{t-1}`: action that caused the next transition.
- `h_t`: updated history summary before seeing `o_t` in the posterior.

Why needed: it transports temporal information across frames and gives the prior a history-dependent condition.

### Prior

```text
p(z_t | h_t) = N(mu^p_t, diag((sigma^p_t)^2))
```

- `mu^p_t`, `sigma^p_t`: predicted mean and standard deviation from `h_t`.
- `z_t`: stochastic state that the model thinks may occur before observing the frame.

Why needed: future imagination cannot use an unknown future image. It requires a predictive distribution based only on the internal state/action history.

### Posterior

```text
q(z_t | h_t,o_t) = N(mu^q_t, diag((sigma^q_t)^2))
```

The implementation uses `e_t=Encoder(o_t)` and conditions the Gaussian head on `[h_t,e_t]`.

Why needed: the actual observation contains information that the prior may not know. The posterior performs a learned correction/inference step and supplies a trainable latent explanation of the image.

### Reparameterization

```text
epsilon ~ N(0,I)
z_t = mu^q_t + sigma^q_t odot epsilon
```

- `epsilon`: parameter-free random noise.
- `odot`: elementwise multiplication.

Why needed: direct sampling appears nondifferentiable with respect to distribution parameters. Moving randomness into `epsilon` lets gradients flow through `mu` and `sigma`. The tests explicitly verify gradients to both.

### Reconstruction

```text
o_hat_t = g_theta([h_t,z_t])
L_image = mean(w_channel odot (o_hat_t-o_t)^2)
```

Why needed: it forces the latent state to retain observation information. The green-channel weight is 20 because the Goal is a tiny fraction of pixels; this is a task-specific modification.

### KL divergence

For each diagonal component:

```text
KL(q||p) = log(sigma_p/sigma_q)
         + (sigma_q^2 + (mu_q-mu_p)^2)/(2 sigma_p^2)
         - 1/2
```

- `q`: observation-informed posterior.
- `p`: predictive prior.

Why needed: reconstruction alone can teach only the posterior path. KL asks the prior to place probability mass where the posterior inferred the actual state, making observation-free rollout meaningful.

The implementation uses `max(KL, 1 nat)` per sequence state before averaging. This free-nats threshold reduces pressure to erase useful information once KL is already small. It does not make KL disappear from metrics: raw KL is logged separately.

### Complete smoke objective

```text
L_total = L_image + 0.1 L_goal + 0.001 L_KL_free_nats
```

`L_goal` is ten-class cross-entropy from `[h_t,z_t]`: nine local Goal cells and one not-visible class. It probes semantic state and combats image imbalance. It is not part of the PlaNet RSSM definition.

## Code Mapping

| Concept | Code |
|---|---|
| `h_t` and GRU transition | `rssm.py::RecurrentStateSpaceModel.transition` |
| prior distribution | `rssm.py::prior`, `GaussianHead.forward` |
| posterior distribution | `rssm.py::infer_posterior` |
| reparameterized sample | `rssm.py::DiagonalGaussian.sample` |
| observed sequence filtering | `rssm.py::observe` |
| observation-free rollout | `rssm.py::imagine` |
| image model | `rssm.py::ObservationEncoder`, `ObservationDecoder` |
| semantic probe/head | `rssm.py::goal_head`, `rssm_losses.py::goal_class_targets` |
| analytic KL | `rssm_losses.py::diagonal_gaussian_kl` |
| combined objective | `rssm_losses.py::rssm_loss` |
| sequence data contract | `rssm_dataset.py`, reused `02_partial_observation/partial_dataset.py` |
| metrics and visualization | `evaluate.py::evaluate` |

## Important Components

### Deterministic state `h_t`

Why necessary: it retains ordered history without requiring every stochastic state to encode the entire past. It is the recurrent backbone of the predictive prior.

### Stochastic state `z_t`

Why necessary: it provides a distribution-valued current state and can represent residual uncertainty or alternative realizations not encoded deterministically in `h_t`. On this deterministic dataset, architectural presence is not evidence of useful calibrated uncertainty.

### Prior and posterior as separate heads

Why necessary: “what I predict before seeing the frame” and “what I infer after seeing it” answer different questions. Using one head would blur the learning/imagination boundary.

### Positive standard deviation

Why necessary: Gaussian standard deviations must be positive. `softplus(raw_std)+0.1` provides a differentiable lower bound and avoids division by zero in KL.

### Sequence processing

Why necessary: `h_t` depends recursively on earlier `z` and actions. Random independent transitions cannot reproduce that computation and cannot teach the model what to carry through time.

### Decoder

Why necessary: without an observation/reward/value prediction target, the latent state could satisfy prior/posterior alignment while containing no task-relevant visual information.

## What happens if we remove it?

- Remove `h_t`: the prior loses recurrent history; `z_t` becomes closer to a frame-local variational latent.
- Remove `z_t`: the model becomes deterministic recurrent dynamics and cannot express a latent distribution.
- Remove the prior: observed reconstruction can work, but no principled observation-free future state exists.
- Remove the posterior: the latent cannot be corrected using current evidence; training becomes prior-only prediction.
- Remove KL: posterior reconstruction may remain good while prior rollout visits latents the decoder never learned to interpret.
- Remove reparameterization: stochastic samples stop carrying reconstruction gradients to Gaussian parameters.
- Remove the decoder/task heads: the model has no reason to encode observation semantics.
- Reset `h_t` each step: history is discarded and partial-observation aliases become indistinguishable except through current evidence.
- Feed future images during rollout: reported “prediction” becomes posterior filtering and leaks ground truth.
- Use only plain pixel MSE here: the one-cell Goal contributes little, so the model may learn background while failing semantically.

## What I Should Be Able to Explain

- Why is `h_t` deterministic while `z_t` is stochastic?
- Why is the encoder output `e_t` not the same object as `z_t`?
- What does the prior know, and what extra information does the posterior know?
- Why does training use the posterior while imagination must use the prior?
- Where exactly does action `a_{t-1}` enter the equations and code?
- How does reparameterization allow gradients to reach `mu` and `sigma`?
- What distribution is on each side of `KL(q||p)`, and why this direction?
- Why can reconstruction be good while prior rollout is bad?
- What does free nats change, and what does raw KL still tell us?
- Why does 100% Goal **state-head** accuracy not imply a perfect decoded image?
- Which parts follow PlaNet-family ideas and which are repository-specific modifications?
- Why does stochastic architecture alone not prove learned epistemic or aleatoric uncertainty?

## Questions

- Does this dataset contain enough ambiguity for the Gaussian state to learn meaningful nonzero diversity?
- Would a categorical latent like DreamerV2 avoid or merely change the first-frame averaging artifact?
- How do KL balancing and stop-gradient variants affect prior/posterior agreement?
- Does latent overshooting improve longer rollouts more than one-step KL?
- Can an image likelihood or perceptual/object loss replace the task-specific Goal head?
- When causal Transformer memory is introduced, should it replace `h_t` or provide context to both prior and posterior?
