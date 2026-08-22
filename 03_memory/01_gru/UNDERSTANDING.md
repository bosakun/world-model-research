# Understanding GRU Memory in a Latent World Model

## What problem does this solve?

A feed-forward dynamics function sees only the variables provided now. If the observation is incomplete, the same `z_t` can arise from different real states and require different predictions. GRU adds a state `h_t` whose value depends on earlier `(z,a)` pairs, so prediction can depend on history without storing the entire sequence explicitly.

This experiment validates the mechanism, not its necessity: the current Grid World image shows the complete state, so memory should be redundant in principle.

## Before

```text
predicted z_{t+1} = f(z_t, a_t)
```

Every transition is conditionally independent of earlier inputs once `z_t,a_t` are known. This is correct only when `z_t` captures a sufficient Markov state.

## After

```text
h_{t+1} = GRUCell([z_t; a_t], h_t)
predicted z_{t+1} = g(h_{t+1})
```

Earlier inputs can affect future predictions through the recurrent path `h_t -> h_{t+1}`.

## Core Idea

### GRU and its relation to an RNN

A vanilla RNN repeatedly updates one hidden vector with a nonlinear transformation. GRU is an RNN cell with gates that learn when to retain old hidden content and when to replace it with a new candidate. Gates make persistent information paths easier to learn than repeatedly overwriting the entire vector.

### What is hidden state?

`h_t` is a learned, fixed-width internal summary produced by previous recurrent updates. It is not an explicit log and no coordinate has a predefined meaning. Saying a GRU “remembers” means that information in earlier inputs changes `h_t`, survives later gated updates, and is usable by the prediction loss to reduce future error.

### `z_t` versus `h_t`

- `z_t`: Encoder output for the current image only; shape `[B,16]`.
- `h_t`: history-dependent dynamics state before processing step `t`; shape `[B,64]`.

Two trajectories can have identical `z_t` but different `h_t`. That distinction becomes useful only when history contains predictive information absent from the current observation.

## Data Flow

At training step `t`:

1. Encode all ground-truth images to `z_0...z_T`.
2. Concatenate `z_t` and one-hot `a_t` to `[B,20]`.
3. Give this and `[B,64]` `h_t` to one shared `GRUCell`.
4. Obtain updated `h_{t+1}` immediately after processing `(z_t,a_t)`.
5. Map `h_{t+1}` to predicted `z_{t+1}`.
6. Compare with detached encoded ground truth.
7. Carry `h_{t+1}` into the next sequence step.

During rollout, only `z_0` is encoded from truth. Predicted `z_{t+1}` becomes the next latent input, and the updated hidden is carried alongside it. Reset hidden to zeros at an episode boundary; carrying it between unrelated episodes would leak context.

## Mathematics

### Input construction

```text
x_t = [z_t; a_t]
```

- `z_t in R^16`: current encoded observation.
- `a_t in {0,1}^4`: one-hot action.
- `[;]`: concatenation, so `x_t in R^20`.

Why: the transition must depend on both perceived state and intervention.

### Reset gate

```text
r_t = sigmoid(W_ir x_t + b_ir + W_hr h_t + b_hr)
```

- `r_t in (0,1)^64`: element-wise reset gate.
- `W_*`, `b_*`: learned affine parameters.

Why: controls how strongly old hidden content contributes while constructing new candidate content.

### Update gate

```text
u_t = sigmoid(W_iu x_t + b_iu + W_hu h_t + b_hu)
```

- `u_t in (0,1)^64`: element-wise update/retention gate.

Why: learns different persistence time scales. With this convention, values near one preserve the old hidden coordinate.

### Candidate hidden state

```text
n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_t + b_hn))
```

- `n_t in (-1,1)^64`: proposed new content.
- `*`: element-wise multiplication.

Why: forms new state content from the current latent/action and the reset-filtered past. This equation is PyTorch's `GRUCell` convention; exact bias/reset placement differs among GRU descriptions.

### Hidden update

```text
h_{t+1} = (1-u_t) * n_t + u_t * h_t
```

Why: creates a gated additive path between the old state and candidate, letting optimization learn retention instead of overwriting every coordinate at every step.

### Latent prediction

```text
predicted z_{t+1} = g_theta(h_{t+1})
```

- `g_theta`: two-layer MLP prediction head.

Why: hidden size and latent size serve different roles; the head translates the memory state into the representation space compared by the dynamics loss.

### Reconstruction loss

```text
L_rec = mean((Decoder(Encoder(o_t)) - o_t)^2)
```

Why: forces `z_t` to retain information about the full current observation. Without it, deterministic latent prediction alone has a trivial constant-latent collapse. Pixel MSE alone still found a near-background shortcut because the agent is small, so it is paired with the position loss below.

### Agent position loss

```text
L_pos = cross_entropy(cell_logits(Decoder(z_t)), agent_cell_t)
```

`cell_logits` averages red-minus-other-channel evidence within each of 25 cells and scales it before softmax. Why: the mutually exclusive classification loss penalizes both a missing agent and red predictions in the wrong cells. This uses known Grid World rendering semantics; it is an educational auxiliary objective, not an unsupervised paper-derived loss.

### Dynamics loss

```text
L_dyn = mean((predicted z_{t+1} - stopgrad(Encoder(o_{t+1})))^2)
```

Why: trains the recurrent transition to predict the next learned representation. `stopgrad` makes the target fixed for this loss's backward pass; the Encoder still learns through reconstruction.

### Total loss

```text
L = L_rec + 0.2 L_pos + 2 L_dyn
```

Why: balances perceptual grounding and transition fitting in this smoke-scale experiment. The coefficient is experimental, not paper-derived.

## Code Mapping

| Understanding target | Location |
|---|---|
| full observation and deterministic action | `env.py` |
| `[B,T+1,3,20,20]` trajectories and one-hot actions | `dataset.py` |
| current visual state `z_t` | `model.py::VisualEncoder` |
| reconstruction | `model.py::VisualDecoder` |
| GRU inputs/outputs and update timing | `model.py::GRUDynamics.step` |
| teacher-forced sequence handling | `model.py::GRUDynamics.forward` |
| hidden state during imagined rollout | `model.py::GRUDynamics.rollout` |
| prediction head | `model.py::GRUDynamics.prediction_head` |
| all losses and detached target | `losses.py` |
| no-memory counterfactual | `baseline.py::SimpleDynamics` |

## Important Components

### One-hot action

It identifies the intervention without imposing a false numeric order such as `right > left`. Removing action makes opposite transitions indistinguishable at the same state.

### GRUCell rather than GRU

`GRUCell` exposes one transition and makes the timing of `z_t`, `a_t`, `h_t`, and `h_{t+1}` explicit. A sequence-level `nn.GRU` could compute equivalent recurrence more compactly, but would hide the rollout state handling this experiment is meant to teach.

### Hidden reset

Zeros define “no prior episode history.” Without reset between independent episodes, predictions may depend on unrelated previous trajectories and evaluation leaks context.

### Prediction head

The recurrent state is 64-dimensional while the visual latent is 16-dimensional. The head supplies the learned mapping between those spaces. Removing it would require equal dimensions and conflate memory capacity with representation size.

### Teacher forcing during training

Ground-truth encoded `z_t` stabilizes one-step learning. Removing teacher forcing and training only autoregressively increases difficulty but can address exposure bias; this tradeoff belongs in a later long-horizon experiment.

### Autoregressive evaluation

It reveals compounding error hidden by teacher-forced one-step metrics. Replacing predicted latents with ground truth during evaluation would not test imagination/rollout.

## What happens if we remove it?

| Removed component | Expected consequence |
|---|---|
| hidden state / replace GRU with MLP | history cannot affect prediction; full-observation task may be unchanged, partial observation should expose failure |
| update gate | state is overwritten more aggressively, making long retention harder |
| reset gate | candidate cannot selectively suppress irrelevant past content |
| action | next position becomes ambiguous whenever multiple actions are possible |
| reconstruction loss | latent collapse becomes a valid low-dynamics-loss solution |
| dynamics loss | encoder/decoder may reconstruct but cannot predict futures |
| hidden reset at episode boundaries | cross-episode leakage and unstable semantics |
| hidden carry during rollout | recurrence degenerates into repeatedly restarting a one-step model |
| target detach | encoder and dynamics can cooperate toward easier but less grounded latent targets; collapse pressure increases |

## Is memory actually needed in the current Grid World?

Probably not. The image reveals agent and goal positions, action effects are deterministic, and therefore the next state is a function of `(o_t,a_t)`. The GRU may learn a valid transition while ignoring history. Good loss here proves implementation viability, not useful memory.

The next experiment must deliberately create perceptual aliasing—for example, crop the view so orientation/position information disappears or show observations that require remembering an earlier cue. Then identical present observations can require different predictions, and `h_t` becomes causally testable.

## What I Should Be Able to Explain

- Can I distinguish `z_t` from `h_t` in source, meaning, and shape?
- Why are there `T+1` observations but only `T` actions?
- Exactly when does `h_t` become `h_{t+1}`?
- What are the inputs and outputs of `GRUCell`?
- What does it operationally mean for the GRU to remember something?
- Why must hidden state be carried during rollout and reset between episodes?
- Why is one-step teacher forcing insufficient evidence of stable rollout?
- Which gate preserves old hidden content in the implemented equation?
- Why does reconstruction loss help prevent latent collapse?
- Why is a fully observable Grid World a weak test of memory?
- Why is this implementation not PlaNet, RSSM, or Dreamer?
- What experiment would show that history, rather than parameter count, caused improvement?

## Questions

- Does `h_t` retain state when observations are masked for several steps?
- How should hidden-state utilization be measured beyond prediction loss (linear probes, gate statistics, history shuffling)?
- Would scheduled sampling or multi-step loss reduce rollout drift?
- Should later RSSM observation decoding condition on deterministic `h_t`, stochastic `z_t`, or both?
- How sensitive are results to latent size, hidden size, sequence length, and seed?
- Can a matched MLP with similar parameter count separate recurrence benefit from capacity benefit?
