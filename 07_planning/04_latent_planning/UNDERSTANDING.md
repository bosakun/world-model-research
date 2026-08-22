# Understanding Decoder-Free Latent Planning

## What problem does this solve?

It learns an internal state that answers a control question—“what reward/value follows these actions?”—without first requiring every observation detail to be reconstructed.

## Before

The planner had perfect hand-written dynamics. The learned prediction heads did not own a latent transition, so the full observation-to-plan path was missing.

## After

An observation is encoded once. Candidate actions produce imagined latent futures, reward and terminal-value predictions score them, and CEM selects actions. Decoding is unnecessary for the algorithm, although exact states are plotted afterward to audit the plan.

## Core Idea

A useful world state need not be a photographic copy. It must retain distinctions that change future reward, value, and action consequences. Joint temporal consistency and task supervision decide which distinctions matter in this experiment.

## Data Flow

```text
training: o_t -> encoder -> z_t --a_t--> z_hat_(t+1)
              o_(t+1) -> encoder -> target z_(t+1)
              z_hat_(t+1) -> reward/value heads -> supervised targets

planning: o_0 -> z_0 -> many candidate action rollouts -> predicted returns -> CEM elites
```

## Mathematics

### Representation

`z_t=e_theta(o_t)`.

- `o_t`: available observation (four Point World coordinates here)
- `e_theta`: learned encoder
- `z_t`: 16-dimensional task-oriented latent
- Why: planner compute is performed in a compact learned state instead of the observation domain.

### Latent dynamics

`z_hat_(t+1)=tanh(z_hat_t+f_theta(z_hat_t,a_t))`.

- `a_t`: two-dimensional continuous action
- `f_theta`: learned residual change
- Why: actions must have predictable consequences across multiple imagined steps.

### Consistency

`L_cons=||z_hat_(t+1)-sg(e_theta(o_(t+1)))||^2`.

- `sg`: stop-gradient on the target branch
- Why: the transition should land near the representation of what actually followed. Stop-gradient prevents both sides from moving toward one another within the same term, but does not alone prevent collapse.

### Task heads

`L_r=(r_hat(z_hat_(t+1))-r_(t+1))^2` and `L_v=(v_hat(z_hat_t)-v_t)^2`.

- reward predicts immediate control outcome;
- value predicts negative Goal distance as a terminal heuristic;
- Why: they make collapsed or task-irrelevant representations costly and supply the planner objective.

### Planning objective

`J=sum gamma^t r_hat(z_hat_(t+1))+gamma^H v_hat(z_hat_H)`.

- Why: a finite horizon needs both within-horizon reward and an estimate beyond its boundary.

## Code Mapping

- `dataset.py`: true sequence construction and targets
- `model.py::encode`: `e_theta`
- `model.py::transition`: `f_theta` and recursive imagination
- `model.py::reward/value`: task predictions
- `losses.py`: the three equations and their weights
- `planner.py`: CEM over the learned `J`
- `evaluate.py`: transfer of chosen actions back to exact dynamics

## Important Components

The encoder chooses a representation; dynamics makes it actionable; consistency ties imagination to observations; reward makes immediate consequences visible; value handles the planning boundary; recursive training exposes the model to its own latent errors; CEM optimizes actions without differentiating through action variables.

“Decoder-free” means the training/planning path has no reconstruction model. It does not mean the environment state is unavailable during dataset generation or evaluation.

## What happens if we remove it?

- Encoder: there is no learned observation-to-planning state.
- Dynamics: candidates cannot be distinguished by future consequences.
- Consistency: reward/value supervision may fit shortcuts with poor temporal geometry.
- Reward head: the planner ignores immediate outcomes and sparse success bonus.
- Value head: horizon truncation favors locally good but globally weak sequences.
- Task heads: consistency admits a constant representation.
- Stop-gradient: online/target representations can chase one another and destabilize the target.
- Recursive unroll: training sees only one-step teacher states, but planning sees self-generated states.

## What I Should Be Able to Explain

- Why can a planning latent omit details required for image reconstruction?
- What prevents consistency loss alone from learning a useful state?
- Why is `z_hat_(t+1)` used recursively during training?
- Where can model exploitation enter CEM?
- Why are learned score and exact-world score different?
- What does the terminal value add to a finite-horizon return?
- In what ways is this not MuZero or TD-MPC2?

## Questions

- Would a target encoder updated by EMA improve stability?
- Does a decoder regularize dynamics or waste capacity on irrelevant detail here?
- How should ensemble uncertainty penalize exploitable candidate sequences?
- Can imagined actor learning amortize the repeated CEM search?
