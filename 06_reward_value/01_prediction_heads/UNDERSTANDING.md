# Understanding Reward, Value, and Continuation

## What problem does this solve?

A world model needs to answer not only “what next?” but “how good?”, “how much future return?”, and “is there a future after this transition?”

## Before

State/image predictors have no preference or episode boundary. Two equally accurate futures cannot be ranked.

## After

- reward `r_t`: immediate outcome of state/action;
- value `V(s_t)`: discounted future return under a specified policy/data process;
- continuation `c_t`: probability that returns/dynamics continue beyond the transition.

## Core Idea

These targets live at different temporal scopes. Reward is local, value summarizes a future, and continuation gates whether that future exists. They must not be treated as interchangeable scalar heads.

## Data Flow

```text
s_t -> shared features -> value
  + a_t -> transition features -> reward and continuation
targets: environment reward, terminal flag, backward discounted return
```

## Mathematics

```text
G_t=r_t+gamma c_t G_{t+1}.
```

- `gamma=0.95`: time preference/effective horizon.
- `c_t=0` on terminal transition: prevents value leaking past episode end.

Losses:

```text
L_r=MSE(r_hat_t,r_t)
L_V=MSE(V_hat(s_t),G_t)
L_c=BCEWithLogits(l_hat_t,c_t)
L=L_r+L_V+L_c,
```

all averaged only where `valid=1`.

Why logits for continuation: BCEWithLogits is numerically stable and the sigmoid probability remains available for rollout weighting/calibration.

## Code Mapping

| Concept | Code |
|---|---|
| terminal reward/continuation | `navigation_dataset.py::GoalNavigationSequenceDataset` |
| return target | `discounted_returns` |
| heads | `prediction_heads.py` |
| valid mask/losses | `prediction_losses.py` |
| Brier/terminal diagnostics | `evaluate.py` |

## Important Components

- State-action reward: the action causes the immediate transition outcome.
- State value: estimates future behavior from the current state before choosing the recorded action.
- Continuation: distinguishes terminal zero future from an ordinary low reward.
- Valid mask: padded rows are storage, not experience.
- Discount metadata: value numbers are meaningless without target discount/policy.

## What happens if we remove it?

- Remove reward: planner cannot score immediate outcomes.
- Remove value: finite planning must use very deep rollouts or truncate future utility to zero.
- Remove continuation: terminal states can bootstrap nonexistent future value.
- Remove valid mask: padding makes continuation/reward look artificially easy and corrupts value.
- Train value without stating behavior policy: invites incorrect “optimal value” interpretation.

## What I Should Be Able to Explain

- Why is reward not the same as value?
- Why does value depend on the policy/data distribution?
- Where does continuation enter the return equation?
- Why is the terminal transition itself valid training data?
- Why is continuation accuracy insufficient under imbalance?
- What does Brier score measure?
- Why does reward use action while this value head does not?
- Which MuZero/Dreamer components are absent?

## Questions

- When should value use TD/lambda targets and a slow target network?
- Should continuation predict environment terminal separately from time-limit truncation?
- How should macro models aggregate reward inside a chunk?
- Can joint gradients make representations value-equivalent but visually incomplete?
- How are uncertainty estimates propagated into risk-sensitive value?
