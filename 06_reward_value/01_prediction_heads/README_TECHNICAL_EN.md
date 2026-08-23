# Reward, Value, and Continuation Prediction Heads

Status: completed on 2026-08-22. This is an independent Goal-navigation smoke task informed by Dreamer and MuZero; it is not either algorithm's reproduction.

## Purpose

Extend a world-state representation with the three signals needed to evaluate imagined behavior: immediate reward, long-term discounted value, and probability that the trajectory continues.

## Problem

Observation prediction says what may happen but not whether it is desirable, how much future return remains, or whether bootstrapping should stop. Planning/policy learning requires those semantics.

## Previous Model

Memory, uncertainty, and long-horizon phases predicted states/observations only. They had no learned task objective or terminal boundary.

## Hypothesis

Shared state features with specialized heads can learn sparse transition reward, Monte Carlo value target, and continuation probability. Each loss should train a distinct meaning and expose different failure metrics.

## Architecture

```text
state s_t [agent_x,agent_y,goal_x,goal_y] [4]
                    |
              state encoder [64]
               /              \
        value head          + action a_t [4]
          V(s_t)                   |
                          transition features [64]
                           /                  \
                    reward head        continuation head
                       r_t              logit c_t
```

## Data Flow

```text
goal-directed/noisy behavior -> padded navigation sequences
 -> reward after each action
 -> continuation=0 on terminal transition
 -> discounted Monte Carlo return backward through continuation
 -> masked joint supervised training
 -> planning-ready prediction interface
```

## Tensor Shapes

| Tensor | Shape | Meaning |
|---|---|---|
| states | `[B,T+1,4]=[B,21,4]` | normalized agent/Goal coordinates |
| actions | `[B,T,4]=[B,20,4]` | one-hot controls |
| rewards | `[B,T]` | `1` success, `-0.05` valid nonterminal, `0` padding |
| continuations | `[B,T]` | 1 nonterminal, 0 terminal/padding |
| valid mask | `[B,T]` | includes terminal transition, excludes padding |
| value targets | `[B,T]` | discounted behavior-policy returns |
| predicted reward/value/logit | `[B,T]` | specialized head outputs |

## Mathematics

```text
r_hat_t = f_r(s_t,a_t)
c_hat_t = sigmoid(f_c(s_t,a_t))
V_hat_t = f_V(s_t).
```

Monte Carlo targets are computed backward:

```text
G_t = r_t + gamma c_t G_{t+1}, gamma=0.95.
```

If `c_t=0`, future padded/next-episode return is not included. The masked objective is

```text
L = MSE_mask(r_hat,r) + MSE_mask(V_hat,G) + BCE_mask(logit_c,c).
```

## Code Mapping

| Concept | File / symbol |
|---|---|
| Goal navigation/terminal padding | `navigation_dataset.py::GoalNavigationSequenceDataset` |
| return recursion | `discounted_returns` |
| shared/features and heads | `prediction_heads.py::RewardValueContinuationHeads` |
| masked MSE/BCE | `prediction_losses.py::prediction_head_loss` |
| evaluation/calibration | `evaluate.py::evaluate` |

## Training

```bash
.venv/bin/python 06_reward_value/01_prediction_heads/train.py
.venv/bin/python 06_reward_value/01_prediction_heads/evaluate.py
.venv/bin/python -m pytest -q 06_reward_value/01_prediction_heads/tests
```

| Reproducibility item | Value |
|---|---|
| seed/dataset | 59 / `goal-navigation-v1` |
| data | 512 train / 128 validation, horizon 20 |
| discount | 0.95 |
| optimizer | Adam `1e-3`, batch 64 |
| epochs/steps | 80 / 640 |
| parameters | 9,091 |
| loss weights | reward/value/continuation all 1 |
| checkpoint/evaluation | format 1 gitignored / `python 06_reward_value/01_prediction_heads/evaluate.py` |

## Losses

- Reward MSE teaches immediate task consequence of `(state,action)`.
- Value MSE teaches expected discounted return under the data-generating behavior policy; it is not an optimal value.
- Continuation BCE teaches whether future return/model rollout remains valid after this transition.
- Valid masking prevents padded post-terminal rows from dominating all heads.

## Evaluation Interface

Evaluation reports reward/value RMSE, continuation accuracy and Brier score, terminal/nonterminal mean continuation probabilities, valid/terminal counts, discount, parameter count, and a sequence plot.

## Smoke Test Results

Five tests passed: sequence/terminal/padding contracts, continuation-aware return recursion, tensor/finite shapes, gradient flow through all heads/shared trunks, and padding-mask invariance.

| Metric | Result |
|---|---:|
| final train / validation total loss | 0.25178 / 0.24810 |
| reward RMSE | 0.25150 |
| value RMSE | 0.13537 |
| continuation accuracy / Brier | 0.97359 / 0.03194 |
| mean `P(continue)` on terminal | 0.18201 |
| mean `P(continue)` on nonterminal | 0.95016 |
| evaluation valid / terminal transitions | 568 / 128 |

## Failure Cases

- Continuation accuracy is inflated by more nonterminal than terminal transitions; terminal probability and Brier expose residual uncertainty.
- Terminal continuation probability `0.182` is not perfectly calibrated to zero.
- Value target describes the fixed noisy goal-directed behavior, not an optimal policy.
- Sparse success reward makes reward RMSE sensitive to rare terminal events.
- Heads receive true compact state, not predicted visual latent; representation errors are deferred.

## Findings

- The same representation can support immediate, long-term, and termination semantics with separate heads/losses.
- Continuation is essential in return recursion and masking, not merely another label.
- Accuracy alone is insufficient for a probabilistic continuation head.
- These outputs now define the contract needed for planning and imagination RL.

## Limitations

- Synthetic fully observed Grid navigation and supervised Monte Carlo returns.
- No TD/lambda return, target network, distributional value, reward transformation, or policy head.
- No learned dynamics in this folder and no imagined-state training.
- One seed; formal comparisons wait for Phase 90.

## Compare Later

- Separate versus shared trunks; state-only versus state-action reward/continuation.
- Monte Carlo versus TD/lambda value targets and target networks.
- Metrics: reward/value error, terminal precision/recall/Brier, return calibration, planning success, shared representation interference.
- Expected advantage: task-relevant compact evaluation of imagined futures.
- Expected weakness: target-policy dependence, sparse reward imbalance, bootstrapping bias later.
- Ablations: no continuation, no value, no reward, padding leakage, shuffled goals.

## Final Model Candidate

```text
Candidate:
Yes, subject to integration and calibration.

Reason:
World-model predictions cannot drive decisions without task value and termination semantics.

Advantages:
- small modular heads
- explicit immediate/long-term/terminal roles
- planning-ready interface

Disadvantages:
- value depends on behavior/policy and target construction
- head errors can mislead planning even with accurate dynamics
- class/reward imbalance requires careful metrics

Possible conflicts:
- MuZero-style value-equivalent representation may trade observation detail for task detail
- Dreamer actor/critic needs target networks and lambda returns
- macro transitions need cumulative reward and continuation across a chunk
```

## Next Questions

1. Can random shooting/CEM select actions using learned reward plus terminal value?
2. How do model errors bias value targets in imagined trajectories?
3. Should continuation model environment termination, truncation, or both?
4. When should reward/value features shape the latent representation itself?

## References

### Dream to Control: Learning Behaviors by Latent Imagination

Authors: Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi. Year: 2019. Paper: https://arxiv.org/abs/1912.01603.

Used for: learned reward/value predictions and continuation-aware latent imagination context. Corresponding code: `prediction_heads.py`, `navigation_dataset.py::discounted_returns`. No actor/critic imagination learning is implemented yet.

### Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)

Authors: Julian Schrittwieser et al. Year: 2019 (Nature 2020). Paper: https://arxiv.org/abs/1911.08265; DOI: https://doi.org/10.1038/s41586-020-03051-4.

Used for: planning-relevant learned reward and value quantities. This experiment does not implement MuZero representation/dynamics unroll, policy head, support transforms, MCTS, or self-play.

### Provenance statement

The navigation environment, behavior policy, Monte Carlo targets, shared architecture, and plots are an **independent educational implementation**.
