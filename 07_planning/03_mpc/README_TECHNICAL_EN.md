# Receding-Horizon Model Predictive Control (MPC)

Status: completed on 2026-08-22. CEM-based exact-model MPC smoke test.

## Purpose

Execute only the first action of a planned sequence, observe the resulting state, and replan. This closes the loop between world-model planning and the physical/environment state.

## Problem

Random shooting and one-shot CEM produce open-loop sequences. Any disturbance or model error after the first step invalidates later actions.

## Previous Model

`02_cem` returns a complete action sequence but does not itself decide how much of it to execute before updating state.

## Hypothesis

Repeated CEM planning with updated state should reach the Goal while retaining a finite planning horizon.

## Architecture

```text
observe s_t -> CEM plan H=8 -> execute first action only
     ^                                  |
     +--------- environment s_{t+1} ----+
repeat until terminal or 20 steps
```

## Data Flow

An environment observation starts each planning call; one executed action produces the next observation and closes the loop.

## Tensor Shapes

Per plan: candidates `[256,8,2]`; execution result states `[executed+1,4]`, actions/rewards/plan scores `[executed,...]`.

## Mathematics

At time `t`, optimize `a_{t:t+H-1}`, execute `a_t*`, discard the remaining suffix, then solve again from observed `s_{t+1}`. This is receding-horizon control.

## Code Mapping

Loop/first-action execution: `mpc.py::RecedingHorizonMPC.run`; optimizer: `../02_cem/cem.py`; environment/model: `../planning_core.py`; metrics: `evaluate.py`.

## Training

No planner training. Each environment step performs four CEM refits. Exact dynamics isolate the replanning mechanism.

## Losses

There is no supervised loss; predicted return is optimized online.

## Evaluation Interface

`python 07_planning/03_mpc/evaluate.py` records success, executed steps/replanning calls, distances, reward, model/search configuration, and executed path.

## Smoke Test Results

Three tests passed. MPC reached the Goal in 12 executed actions/replanning calls; distance `2.2672 -> 0.00065`.

## Failure Cases

- Replanning multiplies inference cost and latency.
- Exact-model result does not measure robustness to learned-model bias.
- Short horizon can be myopic; terminal value quality becomes important.
- No action smoothness, collision, or real-time constraint.

## Findings

MPC establishes the Observation→Plan→Action→Environment feedback loop.

## Limitations

It can correct state after each executed action, but current success occurs under deterministic exact dynamics.

## Compare Later

Open-loop versus MPC under injected disturbances and learned-model errors; horizon/replan frequency/warm start. Metrics: success, reward, planning calls, latency, corrections, safety violations.

## Final Model Candidate

```text
Candidate: Yes for deployment-facing planning.
Reason: Feedback is essential when models/environments are imperfect.
Advantages: state feedback, finite horizon, planner-agnostic wrapper.
Disadvantages: repeated compute and latency, terminal-value dependence.
Possible conflicts: large video/ensemble models may not meet control deadlines.
```

## Next Questions

How does MPC behave with learned dynamics uncertainty, disturbances, and value error? When should a fast policy replace online planning?

## References

PETS: Kurtland Chua et al., 2018, https://arxiv.org/abs/1805.12114. TD-MPC2: Nicklas Hansen, Hao Su, Xiaolong Wang, 2023, https://arxiv.org/abs/2310.16828. Used for receding-horizon learned-model planning context. This implementation is an exact-model educational MPC loop, not either system reproduction.
