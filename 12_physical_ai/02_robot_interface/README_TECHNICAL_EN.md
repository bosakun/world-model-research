# Safe Robot Interface and Demonstration Replay

Status: completed on 2026-08-23. Simulator-only physical interface; no external hardware command was sent.

## Purpose

Define a bounded robot/action/replay boundary, record sequential demonstrations, train an offline imitation policy, and evaluate it through the same safety envelope.

## Problem

A world model cannot safely connect directly to actuators. Action bounds, enable/dead-man state, workspace checks, provenance, temporal alignment, and recoverable simulation tests belong outside learned policy parameters.

## Previous Model

Action-JEPA predicts physical representations but exposes no execution/replay contract. Earlier planners call synthetic dynamics directly.

## Hypothesis

A scripted demonstrator can populate a versioned transition schema; behavior cloning should reach Goals in the adapter simulator while safety logic remains independently testable.

## Architecture

```text
demonstrator/policy -> requested action -> SafetyEnvelope -> bounded action -> RobotInterface
                                           | deadman/workspace/clip
observation, bounded action, next observation, reward, done, source, episode, step -> replay
replay demonstrations -> imitation MLP -> policy -> same SafetyEnvelope
```

## Data Flow

The mobile robot observes `[x,y,vx,vy,goal_x,goal_y]`. A proportional scripted demonstrator generates noisy bounded commands. Records preserve episode/step ordering. The learned policy predicts actions offline; simulator evaluation filters every command before execution.

## Tensor Shapes

Observation/next `[N,6]`; action `[N,2]`; reward/done/episode/step `[N]`; policy batch `[B,6] -> [B,2]`; closed-loop trajectory `[T+1,6]`.

## Mathematics

Safety clipping is `a_safe=clip(a_requested,-v_max,v_max)` when enabled and inside workspace; otherwise zero action. Robot dynamics use `v'=0.5v+0.5a_safe`, `p'=clip(p+v')`. Behavior cloning minimizes `mean ||pi(o)-a_demo||²`, with `pi=0.2 tanh(MLP(o))` providing a second action bound.

## Code Mapping

Safety/robot: `robot.py`; demonstrator/replay alignment: `dataset.py`; bounded policy: `model.py`; offline training/schema: `train.py`; safety-wrapped evaluation: `evaluate.py`.

## Training

Seed 283; `safe-mobile-demonstrations-v1`; 256 episodes/2,083 transitions; 80/20 transition split; Adam `1e-3`; batch 64; 50 epochs/1,350 steps; 4,738 parameters; checkpoint format 1. `replay_schema.json` records field semantics and reserves source 1 for online data.

## Losses

Action MSE teaches imitation only. No environment reward or model-generated action enters policy gradients. This intentionally separates demonstration learning from DayDreamer-style online world-model RL.

## Evaluation Interface

`evaluate.py` runs 64 seeded simulator episodes, applies the external safety envelope at every step, and reports success, final distance, clipping, and whether hardware commands were sent.

## Smoke Test Results

Four tests passed. Validation action MSE `0.000137`; simulator success `64/64`; mean final distance `0.0525`; runtime clips `0`; external hardware commands `false`. Unit tests separately prove clip, dead-man stop, and workspace stop.

## Failure Cases

- Offline policy may fail outside demonstration start/Goal distribution.
- Policy bounding made runtime clipping inactive; this does not make the independent envelope unnecessary.
- Transition-level train/validation split shares episode distributions and is not a generalization benchmark.
- Simulator success does not authorize real hardware execution.

## Findings

Safety belongs at the actuator boundary and remains testable even when a policy appears bounded. Replay provenance/order are part of model correctness, not just logging.

## Limitations

No asynchronous I/O, latency, emergency hardware channel, authentication, calibration, real robot, MuJoCo, Wii Remote, image observations, or online world-model updates. External execution requires separate user approval and engineering review.

## Compare Later

Scripted vs learned policy, episode-level splits, action/noise delays, safety interventions, OOD starts, demonstrations plus online replay, success/reward/latency, and world-model planning.

## Final Model Candidate

```text
Candidate: Yes for the simulator/software boundary.
Reason: Explicit safety and replay contracts are prerequisites independent of model choice.
Advantages: bounded/recoverable execution; provenance; interchangeable adapter.
Disadvantages: simulator assumptions; no hardware timing; simple behavior cloning.
Possible conflicts: planners may request action sequences faster than hardware feedback permits.
```

## Next Questions

Which integrated-model action interface satisfies this contract? How should uncertainty trigger stop/replan rather than merely clip?

## References

### DayDreamer: World Models for Physical Robot Learning

- Authors: Philipp Wu, Alejandro Escontrela, Danijar Hafner, Ken Goldberg, Pieter Abbeel
- Year: 2022
- Paper: https://arxiv.org/abs/2206.14176
- Used for: physical robot world-model/replay motivation and the need to connect learned behavior with real interaction loops.
- Implementation: conceptual lineage for `robot.py`, `dataset.py`.

Classification: **Independent simulator interface and simplified offline imitation experiment**. DayDreamer's online Dreamer algorithm and robot results are not reproduced.
