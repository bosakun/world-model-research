# Understanding MPC

## What problem does this solve?

It turns open-loop planning into feedback control by replanning after observing each executed transition.

## Before / After / Core Idea

Before: plan once and trust all H actions. After: optimize H actions, use only the first, observe, and optimize again. The discarded suffix still matters because it informed the first action.

## Data Flow / Mathematics

```text
a*_{t:t+H}=argmax J(s_t,a); execute a*_t; receive s_{t+1}; repeat.
```

## Code Mapping

MPC loop: `mpc.py`. Inner CEM: `../02_cem/cem.py`. True feedback state: `planning_core.py::PointWorldEnvironment`.

## Important Components

Fresh state observation corrects drift; first-action-only execution preserves feedback; terminal check stops replanning; horizon/terminal value balance short and long concerns.

## What happens if we remove it?

Execute full plan becomes open-loop control. Replan without observing actual state repeats the same error. No terminal stop wastes actions after success.

## What I Should Be Able to Explain

- Why optimize actions that will be discarded?
- How does MPC handle model error, and what can it not fix?
- Why is replanning costly?
- What does terminal value do for a short horizon?
- Why is exact-model success only a smoke result?

## Questions

How should warm starts, adaptive horizons, uncertainty penalties, and real-time budgets be integrated?
