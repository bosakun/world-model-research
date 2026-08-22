# Temporal Abstraction Research Notes

Date: 2026-08-22. Future article material.

## Before implementation

- Expected fewer recursive applications but worse one-macro-step accuracy.
- Important not to call a fixed open-loop chunk an “option.”
- Wanted primitive-equivalent horizon labels in every metric.

## Results and findings

- Five tests passed; seed 53; 10,178 parameters; 400 Adam steps.
- Teacher-forced five-step macro MSE `0.016751`.
- Six-step macro rollout (30 primitive steps) MSE `1.185631`.
- Errors at primitive horizons 5/10/15/20/25/30 are saved in JSON.
- The transition count fell from 30 to 6, but error remained large: abstraction changes where approximation difficulty appears.

## Implementation insight

- Reshaping actions must preserve exact time order: `[B,30,4] -> [B,6,5,4]`.
- Boundary state count is macro steps plus one.
- A GRU chunk encoder is recurrent inside each macro even though macro rollout is only six transitions.

## Article figures

- `outputs/macro_rollout.png`: boundary position/velocity at primitive time labels.
- `outputs/macro_error.png`: coarse error accumulation.
- Diagram comparing 30 small arrows with six large arrows, annotated with hidden intermediate states.

## Questions / later comparisons

- Matched primitive versus macro seed/data/training budget.
- Chunk size sweep and inference latency.
- Intermediate event, reward, and termination prediction.
- Learned variable duration and true options.
