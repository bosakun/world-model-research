# Research Notes

## Before

- Safety checks must stay outside the neural network.
- Replay needs episode/step/source before adding online learning.

## Results

- Four tests passed: safety branches, robot transition, replay alignment, policy bounds/gradients.
- 2,083 demo transitions; action MSE `0.000137`; 64/64 simulator success; mean final distance `0.0525`.
- Runtime clip count zero because policy output is already bounded; unit tests retain envelope evidence.

## Article material

- `robot_rollout.png`; `replay_schema.json`.
- Explain requested action versus executed action.
- Simulator success versus authorization is an important distinction.

## Compare later

Episode splits, delay/noise, OOD starts, safety intervention rate, model-based planning, and demonstration/online mixtures.
