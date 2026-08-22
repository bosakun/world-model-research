# Research Notes

## Before implementation

- Question: can the exact transition used by the first planners be removed without adding a decoder?
- Prediction: reward/value supervision will make a small latent sufficient for short Point World plans, but learned-model CEM will underperform exact-model CEM.
- Easy misconception: “no decoder” does not mean “no model”; the latent transition and task heads are still a world model for control-relevant quantities.

## Implementation observations

- Mixed random and noisy goal-directed sequences were used. Purely greedy data would narrow action coverage; purely random data would rarely expose the success region.
- Reward/value heads are evaluated on recursively predicted latents rather than teacher-encoded latents. This makes the training path closer to planning, where future observations are unavailable.
- A residual latent transition was easier to interpret than predicting an unrelated latent from scratch.
- PyTorch warned when logging a tensor that still required gradients. Logging now explicitly detaches it; the optimization behavior was unchanged.

## Results and surprises

- Validation total/consistency/reward/value: `0.2961 / 0.02193 / 0.12991 / 0.01430`.
- Exact-world Goal distance after the learned plan: `2.2672 -> 0.9858` (`56.5%` reduction).
- Learned score `-11.470` was more optimistic than exact score `-12.498`. That numerical gap is a compact illustration of model bias and planner exploitation risk.
- Exact distance was smallest around step 8, then rose through step 10. The final point still improves strongly over the start, but the tail is a visible open-loop/model-bias failure case.
- Reward remains much harder than the smooth distance-value target because it includes a discontinuous success bonus.
- Exact-model CEM previously reached a much smaller final distance. This agrees with the pre-implementation prediction, but no matched statistical comparison is claimed yet.

## Errors and fixes

- Initial training logging used `float(loss_tensor)` and emitted a requires-grad warning. Changed to `float(loss_tensor.detach())`.
- No functional test failure occurred in the first implementation; four focused tests passed.

## Article material

- `outputs/latent_plan.png`: separates latent planning from exact-world audit—the plot is not used by the planner.
- `outputs/loss_curve.png`: reward is the dominant error component.
- Useful explanation: a world model can be “correct enough for decisions” without rendering the world, but its optimism must be checked against reality.
- Useful contrast: MuZero uses learned-model MCTS and policy/value targets; TD-MPC2 contains substantially more stabilization and scalable control machinery; this folder isolates only their decoder-free task-oriented latent-planning intuition.

## Compare later

- exact CEM vs learned latent CEM with identical candidates/iterations;
- decoder vs no decoder;
- one-step teacher forcing vs recursive consistency;
- remove consistency, reward, or value head;
- open loop vs replanning;
- in-distribution vs OOD starts and uncertainty penalty.
