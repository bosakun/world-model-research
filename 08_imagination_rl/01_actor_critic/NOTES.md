# Research Notes

## Before implementation

- Prediction: an actor can learn a directionally useful policy inside the small learned model, but its optimization will expose errors more strongly than CEM.
- Question: does “frozen world model” also stop actor gradients? No—parameters are frozen, but operations remain differentiable with respect to action/latent inputs.
- Important distinction: start observations are sampled, but no exact transition/reward is used in behavior losses.

## Implementation observations

- Actor update and critic update use separate imagined rollouts to avoid stale/reused autograd graphs after actor parameters change.
- The target critic is frozen as parameters, while input gradients are deliberately retained during the actor update.
- Critic targets are detached; otherwise critic optimization could modify both prediction and target paths.
- Evaluation re-encodes each exact observation. This is closed-loop state estimation, not open-loop latent rollout.

## Results

- Four unit tests passed.
- Imagined mean reward: `-0.6664 -> -0.1922`.
- Imagined mean λ-return: `-3.1524 -> +5.1975`.
- Exact Goal distance: `2.2672 -> 0.5128` (77.4% reduction), but success remained false.
- Exact total reward: `-16.18`.
- The actor climbed toward the upper boundary and settled away from the Goal; the critic became optimistic. This is model exploitation, not a successful controller.

## Errors and fixes

- No test/runtime error occurred in the first pass.
- Scientific failure discovered in evaluation: imagination metrics alone suggest strong improvement, while exact rollout rejects the policy. Kept as a result instead of tuning it away.

## Article material

- `outputs/actor_rollout.png`: especially useful because it overlays exact distance and critic value; their disagreement is the main story.
- `outputs/imagination_training.png`: shows that a smoother training curve does not guarantee reality alignment.
- Explanation hook: “A policy is an adversarial optimizer against its own world model.”
- Dreamer generations should be explained as complete algorithms with different latent/stability choices, not as labels for this minimal actor–critic loop.

## Compare later

- actor trained in exact differentiable dynamics vs learned dynamics;
- uncertainty-penalized imagination;
- continuation head and terminal-aware λ-return;
- actor initialization from CEM demonstrations;
- DreamerV3-style stabilizers and multiple seeds.
