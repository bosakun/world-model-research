# RSSM Research Notes

Date: 2026-08-22. These are source notes for a future article, not the article itself.

## Questions before implementation

- Is the stochastic state an additional memory, or a current uncertain state conditioned on deterministic memory?
- If training sees every observation through the posterior, what forces the prior to work during rollout?
- Will a diagonal Gaussian learn meaningful uncertainty in a deterministic two-case Grid World?
- Can pixel MSE detect failure to remember a one-cell Goal?

## Predictions before implementation

- Posterior reconstruction should become easy on the small image.
- Prior rollout should be harder because future observations are unavailable.
- KL should be indispensable for closing this train/rollout gap.
- Aggregate image MSE may look good even if the Goal is wrong.

## Initial misunderstandings corrected

- `h_t` and `z_t` are not interchangeable latent vectors. `h_t` is deterministically updated history; `z_t` is a distribution-valued current state.
- The prior is not a regularizer-only network. It is the network used to create future states during imagination.
- “Stochastic RSSM” does not automatically mean calibrated uncertainty. The data, loss, and evaluation must reveal whether the standard deviation has useful meaning.
- A low reconstruction MSE on sparse images does not imply the small semantic object was reconstructed.

## Implementation observations

- Keeping `observe` and `imagine` as separate methods made future-observation leakage easy to test.
- At `t=0`, deterministic state is zero. The prior is therefore observation-independent, while the posterior changes with `o_0`; a test checks exactly this distinction.
- Standard deviation is produced by `softplus + min_std`, which made KL finite from the first update.
- Mean rollout (`stochastic=False`) made the smoke evaluation reproducible. Sample-based calibration remains future work.

## Errors, causes, and fixes

### Attempt 1: image MSE plus KL only

Symptom: total and pixel loss fell, but the initial reconstruction averaged the right-Goal and down-Goal cases. The model could obtain a good score by explaining the static background.

Cause: the Goal occupies very few pixels, so its semantic importance was tiny in uniform RGB MSE.

Fix: add a semantic Goal target and inspect reconstruction images, not only loss.

### Attempt 2: classify Goal from decoded RGB evidence

Symptom: the nominal Goal accuracy became 100%, but the decoder painted red artifacts across visible cells. This reduced the green-vs-red evidence used by the `not visible` threshold and gamed the metric/loss.

Cause: target extraction and trainable prediction used the same hand-designed RGB rule, creating a direct image-level shortcut.

Fix: move classification to an independent head on `[h_t,z_t]`; keep target extraction from the ground-truth observation only.

### Attempt 3: separate state head, green weighting 5

Symptom: red artifacts disappeared and the semantic state was correct, but the first posterior image still visibly averaged two Goal candidates.

Cause: the decoder still had weak pressure on green pixels relative to all background pixels.

Fix: use symmetric channel weights `[1,20,1]` in reconstruction. This improved the primary Goal image without coupling the state-head class to decoder tricks.

### Remaining failure

The final `reconstruction.png` still contains a faint secondary green Goal at the ambiguous initial frame. The correct right Goal is stronger, and the separate state head is correct, but the image is not exact. This should be an article example of why pixel averages and semantic state metrics must be reported separately.

## Results worth preserving

- 21 combined tests passed after the final implementation.
- 428,330 trainable parameters; seed 23; `partial-observation-v1`; 160 Adam steps at learning rate `3e-3`.
- Train total: `1.036060 -> 0.016743`.
- Final validation total: `0.010783`; weighted reconstruction: `0.008861`; raw KL: `1.756577` nats.
- Posterior plain pixel MSE: `0.000746`.
- One-step prior plain pixel MSE: `0.000618`.
- Mean six-step prior rollout pixel MSE: `0.000628`.
- Goal state-head accuracy: 100% for posterior, one-step prior, and the six tested rollout horizons.

These are smoke results from one seed and a tiny deterministic dataset, not comparison evidence.

## Interesting behavior

- Horizon-2 pixel error was lower than horizon-1. Error is not guaranteed to grow monotonically when later target frames become visually simpler and the background dominates.
- Prior rollout preserved the hidden Goal class after it disappeared from the image, suggesting that the model state contains the needed synthetic-task information.
- A semantically wrong image can have a very small global MSE; a semantically correct state head can coexist with a visibly imperfect decoder.

## Figures for a future article

- `outputs/reconstruction.png`: posterior correction and the faint initial averaging artifact.
- `outputs/latent_rollout.png`: top row true observations, bottom row prior-only imagination after one posterior seed.
- `outputs/loss_curve.png`: total objective and the different scale/behavior of reconstruction and raw KL.
- `outputs/rollout_error.png`: non-monotonic per-horizon image MSE.
- Potential future figure: one panel comparing Attempt 1 averaging, Attempt 2 red shortcut, and final independent-state-head result. Earlier overwritten images would need to be deliberately regenerated from saved configurations.

## Explanations that may work well in an article

- “Posterior is the answer key available during learning; prior is the closed-book prediction used in imagination. KL transfers knowledge from the former to the latter.”
- “`h_t` is the deterministic notebook; `z_t` is the uncertain current hypothesis.” This analogy needs a warning that both are learned vectors, not literal symbolic memory.
- Show why `T` actions require `T+1` observations.
- Explain decoder shortcut behavior as evidence that metrics are part of the model's incentive structure.

## Questions raised during the experiment

- Is the prior's learned standard deviation useful, or does the small deterministic task reward only its mean?
- Would learned observation variance or a categorical image likelihood reduce averaging?
- Should the semantic probe be trained jointly, or kept frozen/evaluation-only in controlled comparisons?
- How much of the hidden Goal performance comes from repeated deterministic action patterns in this small dataset?

## Compare later

- Match No Memory, GRU, RSSM, and Transformer data splits and encoders.
- Evaluate hidden-state reset and history-shuffle ablations.
- Add multiple stochastic samples and calibration/negative log-likelihood metrics.
- Separate state representation quality from image decoder quality.
- Compare parameter count, latency, memory, and multi-seed variance—not only pixel error.
