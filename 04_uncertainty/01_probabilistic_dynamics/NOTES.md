# Probabilistic Dynamics Research Notes

Date: 2026-08-22. Material for a future Zenn article, not the article.

## Before implementation

- Question: will one Gaussian network actually learn a changing noise scale rather than use variance to hide mean errors?
- Prediction: in-distribution coverage should approach Gaussian reference values; edge/OOD behavior will be less trustworthy.
- Important distinction to preserve: RSSM stochastic latent and explicit next-state aleatoric likelihood are related but not identical mechanisms.

## What implementation clarified

- The model never receives true noise std during training. It sees only one sampled next state per transition and infers variance through NLL residual statistics.
- Mean error and variance are coupled in NLL: confident errors cost heavily, but wide uncertainty also costs through `log variance`.
- Sampling a trajectory changes the next input, so uncertainty compounds nonlinearly rather than being a plot-only error bar.

## Errors / misunderstandings

- Negative final NLL initially looks suspicious. It is valid because continuous probability **density** may exceed 1; only integrated probability is bounded by 1.
- A good predicted-vs-true std plot within the training range does not prove OOD reliability. The curve overshoots strongly near `x=1`, outside the sampled training support `[-0.8,0.8]`.
- High coverage alone could come from excessively wide predictions. NLL, mean std, RMSE, and correlation were recorded together.

## Results

- Eight tests passed.
- Seed 37, 4,872 parameters, 1,280 Adam steps.
- Validation NLL `-3.73237`; held-out evaluation NLL `-3.59307`.
- RMSE `0.05686`.
- 1σ/2σ coverage `0.6953/0.9531`.
- Predicted/known std correlation `0.9376`.
- Mean predicted std `(0.0532,0.0453)` versus truth `(0.0539,0.0438)`.

## Interesting behavior

- The learned horizontal std follows the sigmoid-like ground truth through the supported region.
- Sampled paths fan out even though their actions are identical.
- Mean path can look clean and plausible while ignoring substantial trajectory risk.
- The OOD overshoot is an ideal bridge from aleatoric to epistemic uncertainty.

## Figures / article material

- `outputs/aleatoric_std.png`: true versus learned heteroscedasticity plus visible OOD failure.
- `outputs/sampled_rollouts.png`: 64 model particles, one true noisy path, and mean path.
- `outputs/loss_curve.png`: explain why negative Gaussian NLL is allowed.
- Possible diagram: “same action, many outcomes” versus “same model family, many parameter hypotheses.”

## Later comparisons

- deterministic MSE mean versus probabilistic mean.
- fixed, homoscedastic, and heteroscedastic variance.
- calibration with more/less training data.
- ensemble epistemic disagreement inside and outside `[-0.8,0.8]`.
- sample propagation choices and their computation cost.
