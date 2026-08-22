# Latent Overshooting Research Notes

Date: 2026-08-22. Future article material.

## Before implementation

- Expected five-step overshooting to improve local rollout robustness, but not necessarily 30-step prediction.
- Wanted an exact state-space error without image-background shortcuts.
- Key question: how misleading can one-step MSE be?

## What implementation clarified

- Overshooting is not merely computing loss at later dataset indices; it must feed predicted intermediate states.
- Distance-1 appears both in the one-step term and overshooting aggregate here. That weighting choice must be disclosed.
- Every start time contributes, so a length-30 sequence creates many overlapping recursive graphs.

## Results

- Five tests passed; 4,738 parameters; seed 47; 320 Adam steps.
- One-step MSE `0.0001605`.
- Overshooting MSE rises by distance: `0.000161, 0.000897, 0.002774, 0.006445, 0.012545`.
- Open-loop MSE: horizon 5 `0.006229`, horizon 10 `0.066463`, horizon 30 `1.408694`.

## Unexpected / important behavior

- The long-horizon failure is much larger than the final training loss suggests.
- Error grows smoothly and strongly after the optimized five-step window.
- Training with an overshooting mechanism is not equivalent to solving long-horizon imagination.

## Figures / article ideas

- `outputs/compounding_error.png`: strongest Before/After-understanding figure; one-step reference versus horizon curve.
- `outputs/long_rollout.png`: position/velocity drift from the true nonlinear system.
- `outputs/loss_curve.png`: optimization success alongside evaluation failure.
- Explain local map composition using a small Jacobian-error diagram.

## Compare later

- One-step-only baseline with exactly matched architecture/seed.
- K=3/5/10 and distance weighting.
- Macro-transition model to reduce 30 primitive transitions to fewer learned steps.
- Probabilistic rollout coverage as deterministic error leaves support.
