# Research Notes

## Before

- Expected severe foreground/background imbalance: a small sphere occupies few of 512 cells.
- Question: is Dice enough to force correct location during recursive latent rollout?

## First run

- BCE+Dice only: validation BCE/Dice `0.533/0.865`, final IoU `0.111`.
- Predictions were broad; voxel mass and location were insufficiently constrained through the recursive bottleneck.

## Revision

- Added stop-gradient future-encoder consistency.
- Added differentiable probability center-of-mass versus synthetic true center, weight 0.25.
- This is privileged supervision and is clearly classified as an experimental modification.

## Final smoke result

- Five tests passed, including exact single-voxel center calculation.
- Final horizon IoU improved to `0.746`; horizon 1 was `0.808`.
- Validation Dice loss fell to `0.1665`.

## Article material

- Before/after IoU `0.111 -> 0.746` is useful for explaining that losses specify geometry.
- `occupancy_rollout.png`: top-down true/predicted projections across six steps.
- Explain why center supervision is powerful but inappropriate for multiple disconnected objects without modification.

## Compare later

Each loss ablation, probability calibration, larger grids, multiple objects, and sparse representations.
