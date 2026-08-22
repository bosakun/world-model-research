# Action-Conditioned 3D Occupancy World Model

Status: completed on 2026-08-22. Compact binary-voxel mechanism study inspired by occupancy world models; not an OccWorld reproduction.

## Purpose

Represent occupied free-space structure explicitly on a 3D voxel grid and predict multi-step future occupancy under three-dimensional actions.

## Problem

Object vectors are compact but do not directly represent volume, free space, occlusion, or geometry needed by embodied and autonomous systems. Dense occupancy offers spatial semantics at a computational cost.

## Previous Model

C-SWM/SlotFormer use two-dimensional point slots. They cannot answer which 3D cells will be occupied.

## Hypothesis

A 3D convolutional encoder/decoder with action-conditioned latent transition can roll a moving voxel sphere forward. Sparse voxel losses alone may blur location; latent consistency and center-of-mass supervision should improve geometry.

## Architecture

```text
occupancy_t [1,8,8,8] -> Conv3D encoder -> z_t [32]
z_t + action_t [3] -> residual latent transition -> z_hat_(t+1)
z_hat_(t+1) -> ConvTranspose3D decoder -> occupancy logits_(t+1)
                                      -> repeat for six-step rollout
```

## Data Flow

A sphere center moves by `0.75*tanh(action)` inside grid bounds. Training starts from the first encoded occupancy and recursively predicts six latent grids, matching occupancy, encoded future latents, and true geometric centers.

## Tensor Shapes

Occupancies `[B,7,1,8,8,8]`; actions/centers `[B,6,3]` / `[B,7,3]`; latent rollout `[B,6,32]`; logits `[B,6,1,8,8,8]`; horizon IoU `[6]`.

## Mathematics

```text
z_0=e(O_0)
z_hat_(t+1)=tanh(z_hat_t+f(z_hat_t,a_t))
p_hat_(t+1)=sigmoid(d(z_hat_(t+1))).
```

Voxel objective combines positive-weighted BCE and soft Dice:

```text
L_voxel = BCE_pos(logits,O) + [1 - (2 sum pO)/(sum p + sum O)].
```

Auxiliary project-specific terms are

```text
L_cons=||z_hat_t-sg(e(O_t))||²
L_center=||sum_x p(x)x/sum_x p(x)-center_t||²
L=L_voxel+L_cons+0.25 L_center.
```

The center loss uses privileged synthetic center labels and must not be confused with unsupervised occupancy learning.

## Code Mapping

- sphere renderer/action sequences: `dataset.py`
- Conv3D latent model: `model.py::OccupancyWorldModel`
- BCE/Dice: `losses.py::occupancy_loss`
- recursive consistency and geometric center: `train.py::batch_loss`, `occupancy_center`
- horizon IoU/projections: `evaluate.py`

## Training

Seed 191; `moving-voxel-sphere-v1`; 512/128 sequences; Adam `8e-4`; batch 32; 80 epochs/1,280 steps; 25,145 parameters; latent 32; checkpoint format 1.

## Losses

BCE handles cell classification and class imbalance; Dice emphasizes overlap; consistency stabilizes recursive latent transitions; center loss supplies explicit spatial localization. Removing any term changes what “correct occupancy” means.

## Evaluation Interface

`python 09_spatial_representation/04_occupancy_3d/evaluate.py` thresholds probability at 0.5, reports voxel IoU at each rollout horizon, and writes top-down true/predicted occupancy projections.

## Smoke Test Results

Five tests passed. Validation BCE/Dice were `0.0962/0.1665`. IoU over horizons 1–6 was `0.808, 0.796, 0.802, 0.777, 0.762, 0.746`.

## Failure Cases

- Initial BCE+Dice-only run produced broad occupancy and final IoU `0.111`.
- Positive weighting can overpredict occupied cells.
- Center supervision can hide failures on multimodal/non-spherical geometry.
- Dense grids scale cubically with resolution.
- Thresholded IoU hides probability calibration.

## Findings

Explicit geometric auxiliary signals changed the six-step result dramatically. A good occupancy objective must distinguish shape overlap from spatial localization. Final IoU still decays with horizon.

## Limitations

Only one binary sphere, no semantics, occlusion, ego pose, multi-agent motion, camera/LiDAR perception, unknown cells, or large 3D scene. OccWorld uses substantially richer occupancy tokens and driving prediction.

## Compare Later

Compare BCE/Dice/center/consistency ablations, dense voxels versus object slots, grid resolutions, calibration, horizon IoU, center error, compute/memory, and multiple/multimodal objects.

## Final Model Candidate

```text
Candidate: Undecided
Reason: Spatial occupancy rollout works at 8^3, but privileged center supervision and cubic scaling limit generality.
Advantages: explicit volume/free-space cells; geometric metrics; 3D action conditioning.
Disadvantages: dense memory; sparse imbalance; auxiliary-label dependence.
Possible conflicts: object slots are compact while voxel grids are dense; fusion needs consistent coordinates.
```

## Next Questions

Can sparse/latent occupancy tokens retain geometry without cubic cost? How should video observations infer occupancy under occlusion?

## References

### OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving

- Authors: Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, Jiwen Lu
- Year: 2023
- Paper: https://arxiv.org/abs/2311.16038
- Used for: motivation for occupancy as world state and future occupancy prediction.
- Implementation: conceptual lineage for `model.py` and `evaluate.py`.

Classification: **Simplified educational implementation** and **independent synthetic modification**. Binary 8³ spheres, continuous actions, and privileged center loss are not an OccWorld reproduction.
