# Contrastive Structured World Model (C-SWM) Mechanism Study

Status: completed on 2026-08-22. Simplified educational implementation with known color-to-object binding; not a paper benchmark reproduction.

## Purpose

Represent a scene as an ordered set of object slots and predict each object's next slot using action effects plus pairwise relational messages.

## Problem

A single flat latent can entangle multiple entities. When one object acts on another, a model should preserve object identity and express pairwise interaction rather than relearn every scene combination globally.

## Previous Model

Earlier phases use one vector for the whole observation. They can encode object information, but expose no explicit object axis or relational aggregation.

## Hypothesis

Object-wise slots, a shared transition, and pairwise messages should learn the two-object transition under a contrastive energy objective; next positions should be recoverable by a post-hoc linear probe.

## Architecture

```text
RGB image --known color channel split--> shared CNN --> slots [K=2,D=8]
                                      each slot i:
                 sum_j!=i edge(z_i,z_j) + action_i --> node MLP --> z_hat_i'

next image --> same encoder --> positive z'
other batch sample ----------> negative z-
```

## Data Flow

Two Gaussian objects move under per-object actions and equal/opposite close-range repulsion. Current and next images share a single encoder. Predicted slots are pulled toward true next slots and separated from batch-shifted negative scenes. A linear position probe is fitted only after representation training.

## Tensor Shapes

Images `[B,3,16,16]`; positions/actions `[B,2,2]`; slots and predicted slots `[B,2,8]`; positive/negative energy `[B]`. The third image channel is background; the first two establish known object identity.

## Mathematics

```text
m_i = sum_(j != i) g_edge(z_i,z_j)
z_hat_i' = z_i + g_node(z_i,a_i,m_i)
E+ = mean_i ||z_hat_i' - z_i'||^2
E- = mean_i ||z_hat_i' - z_i^-||^2
L = E+ + max(0, margin + E+ - E-).
```

The shared edge/node functions encode compositional relational structure. The hinge requires the correct future to have lower energy than an unrelated scene by a margin.

## Code Mapping

- renderer/true relation: `dataset.py::render_objects`, `relational_transition`
- object slots: `model.py::ColorObjectEncoder`
- graph messages: `model.py::RelationalTransition`
- energy objective: `losses.py::contrastive_world_model_loss`
- linear-probe audit: `evaluate.py::linear_probe`

## Training

Seed 151; dataset `two-object-relational-v1`; 768/192 train/validation transitions; Adam `1e-3`; batch 64; 50 epochs/600 steps; margin 1; 275,776 parameters; checkpoint format 1.

## Losses

Positive energy trains action-conditioned next-slot accuracy. Negative hinge prevents a trivial constant encoding by demanding scene discrimination. It does not reconstruct pixels or supervise coordinates.

## Evaluation Interface

`python 09_spatial_representation/01_cswm/evaluate.py` fits a linear probe on held-out representation samples, evaluates another split, and writes metrics plus `object_transition.png`.

## Smoke Test Results

Four tests passed. Validation positive energy `0.00507`, negative energy `13.0353`, hinge `0.00265`. Current-position probe RMSE was `0.00587`; predicted-next-position probe RMSE was `0.01306`.

## Failure Cases

- Slot identity is given by color channel, so this does not discover objects.
- Batch-shift negatives may be accidentally similar and do not cover hard structured confounders.
- A fixed two-object loop does not demonstrate variable cardinality.
- Linear probe quality can be high while unobserved appearance/identity information is absent.

## Findings

The explicit object axis makes one shared relational transition sufficient for both entities. Contrastive training preserves position-like information without an image decoder, but known binding is a strong shortcut.

## Limitations

The visual extractor differs substantially from C-SWM's experimental setup; actions are per-object, colors never swap, and only one-step deterministic transitions are trained. This is an independent synthetic mechanism study.

## Compare Later

Compare flat-vector dynamics, known slots, Slot Attention slots, and relational ablations. Measure energy ranking, object probe, rollout error, slot permutation stability, parameter count, and entity-count transfer. Ablate edges, negatives, shared weights, and fixed binding.

## Final Model Candidate

```text
Candidate: Undecided
Reason: Relational object dynamics work, but the encoder receives object identity as a color-channel shortcut.
Advantages: compositional slots; shared dynamics; interpretable object axis.
Disadvantages: fixed K/order; contrastive negative sensitivity; no unsupervised binding.
Possible conflicts: permutation-equivariant slots require matching before ordered losses.
```

## Next Questions

Can Slot Attention infer object assignment without fixed color binding? How should a transition match slots when their order changes?

## References

### Contrastive Learning of Structured World Models

- Authors: Thomas Kipf, Elise van der Pol, Max Welling
- Year: 2019
- Paper: https://arxiv.org/abs/1911.12247
- Used for: object-factorized latent state, relational transition, energy-based contrastive objective.
- Implementation: `model.py`, `losses.py`.

Classification: **Simplified educational implementation** and **independent synthetic environment**. Known channel binding, renderer, and transition physics are project-specific.
