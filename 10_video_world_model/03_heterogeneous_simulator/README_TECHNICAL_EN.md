# Heterogeneous Conditional Simulator

Status: completed on 2026-08-23. Educational heterogeneous-conditioning study inspired by UniSim; not a generative real-world simulator reproduction.

## Purpose

Train one image world model across datasets whose controls are represented as continuous motor vectors, discrete language commands, or target coordinates.

## Problem

Real world-model datasets differ in sensors, controls, and annotations. Naively concatenating unavailable fields can leak defaults or make one source dominate. Each schema needs an adapter into a shared condition space plus source-aware evaluation.

## Previous Model

Latent-action dynamics used one inferred categorical interface. It did not test supervised heterogeneous data sources or adapter selection.

## Hypothesis

Three modality-specific adapters plus a source embedding can drive one shared latent transition/decoder with comparable per-source error.

## Architecture

```text
current image -> CNN -> z_t
motor[2] --Linear--\
language ID--Embed--+-> select by source type + source embedding -> c_t
goal[2] ---Linear--/
z_t+c_t -> residual transition -> decoder -> next image
```

## Data Flow

The synthetic corpus interleaves sources exactly. All records carry fields for batching, but only the adapter selected by `kind` contributes. The shared transition and decoder receive a fixed 16-dimensional condition.

## Tensor Shapes

Images `[B,3,16,16]`; kind/language `[B]`; motor/goal `[B,2]`; candidate adapter outputs `[B,3,16]`; selected condition `[B,16]`; latent `[B,32]`.

## Mathematics

`c=A_m(m)` if motor, `A_l(l)` if language, or `A_g(g)` if goal, plus learned source embedding. `z'=tanh(z+f(z,c))`; `x'=sigmoid(d(z'))`. Foreground-weighted MSE counters sparse black background.

## Code Mapping

Schemas/source balance: `dataset.py`; adapters and selection: `model.py::condition`; shared simulator: `model.py::forward`; source-stratified losses/plots: `train.py`, `evaluate.py`.

## Training

Seed 239; `heterogeneous-square-controls-v1`; 1,536/384 samples equally interleaved; Adam `8e-4`; batch 64; 50 epochs/1,200 steps; 56,739 parameters; checkpoint format 1.

## Losses

Foreground-weighted image MSE trains all sources through one output space. No cross-modal alignment loss is required because conditions share known transition semantics in this synthetic setup.

## Evaluation Interface

`evaluate.py` reports unweighted MSE separately for motor, language, and goal sources and visualizes one example each; aggregate-only reporting is intentionally avoided.

## Smoke Test Results

Four tests passed. MSE motor/language/goal: `0.01400 / 0.01417 / 0.01427`; maximum source error `0.01427`. Adapter isolation test confirms changing unused motor fields cannot affect non-motor conditions.

## Failure Cases

- Shared decoder can output blurry averages; MSE does not assess perceptual realism.
- All sources describe the same five simple motions, so schema heterogeneity exceeds semantic heterogeneity.
- Source embedding can encourage separate shortcuts rather than shared grounding.
- Missing/noisy/conflicting conditions are not represented.

## Findings

A typed adapter boundary prevents absent-field ambiguity and makes per-source failure measurable. Comparable synthetic errors do not imply real heterogeneous dataset scaling.

## Limitations

One object, one-step deterministic generation, aligned image domain, balanced sources, no diffusion, video history, cameras, robots, internet data, or real-world domain gaps. UniSim operates at an incomparable scope.

## Compare Later

Remove source embedding, imbalance sources, hold out command/schema combinations, add conflicts/missing inputs, compare shared vs separate models, and measure worst-source error, transfer, calibration, and sample efficiency.

## Final Model Candidate

```text
Candidate: Undecided
Reason: Typed heterogeneous conditioning works on aligned synthetic sources only.
Advantages: one shared dynamics; explicit schema adapters; per-source auditability.
Disadvantages: source shortcuts; aligned-domain assumption; blurry MSE generation.
Possible conflicts: multimodal fusion needs simultaneous evidence, while this module selects one schema.
```

## Next Questions

How should multiple modalities be fused simultaneously rather than selected? How do modality masks distinguish missing from zero-valued evidence?

## References

### Learning Interactive Real-World Simulators (UniSim)

- Authors: Sherry Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, Pieter Abbeel
- Year: 2023
- Paper: https://arxiv.org/abs/2310.06114
- Used for: motivation to combine heterogeneous data and conditioning into a shared interactive simulator.
- Implementation: `dataset.py`, `model.py`.

Classification: **Simplified educational implementation**. The adapter experiment does not reproduce UniSim's generative architecture, datasets, or scale.
