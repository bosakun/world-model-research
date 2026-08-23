# Simultaneous Multimodal Fusion

Status: completed on 2026-08-23. Independent educational fusion model.

## Purpose

Fuse vision, proprioception, language command, and touch simultaneously while representing missing modalities explicitly.

## Problem

Different sensors provide complementary state. Numerical zero cannot distinguish missing evidence from a valid zero measurement, and concatenation does not expose which modality the model relies on.

## Previous Model

Phase 10/03 selected one heterogeneous schema per sample. This phase attends across four simultaneous modality tokens.

## Hypothesis

Typed modality tokens and learned missing tokens should predict the next state/image under random dropout. Ablation should reveal proprioception as location-critical and language as action-critical.

## Architecture

```text
vision CNN --\
proprio MLP -+-> four 48-D typed tokens -> 2-layer self-attention -> mean fusion
language Emb-+                                                -> next position
touch MLP ---/                                                -> next image
missing mask -> learned token replacement before attention
```

## Data Flow

Current image/position, a five-way language movement, and four wall-contact bits are encoded separately. A boolean `[B,4]` mask replaces unavailable tokens before fusion. Position and image heads predict the same future.

## Tensor Shapes

Vision `[B,3,16,16]`; proprio/next position `[B,2]`; language `[B]`; touch `[B,4]`; availability `[B,4]`; typed tokens `[B,4,48]`; fused `[B,48]`; next image `[B,3,16,16]`.

## Mathematics

`u_m=A_m(x_m)+e_m` when present and learned `u_missing,m` otherwise. `f=mean Transformer([u_vision,u_prop,u_lang,u_touch])`. Loss is position MSE plus `0.5` image MSE. The mask is applied before attention, so hidden values cannot leak.

## Code Mapping

Modalities/dropout: `dataset.py`; adapters/type/missing embeddings and fusion: `model.py`; joint loss: `train.py`; single-modality ablation: `evaluate.py`.

## Training

Seed 251; `multimodal-navigation-v1`; 1,536/384 samples; 20% random dropout (proprio retained during training-data generation); Adam `8e-4`; batch 64; 50 epochs/1,200 steps; 108,925 parameters; checkpoint format 1.

## Losses

Position loss teaches physical transition; image loss retains visual prediction. There is no contrastive alignment objective: shared future supervision grounds modalities indirectly.

## Evaluation Interface

`evaluate.py` evaluates all modalities and removes each one separately, reporting position RMSE and saving prediction images.

## Smoke Test Results

Four tests passed. RMSE: all `0.0323`, no vision `0.0374`, no proprio `0.5307`, no language `0.1149`, no touch `0.0351`. Validation image MSE `0.01270`.

## Failure Cases

- Proprioception dominates location; visual encoder has little incentive to localize precisely.
- Touch is almost redundant except near walls.
- Image MSE permits blurry forecasts.
- Training always retained proprioception, so its evaluation removal is OOD and intentionally severe.

## Findings

Ablation makes modality roles measurable. Missing-token replacement prevents sentinel-value ambiguity and leakage, but dropout distribution strongly shapes robustness.

## Limitations

No audio, continuous language, asynchronous sensors, calibration, cross-modal reconstruction, or temporal histories. “Touch” is four synthetic bits.

## Compare Later

Concatenation vs attention; modality dropout schedules; no type/missing embedding; vision-only localization; Shapley/attention attribution; missing/noisy/conflicting sensors; error, latency, and robustness.

## Final Model Candidate

```text
Candidate: Yes as an interface, not yet as a final encoder.
Reason: Typed masked fusion works and exposes modality dependence.
Advantages: simultaneous evidence; explicit missingness; modality ablation.
Disadvantages: dominant-modality shortcuts; synchronous fixed token count.
Possible conflicts: large video/3D tokens require scalable cross-attention rather than four-token self-attention.
```

## Next Questions

How does this interface connect to real robot observations/actions? Which modalities remain useful under sensor delay and physical noise?

## References

### Attention Is All You Need

- Authors: Ashish Vaswani et al.; Year: 2017; Paper: https://arxiv.org/abs/1706.03762
- Used for: typed-token self-attention fusion in `model.py`.

### Learning Interactive Real-World Simulators (UniSim)

- Authors: Sherry Yang et al.; Year: 2023; Paper: https://arxiv.org/abs/2310.06114
- Used for: heterogeneous conditioning context and transition from selected schemas to simultaneous evidence.

Classification: **Independent educational implementation**; no specific multimodal benchmark is reproduced.
