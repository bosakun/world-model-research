# VQ Video Tokenizer

Status: completed on 2026-08-22. Small per-frame VQ tokenizer; not a Genie tokenizer reproduction.

## Purpose

Convert image frames into a discrete 4×4 token grid that later temporal models can predict categorically instead of generating every pixel directly.

## Problem

Raw video prediction is high dimensional. Continuous latents can drift, while discrete tokens define a finite vocabulary and categorical prediction target. Quantization may underuse its codebook.

## Previous Model

Earlier visual models use continuous latent vectors/slots/voxels. None exposes a reusable discrete spatial vocabulary for video.

## Hypothesis

A 32-entry VQ codebook should reconstruct moving-square frames with 16 tokens per frame, but the simple dataset may activate far fewer codes.

## Architecture

```text
frame [3,16,16] -> stride-2 CNN x2 -> continuous [16,4,4]
 -> nearest of 32 learned embeddings -> token IDs [4,4]
 -> straight-through quantized grid -> transpose CNN -> reconstructed frame
```

## Data Flow

Six-frame synthetic videos are flattened into frames for tokenizer training. The encoder produces 16-dimensional vectors at 16 spatial locations. Nearest code IDs are stored; straight-through quantized embeddings feed the decoder.

## Tensor Shapes

Video `[B,6,3,16,16]`; flattened frames `[6B,3,16,16]`; continuous/quantized grid `[6B,16,4,4]`; IDs `[B,6,4,4]`; reconstruction matches the video shape.

## Mathematics

```text
k*=argmin_k ||e(x)-c_k||²
z_q=c_k*
z_st=e(x)+sg(z_q-e(x))
L=MSE(x_hat,x)+||sg(e)-z_q||²+beta||e-sg(z_q)||².
```

Nearest-neighbor quantization creates tokens. The straight-through expression uses quantized values forward but encoder gradients backward. Codebook and commitment terms respectively move codes toward encodings and encodings toward codes.

## Code Mapping

Dataset/action alignment: `dataset.py`; nearest lookup/loss: `model.py::VectorQuantizer`; frame tokenization/reconstruction: `VQFrameTokenizer`; code usage: `evaluate.py`.

## Training

Seed 211; `moving-square-video-v1`; 512/128 videos; Adam `1e-3`; batch 64 videos; 50 epochs/400 steps; beta 0.25; 20,051 parameters; checkpoint format 1.

## Losses

Pixel MSE preserves visible content. Codebook loss updates embeddings. Commitment loss limits encoder/code mismatch. No temporal loss is used, so token semantics are spatial, not explicitly motion-aware.

## Evaluation Interface

`python 10_video_world_model/01_vq_video_tokenizer/evaluate.py` reports reconstruction, active codes, perplexity, and writes all six reconstructed frames.

## Smoke Test Results

Four tests passed. Reconstruction MSE `0.003565`; 5/32 codes active; perplexity `2.891`; 16 tokens/frame.

## Failure Cases

- Codebook underuse: most entries never win nearest-neighbor assignment.
- Per-frame training ignores temporal consistency; nearby frames may change codes abruptly.
- Sparse background consumes vocabulary and dominates pixel loss.
- The square's limited appearance does not require a rich codebook.

## Findings

Discrete tokenization and reconstruction work, but nominal vocabulary size is not effective capacity. Perplexity/active-code reporting is mandatory before assuming 32 useful symbols.

## Limitations

Tiny 16×16 videos, one moving shape, no perceptual/adversarial loss, EMA codebook, code reset, multi-scale tokenizer, or joint dynamics training.

## Compare Later

Continuous vs VQ latents; code sizes/dimensions; EMA/reset; temporal consistency; reconstruction/perceptual error; perplexity, bitrate, downstream next-token accuracy, and rollout quality.

## Final Model Candidate

```text
Candidate: Undecided
Reason: Discrete interface works, but effective codebook usage is only ~3 perplexity.
Advantages: categorical dynamics target; compact spatial grid; inspectable usage.
Disadvantages: codebook collapse; quantization error; per-frame inconsistency.
Possible conflicts: joint temporal training may destabilize the visual codebook.
```

## Next Questions

Can future tokens be predicted from past tokens when physical actions are hidden? Can a learned latent action recover intervention categories?

## References

### Neural Discrete Representation Learning (VQ-VAE)

- Authors: Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
- Year: 2017
- Paper: https://arxiv.org/abs/1711.00937
- Used for: nearest-neighbor codebook, straight-through estimator, codebook and commitment losses.
- Implementation: `model.py`.

### Genie: Generative Interactive Environments

- Authors: Jake Bruce et al.
- Year: 2024
- Paper: https://arxiv.org/abs/2402.15391
- Used for: motivation for discrete video tokens before latent-action dynamics.
- Implementation: conceptual phase decomposition only; Genie's tokenizer scale/objective is not reproduced.

Classification: **Simplified educational implementation**.
