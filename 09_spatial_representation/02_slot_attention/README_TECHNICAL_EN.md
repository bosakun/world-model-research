# Slot Attention Object Binding

Status: completed mechanism study on 2026-08-22. The algorithm executes and reconstructs, but this smoke run did not obtain clean object masks; this is recorded as a failed discovery result.

## Purpose

Replace manually routed object channels with iterative competitive attention that maps spatial image tokens into a fixed set of exchangeable slots.

## Problem

C-SWM assumed that red pixels always belong to slot 0 and green pixels to slot 1. A general object-centric model must infer which tokens belong together and tolerate arbitrary slot order.

## Previous Model

`01_cswm::ColorObjectEncoder` receives object identity as preprocessing. Its relational state is useful only after binding has already been solved.

## Hypothesis

Three Slot Attention iterations plus a spatial broadcast decoder should reconstruct two colored objects and induce two foreground slots plus a background slot. Sparse images may instead admit a background/global reconstruction shortcut.

## Architecture

```text
image [3,16,16] -> CNN + coordinate embedding -> 256 feature tokens
                                          |
                              3 sampled initial slots
                                          |
              repeat 3x: token-to-slot competition -> weighted updates
                         -> GRUCell -> residual MLP
                                          |
                         spatial broadcast decoder per slot
                                          |
                       RGB components + softmax masks -> reconstruction
```

## Data Flow

All RGB channels enter one shared encoder; there is no hard channel-to-slot split. Attention normalizes over slots for each token (competition), then normalizes each slot over tokens (weighted mean). Slots are independently decoded and mask-softmaxed into one image.

## Tensor Shapes

Images/reconstruction `[B,3,16,16]`; tokens `[B,256,32]`; slots `[B,3,32]`; competitive attention `[B,256,3]`; slot RGB `[B,3,3,16,16]`; masks `[B,3,1,16,16]` summing to one over slots.

## Mathematics

```text
logit_nk = k(x_n)^T q(s_k) / sqrt(D)
a_nk = softmax_k(logit_nk)
w_nk = a_nk / sum_n a_nk
u_k = sum_n w_nk v(x_n)
s_k <- GRU(u_k,s_k) + MLP(LN(s_k)).
```

Competition is over `k`, so each token distributes responsibility among slots. Re-normalization over `n` makes each slot update a weighted mean. The decoder produces `(rgb_k,alpha_k)` and uses `softmax_k(alpha)` before summation.

Training minimizes a foreground-weighted reconstruction objective:

```text
L = mean[(1 + 8 mean_channel(image)) (reconstruction-image)^2].
```

This weighting is an **experimental modification**, not a defining Slot Attention equation. It was added after plain MSE learned an almost-black sparse-background shortcut.

## Code Mapping

- synthetic scenes/true masks: `dataset.py`
- competitive iteration: `model.py::SlotAttention`
- image tokenizer and broadcast decoder: `model.py::SlotAttentionAutoencoder`
- weighted reconstruction: `train.py::reconstruction_loss`
- permutation-aware audit: `evaluate.py::best_permutation_iou`

## Training

Seed 163; `colored-two-object-images-v2`; 768/192 examples; Adam `4e-4`; batch 64; 80 epochs/960 steps; 46,788 parameters; three slots/iterations; checkpoint format 1. Slot initialization is sampled from learned Gaussian parameters; evaluation is reproducible by resetting the RNG seed.

## Losses

Only reconstruction supervises slots. Ground-truth masks are used for evaluation, never loss. Foreground weighting corrects class imbalance but does not force one object per slot.

## Evaluation Interface

`python 09_spatial_representation/02_slot_attention/evaluate.py` reports unweighted reconstruction MSE and best-permutation mean mask IoU, avoiding a false penalty for arbitrary slot numbering.

## Smoke Test Results

Four tests passed. Weighted validation loss reached `0.03484`; unweighted reconstruction MSE was `0.01453`. Mean best-permutation IoU was only `0.2713` (best example `0.3437`). Therefore reconstruction worked partially, but object decomposition did not.

## Failure Cases

- Plain-MSE identical-object v1 collapsed to near-black reconstruction (`MSE 0.0188`, IoU `0.302`).
- Larger objects and foreground weighting prevented black output but masks remained diffuse/global.
- Colored-object v2 improved reconstruction, not binding (`IoU 0.271`).
- Multiple slots can cooperate to draw a whole scene rather than specialize one object each.
- Sampled initialization changes assignments; slot order has no semantic guarantee.

## Findings

Correct attention normalization and iterative updates are necessary but not sufficient for object discovery on every dataset/decoder/training setup. Reconstruction is not evidence of object-centric segmentation. This failed result is retained rather than relabeled as successful slots.

## Limitations

Tiny synthetic data, low resolution, a pixel MLP broadcast decoder, short training, no curriculum, and one seed. Original Slot Attention experiments use richer architectures/data/training. Foreground weighting changes the objective, and no segmentation supervision is used.

## Compare Later

Compare plain MSE/weighted loss, identical/colored objects, decoder capacity, slot count/iterations, entropy/diversity regularizers, seeds, and supervised masks as an upper bound. Metrics: ARI, permutation IoU, reconstruction, slot stability, and downstream dynamics.

## Final Model Candidate

```text
Candidate: No for the current trained configuration; mechanism remains under study.
Reason: Reconstruction improved but slots did not align with objects.
Advantages: exchangeable slots; no hard routing; explicit competitive binding.
Disadvantages: collapse/global reconstruction; stochastic ordering; sensitive training.
Possible conflicts: temporal dynamics require stable slot correspondence or matching.
```

## Next Questions

Can temporal consistency make slots more stable? To isolate that question, SlotFormer should first use reliable ordered object slots rather than treating these failed masks as ground truth.

## References

### Object-Centric Learning with Slot Attention

- Authors: Francesco Locatello et al.
- Year: 2020
- Paper: https://arxiv.org/abs/2006.15055
- Used for: Gaussian slot initialization, competitive iterative attention, GRU updates, spatial broadcast reconstruction, permutation interpretation.
- Implementation: `model.py`.

Classification: **Simplified educational implementation**. Foreground weighting and synthetic scenes are experimental project modifications; benchmark discovery quality is not claimed.
