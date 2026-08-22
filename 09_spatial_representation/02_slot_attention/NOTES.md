# Research Notes

## Before

- Prediction: identical Gaussian objects may be difficult but three slots should roughly split two foregrounds/background.
- Key risk: sparse images allow near-black reconstruction to look numerically good.

## Iteration 1: identical objects, plain MSE

- Validation/unweighted MSE around `0.019`; mean mask IoU `0.302`.
- Visualization showed essentially black reconstruction and diffuse masks.
- Lesson: reconstruction loss magnitude must be compared with a trivial background baseline.

## Iteration 2: larger objects and foreground-weighted MSE

- Reconstruction became visible and MSE improved, but masks still cooperated globally rather than one slot per object.
- Added weighting is an experimental modification and is recorded in README.

## Iteration 3: colored objects, shared encoder

- Removed the identical-appearance ambiguity while keeping a single shared image encoder (no channel-to-slot routing).
- Unweighted MSE `0.01453`; IoU only `0.2713`.
- Therefore the main remaining problem is specialization/decoder dynamics, not merely visual indistinguishability.

## Errors and fixes

- Evaluation initially passed `[1,H,W]` to Matplotlib for an argmax mask; selected the singleton mask channel explicitly.
- Deterministic use of the shared slot mean made every initial slot identical. Evaluation now samples slot initialization with a fixed RNG seed, matching the symmetry-breaking requirement.

## Article material

- `slot_decomposition.png` is a failure-case figure: good enough reconstruction does not imply objects.
- `loss_curve.png` shows continued loss improvement without validating slot semantics.
- Explain the two attention normalizations with axes `[tokens,slots]`.
- Useful message: “object-centric” is an empirical property to measure, not guaranteed by naming a tensor `slots`.

## Compare later

Conv decoder, more seeds/steps, ARI, temporal consistency, hard masks, entropy/diversity terms, and supervised-mask upper bound.
