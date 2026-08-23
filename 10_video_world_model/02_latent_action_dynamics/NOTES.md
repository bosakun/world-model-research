# Research Notes

## Before

- Expected five codes to align with five moves up to permutation.
- Risk: 4×4 VQ grid may not change for a one-pixel motion.

## First run

- Overall token accuracy `0.893`, action alignment `0.253`.
- Most tokens are static background; high accuracy mostly reflected copying.

## Revision

- Increased synthetic action from one to two pixels.
- Weighted changed token positions by 6 total versus 1 unchanged.
- Added changed fraction, changed accuracy, and copy baseline.

## Final result

- Changed accuracy `0.787`, copy baseline `0.777`, overall `0.830`.
- Action alignment still `0.270`; all codes used.
- Conclusion: bottleneck usage is not semantic identifiability.
- Rollout frame-6 accuracy `0.563` shows compounding categorical error.

## Errors and fixes

- Cross-folder `config.py`/`dataset.py` names caused ambiguous Python cache resolution. Replaced path-based imports with a checkpoint-compatible local tokenizer adapter and explicit generator specification.
- Evaluation initially built decoded tensors outside `no_grad`, causing Matplotlib/Numpy conversion failure; decoding now runs under `torch.no_grad()`.

## Article material

- “0.893 accuracy, but no action learned” is a strong metric-shortcut example.
- `latent_action_rollout.png` visualizes compounding token errors.
- Explain code permutation versus genuine semantic mismatch.

## Compare later

Supervised action classifier, longer sequences, higher-resolution tokens, mutual information, and intervention consistency.
