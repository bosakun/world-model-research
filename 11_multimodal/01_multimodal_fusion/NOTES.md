# Research Notes

## Before

- Expected proprio to dominate position and language to specify delta.
- Explicit mask required because touch zero means valid “no wall contact.”

## Results

- Four tests passed, including masked-value non-leakage after adding +100 to touch.
- All/no-vision/no-proprio/no-language/no-touch RMSE: `0.0323/0.0374/0.5307/0.1149/0.0351`.
- This simple environment makes vision/touch redundant; do not generalize to robotics.

## Article material

- `modality_ablation.png` and RMSE table.
- Explain schema selection (Phase 10) versus simultaneous fusion (Phase 11).
- Dominant modality shortcuts persist even with Attention.

## Compare later

Train with proprio dropout, inject noise/delay, conflicting commands, vision-only localization, and attribution stability.
