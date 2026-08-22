# Research Notes

## Before

- Expected high reconstruction quality because scenes contain one square and mostly background.
- Risk: a 32-code vocabulary is much larger than visual diversity.

## Results

- Four tests passed.
- MSE `0.003565`, active codes `5/32`, perplexity `2.891`.
- VQ loss rose transiently near epoch 10 (`~0.71`) while encoder/codebook reorganized, then fell; useful stability plot.

## Implementation insights

- Token shape must restore both video time and spatial 4×4 axes; flattening frames is only a training convenience.
- Straight-through was tested by confirming gradients reach continuous input.
- Good pixels do not imply a rich discrete vocabulary.

## Article material

- `video_reconstruction.png`; codebook-size vs active-code/perplexity comparison.
- Explain “32 IDs exist, but effective vocabulary is ~2.9.”

## Compare later

Dead-code reset, EMA VQ, smaller vocabulary, temporal token stability, and downstream dynamics accuracy.
