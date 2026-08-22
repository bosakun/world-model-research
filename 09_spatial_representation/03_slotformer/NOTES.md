# Research Notes

## Before

- Chose ground-truth ordered positions after Slot Attention failed; otherwise a rollout failure would have two inseparable causes.
- Prediction: teacher-forced loss will look much better than horizon-8 rollout.

## Implementation

- The causal mask is constant inside each frame block. A standard token triangular mask would make object 0 unable to see object 1 in the same frame depending on flattening order.
- Inputs omit velocity intentionally, making history functionally necessary.
- Slot-index embeddings are a convenience for ordered synthetic objects, not a solution for exchangeable learned slots.
- Disabled nested-tensor optimization explicitly to avoid an irrelevant PyTorch warning with pre-norm layers.

## Results

- Five tests passed; editing future input frames leaves earlier predictions unchanged.
- Teacher-forced validation MSE `0.001207`.
- Rollout RMSE by horizon: `0.0303, 0.0616, 0.0948, 0.1306, 0.1649, 0.1991, 0.2334, 0.2682`.
- Near-linear early growth is a clean compounding-error figure.

## Article material

- `slot_rollout.png` shows entity paths and error-by-horizon.
- Explain causal masks as frame blocks, not just a diagonal triangle.
- Contrast “perfect object state, imperfect future” with the Slot Attention failure “imperfect object state before dynamics.”

## Compare later

Graph transition vs Transformer, no slot ID, shuffled slot order, multi-step loss, and learned Slot Attention inputs.
