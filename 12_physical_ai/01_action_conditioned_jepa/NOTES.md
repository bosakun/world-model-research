# Research Notes

## Before

- Expected target representation to be easier to probe than predicted next representation.
- Main risk was a low prediction loss caused by collapsed embeddings.

## Results

- Four tests passed; target encoder receives no gradients and EMA changes it.
- Action/zero-action probe RMSE `0.160/0.184`; target representation `0.0139`.
- Predicted latent std `0.346`, not constant but below variance target.
- Prediction loss stayed near `0.004` while variance dominated total loss: a useful warning against reporting total/alignment alone.

## Article material

- `physical_prediction.png`: action/no-action/target probe gap.
- Explain “decoder-free” as changing the target, not removing dynamics.
- Predictor error versus representation error separates two research problems.

## Compare later

EMA/variance/covariance ablations, multi-step JEPA, noisy images, and downstream MPC.
