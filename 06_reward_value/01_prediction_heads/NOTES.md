# Reward / Value / Continuation Notes

Date: 2026-08-22. Future article source material.

## Before implementation

- Wanted the distinction between immediate reward and policy-dependent value to be impossible to miss.
- Expected continuation accuracy to look deceptively good because terminal transitions are rarer.
- Required padding after termination so masking behavior could be tested.

## Results / findings

- Five tests passed; seed 59; 9,091 parameters; 640 Adam steps.
- Reward RMSE `0.2515`, value RMSE `0.1354`.
- Continuation accuracy `0.9736`, Brier `0.03194`.
- Mean continuation probability: terminal `0.1820`, nonterminal `0.9502`.
- Accuracy hid the imperfect terminal confidence; probability diagnostics were necessary.

## Implementation insights

- Terminal transition has `valid=1` and `continuation=0`; only following padded transitions have `valid=0`.
- Return recursion uses continuation, so padded rewards cannot leak backward.
- Shared state features receive gradients from all three meanings; later ablation should measure interference.

## Article figures / points

- `outputs/prediction_sequence.png`: align local reward spike, rising value, and continuation drop.
- `outputs/loss_curve.png`: three losses have different scales and convergence.
- Show one timeline with “reward now / value later / continuation exists?” labels.

## Later questions

- TD/lambda versus Monte Carlo targets.
- terminal imbalance and calibration.
- predicted latent rather than true state inputs.
- use these heads in random shooting and CEM planning.
