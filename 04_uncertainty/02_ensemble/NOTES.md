# Probabilistic Ensemble Research Notes

Date: 2026-08-22. Research material for a future article.

## Before implementation

- Prediction: epistemic disagreement should rise outside the `[-0.8,0.8]^2` training square.
- Question: will member variance heads absorb OOD errors and obscure mean disagreement?
- Question: how different are TS1 and TS∞ when models are similar?

## Implementation insights

- The decomposition is simple; obtaining genuinely diverse, calibrated members is the hard part.
- Member identity is hidden state for TS∞ particles: the same physical state/action can evolve differently because the sampled model hypothesis persists.
- TS1 and TS∞ are assumptions about how model uncertainty should correlate across time.

## Error and fix

Evaluation initially imported `01_probabilistic_dynamics/train.py` instead of local `02_ensemble/train.py`.

Cause: adding the sibling experiment to `sys.path` at index 0 shadowed local generic names (`train.py`, `config.py`).

Fix: append the reusable sibling path so the current experiment directory remains first. This is a concrete warning about numeric independent folders and conventional wrapper names.

## Results

- 15 combined uncertainty tests passed.
- Seed 41, bootstrap seed 42, five members, 24,360 total parameters.
- 960 Adam steps per member.
- Held-out moment NLL `-3.58230`, RMSE `0.06051`.
- 1σ/2σ total coverage `0.7324/0.9668` (somewhat conservative).
- Aleatoric std correlation `0.9526`.
- ID epistemic std `0.01066`; OOD `0.01640`; ratio `1.5383`.

## Unexpected / important behavior

- Epistemic uncertainty increased OOD, but much less than hoped.
- Member aleatoric variance dominated total OOD uncertainty. The formula separates statistical moments, but the networks can assign extrapolation error to the within-member term.
- The epistemic heatmap is darker through most of the training square and brighter toward boundaries, which is qualitatively useful despite the modest ratio.

## Figures / article material

- `outputs/epistemic_map.png`: training support rectangle against spatial disagreement.
- `outputs/uncertainty_decomposition.png`: strongest figure for explaining both successful decomposition and aleatoric OOD absorption.
- `outputs/trajectory_sampling.png`: TS∞ versus TS1 model assignment over the same action sequence.
- Suggested equation figure: within-member spread + between-member spread.

## Later comparisons

- More/less data and whether epistemic decreases.
- Ensemble size and bootstrap/no-bootstrap.
- Randomized prior functions or explicit diversity.
- Mixture log-likelihood versus moment Gaussian NLL.
- TS1/TS∞ rollout coverage and downstream CEM decisions.
