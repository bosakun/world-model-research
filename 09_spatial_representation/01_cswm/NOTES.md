# Research Notes

## Before

- Expected object-wise latent prediction to be easy once binding is fixed; the main check was whether contrastive negatives prevent collapse.
- Initially easy to conflate object factorization with object discovery. They are separate problems.

## During implementation

- Equal/opposite close-range repulsion creates a real pairwise term rather than two independent controlled dots.
- Colors map deterministically to slots. This makes evaluation interpretable but is an explicit shortcut.
- The linear probe is post-hoc only; coordinate labels never enter the representation loss.

## Results

- Four tests passed.
- Positive/negative energies: `0.00507 / 13.0353`; hinge nearly inactive (`0.00265`) after separation.
- Probe RMSE current/next: `0.00587 / 0.01306`.
- The very large negative energy suggests easy negatives; hard-negative evaluation is needed.

## Article material

- `object_transition.png`: true versus probed object positions.
- `energy_curve.png`: early collapse pressure and later margin separation.
- Explanation: C-SWM answers “how objects interact” after someone has told it “which pixels belong to which object” in this simplified version.

## Compare later

Remove messages, permute colors, add a third entity, construct near-identical negatives, and replace fixed binding with Slot Attention.
