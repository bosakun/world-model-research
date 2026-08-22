# Generated outputs

Produced on 2026-08-22 with the default seed/configuration.

- `checkpoint.pt`: trained model and serialized configuration (locally generated, Git-ignored because it is binary/reproducible).
- `training_history.csv`: per-epoch train/validation loss components.
- `training_summary.json`: final training metrics and parameter count.
- `evaluation_metrics.json`: one-step and rollout metrics, shapes, timing, and parameter counts.
- `loss_curve.png`: train/validation total loss.
- `rollout_comparison.png`: fixed held-out truth and autoregressive decoded rollout.
- `rollout_error.png`: held-out pixel MSE by rollout horizon.

Reproduce from repository root with:

```bash
uv run pytest
uv run python 03_memory/01_gru/train.py
uv run python 03_memory/01_gru/evaluate.py
```
