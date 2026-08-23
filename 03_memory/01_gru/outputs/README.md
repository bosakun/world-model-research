# 生成物

既定seed/configurationで2026-08-22に生成した出力である。

- `checkpoint.pt`: 学習済みmodelとconfig。再生成可能なbinaryのためGit対象外。
- `training_history.csv` / `training_summary.json`: epochごとのlossと最終指標。
- `evaluation_metrics.json`: one-step・rollout指標、shape、時間、parameter数。
- `loss_curve.png`: train/validation loss。
- `rollout_comparison.png`: 正解系列とautoregressive decoded rollout。
- `rollout_error.png`: horizonごとのpixel MSE。

再生成:

```bash
uv run pytest -q 03_memory/01_gru/tests
uv run python 03_memory/01_gru/train.py
uv run python 03_memory/01_gru/evaluate.py
```
