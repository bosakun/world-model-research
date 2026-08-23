# 生成したTransformer Memoryの証拠

seed 29、`config.py`既定設定で2026-08-22に生成した。

| ファイル | 意味 |
|---|---|
| `loss_curve.png` | train/validation objectiveとfuture prediction項 |
| `one_step_prediction.png` | 正解次画像とteacher-forced causal prediction |
| `sequence_rollout.png` | 正解系列とpredicted-latent feedback rollout |
| `attention_map.png` | 最終layer、head平均のcausal attention |
| `rollout_error.png` | autoregressive horizonごとのplain pixel MSE |
| `training_history.csv` / `training_summary.json` | epoch履歴と再現情報 |
| `evaluation_metrics.json` | one-step、rollout、state head、shape指標 |
| `checkpoint.pt` | local checkpoint。生成binaryのためGit対象外 |

再生成:

```bash
.venv/bin/python 03_memory/04_transformer_memory/train.py
.venv/bin/python 03_memory/04_transformer_memory/evaluate.py
```
