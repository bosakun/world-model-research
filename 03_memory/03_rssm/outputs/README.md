# 生成したRSSMの証拠

seed 23、`config.py`の既定設定で2026-08-22に生成した。

| ファイル | 意味 |
|---|---|
| `loss_curve.png` | train/validation objective、weighted reconstruction、raw KL |
| `reconstruction.png` | 正解部分観測とposterior mean reconstruction |
| `latent_rollout.png` | posterior seed後にpriorだけで想像した6 frame |
| `rollout_error.png` | prior rollout horizonごとのplain pixel MSE |
| `training_history.csv` / `training_summary.json` | epoch履歴と再現情報 |
| `evaluation_metrics.json` | reconstruction、state head、rollout、shape、parameter指標 |
| `checkpoint.pt` | local checkpoint。生成binaryのためGit対象外 |

再生成:

```bash
uv run python 03_memory/03_rssm/train.py
uv run python 03_memory/03_rssm/evaluate.py
```
