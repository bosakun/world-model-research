# 生成物

- `experiment_registry.json`: 全raw metricsとmetadataを保つ台帳。
- `experiment_registry.csv`: 検索しやすいcompact index。
- `evaluation_coverage.png`: phaseごとの評価artifact数。

再生成:

```bash
uv run python 90_evaluation/02_system_registry/build_registry.py
```
