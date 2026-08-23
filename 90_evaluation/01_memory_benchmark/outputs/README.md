# 生成物

- `benchmark_results.json`: config、全seed、aggregate結果。
- `per_seed_results.csv`: 12 training runの行ごとの結果。
- `memory_comparison.png`: Goal accuracy、horizon error、latencyなどの比較図。
- `checkpoints/`: 再生成可能なlocal checkpoint。Git対象外。

再生成:

```bash
uv run python 90_evaluation/01_memory_benchmark/run_benchmark.py
```
