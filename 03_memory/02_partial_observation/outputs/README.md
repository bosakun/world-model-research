# 生成物

partial observation環境とpaired aliasを確認する可視化である。

- `full_world.png`: simulator内部の世界全体。
- `partial_observation.png`: modelへ渡すAgent中心3x3 view。
- `observation_sequence.png`: actionに伴う局所観測の時系列。
- `aliasing_pair.png`: `t=2`で同一の局所画像だが異なるhidden Goalを持つpair。

再生成:

```bash
uv run pytest -q 03_memory/01_gru/tests 03_memory/02_partial_observation/tests
uv run python 03_memory/02_partial_observation/visualize.py
```
