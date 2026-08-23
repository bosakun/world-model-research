# 研究ノート: Partial Observation

開始日: 2026-08-22

## 出発点

`01_gru`は履歴を持てるが、全画像にGoalが見えていた。問うべきことは「GRUが動くか」ではなく、「datasetにmemoryだけが利用できる情報があるか」だった。

## 重要な設計と気づき

単にviewを狭めるだけでは不十分なので、Goal-right/Goal-down、同じAgent開始位置、同じ`left,left` prefix、`t=2`で完全に同じ観測というpaired historyを作った。missing informationを直感でなくtest済み性質にした。

- unknownはemptyではない。青い外側は観測されていない、暗い中央セルは観測済みの空間である。
- current frameが同じでも、過去に見たGoalとaction列が異なれば世界の意味は異なる。
- `true_states`と`full_worlds`は評価に必要だが、model入力へ渡すと答えを漏らす。
- sequence datasetがなければ、Goalが消える前のcueをGRUへ渡せない。

## 実装上の問題と結果

`01_gru`と同時にtestすると、standalone folder同士の`env`/`dataset` module名が衝突した。内部実装を`partial_env.py`、`partial_dataset.py`へ分け、従来名のthin wrapperは残した。

- `t=2`のalias pairはbitwise一致し、true Goal座標は異なった。
- 既存Simple/GRUはpartial tensorを受け取れたが、これは性能比較ではない。
- 記事用の図: `full_world.png`、`partial_observation.png`、`observation_sequence.png`、`aliasing_pair.png`。

## 次に比較したいこと

No Memory vs GRUを同じseed/data/parameter accountingで学習し、hidden Goal accuracy、rollout error、history shuffle、hidden resetを比較する。
