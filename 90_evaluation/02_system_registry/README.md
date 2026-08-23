# フェーズ横断の評価Registry

状態: 完了（2026-08-23）。これは順位表ではなく、実験結果を見失わないための台帳である。

## 目的

各experimentが出した評価artifactを、dataset version、seed、実行入口、raw metricsとともに一覧化する。segmentation IoU、planning success、robot success、pixel MSEのように意味も単位も違う値を、無理に一つの点数へまとめない。

## 仕組み

```text
各experimentの outputs/evaluation_metrics.json
  -> metadataを読む
  -> 元のpayloadを保ったJSON registry
  -> 見やすいCSVとphase別coverage図
```

model tensorやtraining lossは扱わない。registryの1行は「評価を実行できるexperiment」であり、phaseの件数はquality scoreではない。

## コード・実行・結果

```bash
uv run python 90_evaluation/02_system_registry/build_registry.py
```

- discovery / metadata extraction: `build_registry.py::discover`
- JSON/CSV/plot生成: `build_registry.py::build`
- tests: required artifactと必須metadataを確認。

24件を登録した。内訳はMemory 3、Uncertainty 2、Long Horizon 2、Reward/Value 1、Planning 4、Imagination 1、Spatial 4、Video 3、Multimodal 1、Physical AI 2、Integrated 1である。

## 限界と使い方

artifactがあることは正しさ・優位性の証明ではない。多くはone-seed smoke resultで、matched multi-seed evidenceはMemory benchmarkだけである。checkpointはGit対象外で再生成が必要。parameter bytesはpeak memoryではない。

この台帳は「どの結果を、どの条件で、どのscriptから再現するか」を辿るために使う。異なるtaskの数値をランキングしない。
