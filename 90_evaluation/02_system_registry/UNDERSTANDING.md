# 評価Registryを理解する

## 解決する問題

実験が増えると結果は各folderに散らばる。後から「このグラフはどのdataset、seed、scriptで作ったか」を追えないと、研究の比較や記事化が危うくなる。

## Registryがすること・しないこと

```text
すること: evaluation_metrics.jsonを見つけ、場所とmetadataを一覧化する。
しないこと: 異なるtaskのmetricを一つの順位に変換する。
```

たとえばoccupancy IoU 0.5とplanning success 0.5は、同じ0.5でも意味がまったく違う。単純比較はできない。

## データフロー

`rglob -> JSON parse -> metadata row -> nested JSON + compact CSV + coverage plot`。

各行にはphase、experiment path、metrics path、dataset version、seed、evaluation entry point、raw payloadが入る。registry自身のoutputは再帰的に読まないよう除外する。

## 説明できるようになる確認項目

- なぜregistryはleaderboardではないか。
- dataset versionとseedを残す意味は何か。
- raw nested metricsを残す理由は何か。
- artifact数がquality scoreではない理由は何か。
- one-seed smoke resultとmulti-seed benchmarkを同等に扱えない理由は何か。

## 次の問い

metricの単位・型をschemaとして検査するか、checkpoint hashをどう追跡するか、OOD/physical評価をどう追加するか。
