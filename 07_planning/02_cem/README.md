# CEM Planning: 良いaction候補の周りを探す

このREADMEは日本語で実験の目的・構造・確認結果を読む入口です。英語の技術原文（全数式、Tensor shape、実行条件、出典）は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) に保存しています。UNDERSTANDING.mdを先に読んでから戻ると、実験記録を追いやすくなります。

## まず読む順番

1. **Purpose / Problem**: 前の方法の何が足りなかったかを読む。
2. **Core Idea / Data Flow**: 入力から出力までの一番短い流れを読む。
3. **Architecture and Training**: どの部品を足し、何を学習させるかを読む。
4. **Evaluation / Limitations**: 何が確認でき、何が未証明かを読む。

> Architectureの専門用語が分からないときは、先にこのフォルダのUNDERSTANDING.mdを読んでください。このREADMEでは「実際にこの実験でどの構成を使ったか」を確認します。



## Purpose / Problem

Random Shootingは良い候補を見つけても、その周辺を次に重点探索できません。

## Core Idea / Data Flow

高得点のelite action列の平均と分散から、次のsampling分布を更新します。

## Architectureを一行ずつ読む

    action分布 -> candidate sample -> world modelで採点
    -> 上位eliteを選ぶ -> eliteの平均・分散で分布更新
    -> 更新分布からもう一度sample

1. 最初は広い分布からaction列を作ります。
2. score上位のeliteだけを残します。
3. eliteの近くを次に多く試せるよう、分布を更新します。
4. 数回後、最良列の先頭actionだけを実行します。

### 結果をどう読むか

candidate数だけでなく、iteration数、elite数、分散の縮み方を見ます。分散が早く縮みすぎると、良い別候補を探す前に探索が止まります。

## Architecture and Training

sample -> elite選択 -> 分布更新を数回繰り返し、最後に最良列の先頭actionを実行します。

実行コマンド、random seed、dataset version、parameter数、checkpoint形式、詳細なloss式、Tensor shape、smoke-test数値は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) にそのまま残しています。対応する実装は、このフォルダの model、dataset、losses、train、evaluate、tests です。

## Evaluation

一つの見栄えのよい例だけで判断せず、forward pass、rollout、loss、task固有の指標、failure caseを確認します。outputsフォルダには学習・評価で作った図と数値を保存しています。

## Limitations

modelの誤りを高得点として最適化する危険があり、elite数や分散下限が結果を左右します。

## Final Model Candidate

この段階の結果はsmoke testです。本格比較、複数seed、ablation、他方式との採否判断は90_evaluationで行います。

## References

対応論文、採用した機構、原実装との差分は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) のReferencesを参照してください。ここでの実装は、論文の完全再現ではなく理解のための小規模実装である場合があります。
