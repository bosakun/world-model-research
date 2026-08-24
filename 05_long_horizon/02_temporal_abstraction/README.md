# Temporal Abstraction: 複数stepをまとめて予測する

このREADMEは日本語で実験の目的・構造・確認結果を読む入口です。英語の技術原文（全数式、Tensor shape、実行条件、出典）は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) に保存しています。UNDERSTANDING.mdを先に読んでから戻ると、実験記録を追いやすくなります。

## まず読む順番

1. **Purpose / Problem**: 前の方法の何が足りなかったかを読む。
2. **Core Idea / Data Flow**: 入力から出力までの一番短い流れを読む。
3. **Architecture and Training**: どの部品を足し、何を学習させるかを読む。
4. **Evaluation / Limitations**: 何が確認でき、何が未証明かを読む。

> Architectureの専門用語が分からないときは、先にこのフォルダのUNDERSTANDING.mdを読んでください。このREADMEでは「実際にこの実験でどの構成を使ったか」を確認します。



## Purpose / Problem

primitive actionを一歩ずつつなぐと、計算回数と誤差の蓄積が増えます。

## Core Idea / Data Flow

短いaction列をmacro actionへまとめ、数step後の変化を一つの遷移として予測します。

## Architectureを一行ずつ読む

```text
primitive action列（例: right, right, right, right）
    ↓ macro action encoder
macro action（4step分の行動を表す短いvector）
    ↓ macro dynamics
現在stateから4step後のstateを予測
    ↓
macro reward / termination
4stepの間に得たrewardと終了も予測
```

1. primitive actionは、環境へ実際に送る一歩ごとのactionです。
2. macro action encoderは、複数stepのaction列を一つの短い表現へまとめます。これは「右を4回」全体を一つの行動単位として見る部品です。
3. macro dynamicsは、毎stepを経由せず、現在stateから数step後のstateを直接予測します。
4. rewardとterminationもmacro単位で扱います。途中でepisodeが終わるなら、その情報を失わないためです。

この仕組みは4stepを魔法のように正確にするものではありません。予測をつなぐ回数を減らし、長い変化を扱いやすくする方法です。

## Architecture and Training

macro action encoder、macro dynamics、macro reward/terminationを学習し、primitiveとmacroのrolloutを比較します。

### lossをどう読むか

- macro state loss: 数step後stateを当てる。
- macro reward loss: そのまとまりの行動で得るrewardを当てる。
- termination loss: 途中または最後でepisodeが終わるかを当てる。

stateだけを当てても、rewardや終了を外せばplanningには使えません。三つを分けて見る理由はここにあります。

実行コマンド、random seed、dataset version、parameter数、checkpoint形式、詳細なloss式、Tensor shape、smoke-test数値は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) にそのまま残しています。対応する実装は、このフォルダの model、dataset、losses、train、evaluate、tests です。

## Evaluation

一つの見栄えのよい例だけで判断せず、forward pass、rollout、loss、task固有の指標、failure caseを確認します。outputsフォルダには学習・評価で作った図と数値を保存しています。

primitive rolloutより速く長い未来を扱えるか、そして途中の重要な変化を飛ばして失敗していないかを両方確認します。

## Limitations

途中の細かい接触や障害物回避は見落としやすく、macro長の選び方が重要です。

## Final Model Candidate

この段階の結果はsmoke testです。本格比較、複数seed、ablation、他方式との採否判断は90_evaluationで行います。

## References

対応論文、採用した機構、原実装との差分は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) のReferencesを参照してください。ここでの実装は、論文の完全再現ではなく理解のための小規模実装である場合があります。
