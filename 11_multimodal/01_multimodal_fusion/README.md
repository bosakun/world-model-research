# Multimodal Fusion: 複数感覚から世界を推測する

このREADMEは日本語で実験の目的・構造・確認結果を読む入口です。英語の技術原文（全数式、Tensor shape、実行条件、出典）は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) に保存しています。UNDERSTANDING.mdを先に読んでから戻ると、実験記録を追いやすくなります。

## まず読む順番

1. **Purpose / Problem**: 前の方法の何が足りなかったかを読む。
2. **Core Idea / Data Flow**: 入力から出力までの一番短い流れを読む。
3. **Architecture and Training**: どの部品を足し、何を学習させるかを読む。
4. **Evaluation / Limitations**: 何が確認でき、何が未証明かを読む。

> Architectureの専門用語が分からないときは、先にこのフォルダのUNDERSTANDING.mdを読んでください。このREADMEでは「実際にこの実験でどの構成を使ったか」を確認します。



## Purpose / Problem

画像だけでは接触や内部状態が分からず、個別sensorだけでは全体像が見えません。

## Core Idea / Data Flow

vision、proprioception、touchなどを各encoderで表現し、欠損maskも含めてlatentへ統合します。

## Architecture and Training

modality encoder、fusion、missing-modality mask、prediction headを学習・評価します。

実行コマンド、random seed、dataset version、parameter数、checkpoint形式、詳細なloss式、Tensor shape、smoke-test数値は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) にそのまま残しています。対応する実装は、このフォルダの model、dataset、losses、train、evaluate、tests です。

## Evaluation

一つの見栄えのよい例だけで判断せず、forward pass、rollout、loss、task固有の指標、failure caseを確認します。outputsフォルダには学習・評価で作った図と数値を保存しています。

## Limitations

一つのsensorへ依存したり、欠損をゼロと誤解したりする失敗を確認する必要があります。

## Final Model Candidate

この段階の結果はsmoke testです。本格比較、複数seed、ablation、他方式との採否判断は90_evaluationで行います。

## References

対応論文、採用した機構、原実装との差分は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) のReferencesを参照してください。ここでの実装は、論文の完全再現ではなく理解のための小規模実装である場合があります。
