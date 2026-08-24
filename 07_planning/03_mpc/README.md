# MPC: 観測するたびに計画を立て直す

このREADMEは日本語で実験の目的・構造・確認結果を読む入口です。英語の技術原文（全数式、Tensor shape、実行条件、出典）は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) に保存しています。UNDERSTANDING.mdを先に読んでから戻ると、実験記録を追いやすくなります。

## まず読む順番

1. **Purpose / Problem**: 前の方法の何が足りなかったかを読む。
2. **Core Idea / Data Flow**: 入力から出力までの一番短い流れを読む。
3. **Architecture and Training**: どの部品を足し、何を学習させるかを読む。
4. **Evaluation / Limitations**: 何が確認でき、何が未証明かを読む。

> Architectureの専門用語が分からないときは、先にこのフォルダのUNDERSTANDING.mdを読んでください。このREADMEでは「実際にこの実験でどの構成を使ったか」を確認します。



## Purpose / Problem

一度決めた長いaction列は、途中で予測が外れても修正できません。

## Core Idea / Data Flow

planning後に最初のactionだけ実行し、本物の次観測から再びplanningします。

## Architectureを一行ずつ読む

    現在観測 -> H stepの計画
    -> 最初のactionだけ実行
    -> 本物の次観測を受け取る
    -> 前の残り計画を捨てて再計画

MPCは新しいmodelではなく、world modelを使うcontrol loopです。毎stepで観測を取り直すため、以前の予測誤差を修正できます。

### 結果をどう読むか

open-loop（最初の計画を最後まで実行）と比べ、予測誤差があるときほどMPCが改善するかを見ます。successだけでなく、毎stepのplanning時間も確認します。

## Architecture and Training

random shootingまたはCEMをreceding horizonで繰り返し、open-loopと比較します。

実行コマンド、random seed、dataset version、parameter数、checkpoint形式、詳細なloss式、Tensor shape、smoke-test数値は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) にそのまま残しています。対応する実装は、このフォルダの model、dataset、losses、train、evaluate、tests です。

## Evaluation

一つの見栄えのよい例だけで判断せず、forward pass、rollout、loss、task固有の指標、failure caseを確認します。outputsフォルダには学習・評価で作った図と数値を保存しています。

## Limitations

毎stepの計算が必要で、modelが大きいと実時間制御が難しくなります。

## Final Model Candidate

この段階の結果はsmoke testです。本格比較、複数seed、ablation、他方式との採否判断は90_evaluationで行います。

## References

対応論文、採用した機構、原実装との差分は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) のReferencesを参照してください。ここでの実装は、論文の完全再現ではなく理解のための小規模実装である場合があります。
