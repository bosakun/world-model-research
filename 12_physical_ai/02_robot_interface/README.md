# Robot Interface: 学習modelの出力を安全に実機へ渡す

このREADMEは日本語で実験の目的・構造・確認結果を読む入口です。英語の技術原文（全数式、Tensor shape、実行条件、出典）は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) に保存しています。UNDERSTANDING.mdを先に読んでから戻ると、実験記録を追いやすくなります。

## まず読む順番

1. **Purpose / Problem**: 前の方法の何が足りなかったかを読む。
2. **Core Idea / Data Flow**: 入力から出力までの一番短い流れを読む。
3. **Architecture and Training**: どの部品を足し、何を学習させるかを読む。
4. **Evaluation / Limitations**: 何が確認でき、何が未証明かを読む。

> Architectureの専門用語が分からないときは、先にこのフォルダのUNDERSTANDING.mdを読んでください。このREADMEでは「実際にこの実験でどの構成を使ったか」を確認します。



## Purpose / Problem

学習modelは範囲外や危険なactionを出すことがあり、直接実機へ送れません。

## Core Idea / Data Flow

action requestをvalidation、rate/position/force limit、emergency stopを通して実行します。

## Architectureを一行ずつ読む

    model action request
    -> schema / unit validation
    -> speed, position, force limits
    -> emergency stop確認
    -> robot execution
    -> observation, action, resultをlog

学習modelの出力は提案であり、robot commandではありません。安全guardを通って初めて実行可能なcommandになります。

### 結果をどう読むか

成功回数だけでなく、guardが拒否したaction、遅延、ログ完全性、非常停止の動作を確認します。

## Architecture and Training

command schema、safety guard、observation/action/result log、simulator fallbackを実装します。

実行コマンド、random seed、dataset version、parameter数、checkpoint形式、詳細なloss式、Tensor shape、smoke-test数値は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) にそのまま残しています。対応する実装は、このフォルダの model、dataset、losses、train、evaluate、tests です。

## Evaluation

一つの見栄えのよい例だけで判断せず、forward pass、rollout、loss、task固有の指標、failure caseを確認します。outputsフォルダには学習・評価で作った図と数値を保存しています。

## Limitations

この境界が曖昧だと、安全性も実験の再現性も失われます。

## Final Model Candidate

この段階の結果はsmoke testです。本格比較、複数seed、ablation、他方式との採否判断は90_evaluationで行います。

## References

対応論文、採用した機構、原実装との差分は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) のReferencesを参照してください。ここでの実装は、論文の完全再現ではなく理解のための小規模実装である場合があります。
