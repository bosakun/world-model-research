# Imagination Actor-Critic: 想像した未来から方策を学ぶ

このREADMEは日本語で実験の目的・構造・確認結果を読む入口です。英語の技術原文（全数式、Tensor shape、実行条件、出典）は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) に保存しています。UNDERSTANDING.mdを先に読んでから戻ると、実験記録を追いやすくなります。

## まず読む順番

1. **Purpose / Problem**: 前の方法の何が足りなかったかを読む。
2. **Core Idea / Data Flow**: 入力から出力までの一番短い流れを読む。
3. **Architecture and Training**: どの部品を足し、何を学習させるかを読む。
4. **Evaluation / Limitations**: 何が確認でき、何が未証明かを読む。

> Architectureの専門用語が分からないときは、先にこのフォルダのUNDERSTANDING.mdを読んでください。このREADMEでは「実際にこの実験でどの構成を使ったか」を確認します。



## Purpose / Problem

毎actionでplanningを最初から行う代わりに、良いactionを出すActorを学びます。

## Core Idea / Data Flow

Actorがlatentからactionを出し、world modelが次latentとrewardを想像し、Criticが将来価値を評価します。

## Architectureを一行ずつ読む

    実データ -> world model
    latent -> Actor -> action
    latent + action -> imagined next latent
    imagined reward + Critic value -> Actor/Criticを更新

Actorはactionを出す役、Criticは将来の良さを採点する役です。本物の環境でなくmodel内部のtrajectoryを使って二つを学びます。

### 結果をどう読むか

imagined returnだけで判断しません。実環境returnとの差が大きければ、Actorがworld modelの誤りを利用している可能性があります。

## Architecture and Training

imagined trajectoryでActor/Criticを更新し、world modelは実環境データで別途更新します。

実行コマンド、random seed、dataset version、parameter数、checkpoint形式、詳細なloss式、Tensor shape、smoke-test数値は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) にそのまま残しています。対応する実装は、このフォルダの model、dataset、losses、train、evaluate、tests です。

## Evaluation

一つの見栄えのよい例だけで判断せず、forward pass、rollout、loss、task固有の指標、failure caseを確認します。outputsフォルダには学習・評価で作った図と数値を保存しています。

## Limitations

world modelの穴を突くactionをActorが学ぶ可能性があるため、実環境評価が必須です。

## Final Model Candidate

この段階の結果はsmoke testです。本格比較、複数seed、ablation、他方式との採否判断は90_evaluationで行います。

## References

対応論文、採用した機構、原実装との差分は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) のReferencesを参照してください。ここでの実装は、論文の完全再現ではなく理解のための小規模実装である場合があります。
