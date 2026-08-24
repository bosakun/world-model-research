# Latent Overshooting: 長い未来のlatentを直接学習する

このREADMEは日本語で実験の目的・構造・確認結果を読む入口です。英語の技術原文（全数式、Tensor shape、実行条件、出典）は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) に保存しています。UNDERSTANDING.mdを先に読んでから戻ると、実験記録を追いやすくなります。

## まず読む順番

1. **Purpose / Problem**: 前の方法の何が足りなかったかを読む。
2. **Core Idea / Data Flow**: 入力から出力までの一番短い流れを読む。
3. **Architecture and Training**: どの部品を足し、何を学習させるかを読む。
4. **Evaluation / Limitations**: 何が確認でき、何が未証明かを読む。

> Architectureの専門用語が分からないときは、先にこのフォルダのUNDERSTANDING.mdを読んでください。このREADMEでは「実際にこの実験でどの構成を使ったか」を確認します。



## Purpose / Problem

1step予測だけでは、rolloutで自分の誤差を次の入力にしてしまい、長い未来ほど崩れます。

## Core Idea / Data Flow

現在から複数step想像したlatentを、将来画像から推論したlatentへ近づけます。

## Architectureを一行ずつ読む

```text
現在の画像列 + action列
    ↓ Encoder / posterior
各時刻の「正解に近いlatent」を作る
    ↓
時刻tのlatentからactionをk回使う
    ↓ Dynamics rollout
k step先の予測latentを作る
    ↓
同じ時刻t+kのposterior latentと比較する
```

1. 学習データには画像列とaction列があります。画像からRSSMのposteriorを作ると、「実際にその時刻で観測した世界」に近いlatentを得られます。
2. 次に、時刻`t`のlatentからfuture imageを見ずにactionを`k`回通します。これは本番のimaginationと同じ、未来画像なしのrolloutです。
3. その結果できた予測latentと、実際の時刻`t+k`の画像を見て作ったposterior latentを比べます。
4. 1step先だけでなく2step先、3step先も比べるので、Dynamics Modelは長い未来で壊れにくい方向へ学びます。

例えば「rightを3回」というaction列なら、`t`から3回想像したlatentを、3step後の正解画像から得たlatentへ近づけます。未来画像は答え合わせだけであり、想像の入力には使いません。

## Architecture and Training

1step lossに加えk-stepのKL/latent consistency lossを使います。未来画像は学習時の答え合わせだけで、rolloutの入力には使いません。

### lossをどう読むか

- 1step loss: 次の一歩を当てる能力を学ばせる。
- k-step overshooting loss: 数step先まで想像したlatentが正しい方向へ進むよう学ばせる。
- KL: priorが、観測を見たposteriorに近い未来を予測できるようにする。

overshooting lossを小さくしても、画像の細部やrewardまで正しくなったとは限りません。評価では、1stepのlossだけでなくhorizonごとのrollout errorを見る必要があります。

実行コマンド、random seed、dataset version、parameter数、checkpoint形式、詳細なloss式、Tensor shape、smoke-test数値は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) にそのまま残しています。対応する実装は、このフォルダの model、dataset、losses、train、evaluate、tests です。

## Evaluation

一つの見栄えのよい例だけで判断せず、forward pass、rollout、loss、task固有の指標、failure caseを確認します。outputsフォルダには学習・評価で作った図と数値を保存しています。

この実験では特に、1stepは良いのに5stepや10stepで急に悪くならないかを見ます。長期誤差が改善して初めて、overshootingが役立った可能性があります。

## Limitations

horizonを長くしすぎると計算と不安定さが増えます。

## Final Model Candidate

この段階の結果はsmoke testです。本格比較、複数seed、ablation、他方式との採否判断は90_evaluationで行います。

## References

対応論文、採用した機構、原実装との差分は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) のReferencesを参照してください。ここでの実装は、論文の完全再現ではなく理解のための小規模実装である場合があります。
