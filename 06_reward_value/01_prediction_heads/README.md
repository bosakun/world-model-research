# Reward / Value / Continuation: 未来の良さを予測する

このREADMEは日本語で実験の目的・構造・確認結果を読む入口です。英語の技術原文（全数式、Tensor shape、実行条件、出典）は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) に保存しています。UNDERSTANDING.mdを先に読んでから戻ると、実験記録を追いやすくなります。

## まず読む順番

1. **Purpose / Problem**: 前の方法の何が足りなかったかを読む。
2. **Core Idea / Data Flow**: 入力から出力までの一番短い流れを読む。
3. **Architecture and Training**: どの部品を足し、何を学習させるかを読む。
4. **Evaluation / Limitations**: 何が確認でき、何が未証明かを読む。

> Architectureの専門用語が分からないときは、先にこのフォルダのUNDERSTANDING.mdを読んでください。このREADMEでは「実際にこの実験でどの構成を使ったか」を確認します。



## Purpose / Problem

画像やstateの未来だけでは、Agentはどのactionが良いか選べません。

## Core Idea / Data Flow

同じlatent stateからreward、value、continuationの三つを予測します。

## Architectureを一行ずつ読む

```text
observation + 過去のmemory
    ↓ world model
latent state
    ├-> reward head: この一歩の得点
    ├-> value head: この先の得点の合計
    └-> continuation head: 次stepも続く確率
```

1. world modelが、現在観測と過去の情報からlatent stateを作ります。
2. reward headは「今の遷移で何点得るか」を出します。短期の評価です。
3. value headは「ここから先も含めると何点になりそうか」を出します。遠回りでも後でGoalへ着くactionを評価できます。
4. continuation headは「episodeが終わらず続く確率」を出します。終了後の存在しないrewardを数えないために必要です。

三つのheadは同じlatentを見るため、world modelは見た目だけでなく、行動の良さと終了に必要な情報もlatentへ残す必要があります。

## Architecture and Training

rewardは一歩の得点、valueは将来を含む得点、continuationはepisodeが続く確率です。各headに対応するsupervised lossを使います。

### lossと結果をどう読むか

- reward lossが小さい: 直近の得点を当てられる。
- value lossが小さい: 将来を含むreturnを見積もれる。
- continuation lossが小さい: 終了時刻を扱える。

一つだけ良くても十分ではありません。例えばrewardが当たってもvalueが外れれば、長い計画は誤ります。

実行コマンド、random seed、dataset version、parameter数、checkpoint形式、詳細なloss式、Tensor shape、smoke-test数値は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) にそのまま残しています。対応する実装は、このフォルダの model、dataset、losses、train、evaluate、tests です。

## Evaluation

一つの見栄えのよい例だけで判断せず、forward pass、rollout、loss、task固有の指標、failure caseを確認します。outputsフォルダには学習・評価で作った図と数値を保存しています。

## Limitations

世界モデルが正しくてもreward定義が悪ければ望む行動にはなりません。

## Final Model Candidate

この段階の結果はsmoke testです。本格比較、複数seed、ablation、他方式との採否判断は90_evaluationで行います。

## References

対応論文、採用した機構、原実装との差分は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) のReferencesを参照してください。ここでの実装は、論文の完全再現ではなく理解のための小規模実装である場合があります。
