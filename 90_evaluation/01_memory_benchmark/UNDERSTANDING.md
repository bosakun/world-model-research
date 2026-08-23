# Memory比較を理解する

## なぜ比較が必要か

modelが複雑になると「きっと良くなった」と思いやすい。しかし別dataset、別seed、別lossで得た数値は比較できない。本benchmarkは、条件を揃えて「履歴が必要なときに、どのmemoryが役に立つか」を測る。

## 主指標: Alias Goal Accuracy

paired dataでは、現在画像がbitwise同一でもhidden Goalはright/downの二通りある。現在frameだけしか使わないmodelは同じ入力に対し二つの正解を出せないため、均等なら最大でも0.5である。

```text
current image A == current image B
hidden Goal A != hidden Goal B
```

0.5を超えるには、以前に見たGoalとaction履歴を利用しなければならない。画像MSEよりもmemory仮説へ直接対応した指標である。

## Memory Ablation

- GRU: hiddenをresetする。
- RSSM: deterministicとstochastic stateをresetする。
- Transformer: historyを最後の1 tokenだけにする。

memoryが本当に使われているなら、この操作でaccuracyは落ちる。実際に全modelが0.5へ戻った。

## 結果をどう読むか

- No Memory 0.5: expected。現在画像だけでは区別不能。
- GRU 0.833 ± 0.236: memoryは使えたが、1 seedで学習に失敗した。
- RSSM / Transformer 1.0: この小規模taskでは3 seedすべてでGoalを保持した。
- 画像MSE: Goal memoryとは別の能力。RSSMがGoalを覚えてもh10画像MSEで最良とは限らない。
- latency: RSSMはTransformerより速いが、No Memoryよりは遅い。性能とコストは同時に見る。

## 説明できるようになる確認項目

- なぜ画像MSEだけでmemoryの有効性を判断できないか。
- paired aliasがNo Memoryの上限を0.5にする理由。
- ablationで0.5へ戻ることが何を示すか。
- GRUの平均accuracyだけで採用できない理由。
- RSSMとTransformerの採否が、accuracyだけでなくlatencyとinterfaceにも依存する理由。

## 限界

この結果は「すべての環境でRSSMが最良」という証拠ではない。小さく決定論的な2-Goal環境、3 seed、auxiliary Goal label上での結果である。長いcontext、noise、control success、real-world observationでは再評価が必要である。
