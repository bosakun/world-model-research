# 統一Partial Observation Memory Benchmark

状態: 完了（2026-08-23）。このリポジトリのために設計した比較実験であり、論文benchmarkそのものではない。

## 先に読むところ

このページで一番大切なのは表の小数ではない。次の問いである。

> 同じ現在画像なのに、過去が違うと正しい答えも違うとき、modelは過去を使えているか？

No Memoryが50%、memoryを外したGRU/RSSM/Transformerも50%へ落ちたことをまず確認する。これが「memoryが実際に使われた」証拠である。parameter数やlatencyは、その後で読む。

## 目的

No Memory、GRU、continuous RSSM、causal Transformerを、同じpartial-observation dataset、loss、context、rollout、seed、評価方法で比較する。以前の個別実験はEncoderや学習stepが違い、その数値を並べても公平ではなかった。

## 問題と仮説

pixel errorだけを見ると、視界外のGoalを忘れたmemoryなしmodelでも、よくある局所画像を予測して低errorを得られる。そこで「同じ現在画像なのに、過去に見たGoalが違う」paired alias上のGoal accuracyを主指標にする。

仮説は、No Memoryはchanceの0.5に留まり、memory modelは0.5を超える。さらにmemoryをresetすると利点は消える。小さい学習budgetではRSSM/TransformerがGRUより安定する可能性がある。

## 比較するmodel

```text
同じpartial images / actions / Goal labels
  -> No Memory: MLP(z_t,a_t)
  -> GRU: GRUCell(z_t,a_t,h_t)
  -> RSSM: h_t + Gaussian prior/posterior z_t
  -> Transformer: causal history tokens (z_t,a_t)

context t=0..2 -> future t=3..12を観測なしrollout
```

Goal分類は各modelが本当に持つmemory stateから読む。GRUはhidden、RSSMは`h+z`、Transformerはcontext token、No Memoryは予測latentを使う。Goal座標やfull stateを観測入力へ渡さない。

## Tensor Shapesとloss

| Tensor | Shape |
|---|---:|
| images | `[B,13,3,20,20]` |
| actions | `[B,12,4]` |
| predicted future images | `[B,10,3,20,20]` |
| Goal logits | `[B,10,2]` |
| alias mask | `[B,10]` |

latent dimensionは16、GRU/RSSM deterministic/Transformer model dimensionは64。

```text
L = 5*MSE(next_image) + MSE(reconstruction)
  + MSE(next_latent, stopgrad(latent_target))
  + CE(goal_logits, goal) + 0.05*KL_RSSM
```

reconstructionは画像情報をlatentへ残す。next-image/latent lossはdynamicsを学習する。Goal CEは「隠れGoalを覚えているか」を測れる形にする。RSSMのKLだけはprior/posteriorを近づけるために必要である。

## 学習・評価

seedは301/302/303、train/testは128/64 paired sequence、batch 32、35 epoch/140 steps、Adam `1e-3`。各modelでlossの重み、data、step数を揃えた。

```bash
uv run python 90_evaluation/01_memory_benchmark/run_benchmark.py
```

各seedの結果、parameter bytes、batch 16のCPU latency、horizon 1/5/10 error、memory ablationを保存する。

## 結果

| Model | Alias Goal accuracy | memoryを外した後 | h10 image MSE | Params | Batch-16 latency |
|---|---:|---:|---:|---:|---:|
| No Memory | 0.500 ± 0.000 | 0.500 | 0.001147 | 336,994 | 1.95 ms |
| GRU | 0.833 ± 0.236 | 0.500 | 0.001170 | 356,418 | 4.55 ms |
| RSSM | 1.000 ± 0.000 | 0.500 | 0.001239 | 397,202 | 4.48 ms |
| Transformer | 1.000 ± 0.000 | 0.500 | 0.001158 | 405,186 | 8.05 ms |

RSSMとTransformerは3 seedすべてでhidden Goalを区別した。memoryをresetまたはhistoryを1 tokenへ制限すると全memory modelが0.5へ戻った。これは、この特別に作ったalias課題では履歴が必要だったことを示す。

## 失敗例・知見・限界

- GRUは1 seedで完全に失敗した。平均だけでは学習の不安定性を隠す。
- pixel MSEはmethod差をほとんど分けず、ときにNo Memoryが良く見える。画像errorだけではmemoryを判定できない。
- RSSMはGoal memoryが完全でもh10 image MSEは最も悪い。すべての指標で一番よいmodelはない。
- Transformerは短い系列でもrecurrent modelより約1.8倍遅い。
- tiny deterministic two-Goal POMDP、auxiliary Goal supervision、CPU timing、3 seedだけなので一般性能は結論できない。

Candidate: **統合にはRSSM、Transformerは代替候補として保留**。両者はGoal memoryで安定したが、RSSMは約44%低latencyでprior/posteriorを持ち、uncertainty/imaginationへ接続しやすい。

## 次の問い・参考文献

matched parameter budget、より多いseed、Goal supervisionなしのprobe、aliasの長さ、noise/stochasticity、calibration、planning success、peak memoryを比較する。

参考: PlaNet (https://arxiv.org/abs/1811.04551)、World Models (https://arxiv.org/abs/1803.10122)、TransDreamer (https://arxiv.org/abs/2202.09481)、GRU (https://aclanthology.org/D14-1179/)。model機構だけを参考にし、benchmark設計と結果は独自である。
