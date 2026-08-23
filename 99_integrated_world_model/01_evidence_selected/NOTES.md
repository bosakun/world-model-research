# Research Notes

## 実装前の疑問

- Phase 90でhidden Goalを読めたことは、実際にGoalへ行動できることを意味するのか。
- RSSM、ensemble、task heads、plannerを単純に接続するだけで動くのか。
- uncertainty penaltyはこの小さい決定論的環境で意味を持つのか。
- 過去の全機構を統合すべきか、それとも失敗したものを明示的に外すべきか。

## 実装前の予想

- RSSMとTransformerは同じhidden-Goal精度だったため、短い系列では低latencyなRSSMが適切と予想した。
- actorはPhase 08でmodel exploitationを示したので、観測で毎step再補正できるMPCの方が安全と予想した。
- 3-head ensembleは不確実性interfaceとして機能するが、in-distributionではrisk-awareとmean-onlyの差は出にくいと予想した。

## 最初に誤解していたこと

「posterior featureでtask headが高精度なら、同じlatent modelのprior rolloutでも使える」と考えかけた。しかしposteriorは未来画像を含み、priorは含まない。headの重みが同じでも入力分布が違うため、これは保証されない。

また、behavior datasetのMonte Carlo returnをvalue targetにすればplanningに使えると考えたが、それは収集方策の将来を表す。候補action列のterminal stateを比較する用途には不安定だった。

## 実装して理解できたこと

- 統合の難しさはclassを接続することではなく、学習時と利用時のstate semanticsを揃えることにある。
- memory probeの1.0 accuracyは必要な証拠だが、closed-loop successは別に検証しなければならない。
- decoder-free planningでも、decoder lossをrepresentation auditとして残せる。
- 不確実性を計算できることと、それが意思決定を改善することは別である。

## エラーと原因

### 初回planning success 0.5

症状:

- risk-aware: 20/40
- mean-only: 20/40
- 右Goalは成功、下Goalでも右へ進み境界で停止

原因:

1. reward/value/goal headはposterior `[h,z^q]`だけで教師あり学習していた。
2. plannerはprior `[h,z^p]`だけを使うため、task headが未学習領域へ入った。
3. value targetはbehavior-policy returnで、terminal stateのGoal距離比較に適さなかった。

修正:

- 三つのprior mean featureすべてにもreward/value/continuation/goal lossを与えた。
- value targetを`-ManhattanDistance/4`へ変更した。
- evaluationのlist-of-array変換を`np.stack`へ変更しwarningを除いた。

結果:

- risk-aware: 40/40
- mean-only: 40/40
- 平均3 steps

## 面白かった挙動

- 同じ位置 `(2,0)`、同じ現在の部分観測から、historyに応じて右へ3回、または右・下・右を選択した。
- ensemble return stdは平均0.44966とゼロではなかったが、penaltyの有無で成功率は変わらなかった。
- KLはfree-nats境界付近の0.5007へ収束した。これはKLが「ゼロへ収束すべきloss」ではないことを図で説明しやすい。

## 予想外の挙動

- prior supervision追加後、goal lossだけでなくreward/valueも急速に安定した。
- overshooting lossは中盤で一度増え、その後低下した。one-step task fittingとmulti-step consistencyが同時には進まない様子が見える。
- 簡単な課題ではrisk-awareの有効性を示せず、むしろ「uncertainty機構を入れただけでは採用根拠にならない」という結果になった。

## 実験中に生じた疑問

- shared backboneの3 headは、どの程度までepistemic uncertaintyと呼べるか。
- prior task supervisionが強すぎると、posterior表現がtask labelへ偏らないか。
- goal auxiliary lossなしでもreconstruction/KLだけで同じplanning successになるか。
- distance potentialはhand-shaped knowledgeなので、将来はTD valueと比較すべき。

## 記事に使えそうな図

- `outputs/loss_curve.png`: total loss、KL free-nats、overshootingの時間変化。
- `outputs/integrated_rollout.png`: 同一の現在観測からhistoryにより右/下へ分岐する経路。
- Phase 90の`memory_comparison.png`: No Memory/GRU/RSSM/Transformerの採用根拠。
- 初回0.5と修正後1.0を示すBefore/After表（初回図は保存せず数値を記録）。

## 記事に使えそうな説明

- 「posteriorで優秀なheadがprior imaginationで優秀とは限らない」
- 「部品の単体テストでは見つからず、closed loopで初めて見つかるinterface bug」
- 「40/40対40/40なのでrisk-aware superiorityは主張しない」
- 「失敗したSlot Attentionやactorを全部入りモデルへ採用しなかった理由」

## 後で比較したい点

- prior task supervisionあり/なし
- goal auxiliary lossあり/なし
- mean-only/risk-awareをOOD Goal・障害物・transition noiseで比較
- shared heads/独立bootstrap ensemble
- RSSM/Transformerを同一integrated plannerへ接続
- mean rollout/particle rollout
- distance potential/Monte Carlo return/TD value

## Reproducibility record

- date: 2026-08-23
- seed: 331
- dataset: `integrated-partial-navigation-v1`
- epochs: 60
- steps: 1,440
- optimizer: Adam, learning rate `8e-4`
- parameters: 456,037
- test result: 4 passed
- evaluation episodes: 40 per variant
- checkpoint: local `outputs/checkpoint.pt`, Git ignored
- external or physical actions: none
