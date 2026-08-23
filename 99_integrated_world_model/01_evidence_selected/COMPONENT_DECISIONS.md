# 統合部品の採否記録

Phase 99では「作ったものを全部足す」のではなく、過去の比較結果、失敗例、現在の課題への必要性から採否を決めた。この表は、後から「なぜ入れた・入れなかったのか」を追えるようにする記録である。

| 部品 | 判断 | 根拠 | 理由・他部品との関係 |
|---|---|---|---|
| CNN image encoder | 採用 | Phase 90の全memory modelで画像入力が動作 | 小さなGrid画像には十分。後でvideo/object encoderと置換可能 |
| RSSM memory | 採用 | hidden-Goal accuracy `1.000±0.000`、ablation `0.500`、4.48 ms | Transformerと同精度で低latency。prior/posteriorがimaginationへ直結 |
| Transformer memory | 代替候補として保留 | hidden-Goal `1.000`、8.05 ms | 長いcontextでは有望だが、短系列では追加latencyを正当化できない |
| Plain GRU | 今回は不採用 | `0.833±0.236`、1 seed失敗 | RSSMよりseed安定性が低かった |
| No Memory | 不採用 | hidden-Goal `0.500` | 同じ現在観測を原理的に区別できない |
| Gaussian stochastic state | 採用 | RSSM test/training成功 | 観測ありposteriorと観測なしpriorをつなぐ。多峰性は未検証 |
| Prior-head ensemble | 条件付き採用 | Phase 04のOOD disagreement比1.538、統合でstd計測可能 | risk scoreの入口になる。ただしbackbone共有なのでepistemic近似は弱い |
| Latent overshooting | 採用 | Phase 05で実装、統合lossがfinite | multi-step prior driftを直接抑える。単独ablationは未実施 |
| Reward/value/continuation | 採用 | Phase 06と統合validation | planningに必要。valueはdistance potentialへ変更 |
| Goal auxiliary head | この実験では採用 | hidden-Goalを直接監査できる | true-state教師を使うtask-specific補助loss。汎用modelでは外す候補 |
| Image decoder | 監査用に採用 | reconstruction MSE `7.45e-5` | 表現を画像として確認できる。planner本体は使わない |
| Random-shooting MPC | 採用 | 統合で40/40 success | 離散4-action・horizon 6には十分。大きなaction空間ではCEM等が必要 |
| Risk penalty | 保持するが優位性は未証明 | risk/mean-onlyとも40/40 | OODやstochastic hazardで再評価が必要 |
| Imagined actor-critic | 今回の統合では不採用 | Phase 08 success false | actorがworld-model誤差を利用。MPCの再計画を優先 |
| Slot Attention | 修復まで不採用 | mean best IoU 0.271 | object bindingが崩れた表現を統合しない |
| C-SWM / SlotFormer | 保留 | 機構は動作したがordered slot前提 | unknown objectを安定抽出できるまで統合しない |
| Latent action discovery | 修復まで不採用 | true-action alignment 0.270 | future predictionだけではactionの意味を保証しない |
| VQ video tokens / UniSim adapter | 保留 | 個別smoke test成功 | 現在の単一画像Grid課題には不要。video/multimodalで候補 |
| 3D occupancy | 保留 | voxel rolloutが動作 | 2D Grid課題と表現が合わない |
| Action-conditioned JEPA | 保留 | physical transition probe成功 | physical/video observationへ移るときのEncoder候補 |
| Robot safety boundary | 考え方を採用 | Phase 12のguard test成功 | 統合版は離散action validationだけ。実機前にdead-man/workspace制約が必要 |

## 根拠の強さ

- **このリポジトリ内で強い**: Phase 90の3-seed hidden Goal accuracyとmemory ablation。
- **中程度**: 各phaseのone-seed smoke test。機構が動くことは示すが、一般的な優位性は示さない。
- **弱い**: 統合版のone-seed、二つのGoal、in-distribution success。
- **未確立**: real-world transfer、risk-awareの優位性、uncertainty calibration、汎用object/video/multimodal性能。

## 判断のルール

統合する部品は、次の三条件を優先した。

1. 現在の課題で必要である。
2. testとsmoke runが成功している。
3. 次の部品へ渡すinput/outputが明確である。

失敗した機構も削除せず、修復条件とともに保留または不採用として残す。これは「失敗を隠す」のではなく、次に同じ理由で安易に採用しないためである。
