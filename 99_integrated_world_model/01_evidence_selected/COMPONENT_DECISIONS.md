# Component Decision Record

この記録は、Phase 99で「全部入れる」のではなく、過去の実験結果から採用・保留・不採用を決めた根拠を固定する。

| Component | Decision | Evidence | Reason / interaction |
|---|---|---|---|
| CNN image encoder | Adopt | Phase 90の全memoryモデルが画像入力で動作 | 小さいGrid画像には十分。将来video/object encoderと置換可能 |
| RSSM memory | Adopt | hidden-Goal accuracy `1.000±0.000`; ablation `0.500`; 4.48 ms | Transformerと同精度で低latency。prior/posteriorがimaginationへ直結 |
| Transformer memory | Hold as alternative | hidden-Goal `1.000`; 8.05 ms | 長いcontextなら有望だが今回の短系列では追加latencyを正当化できない |
| Plain GRU | Do not select | `0.833±0.236`; 1 seed failed | RSSMよりseed安定性が低かった |
| No Memory | Reject | hidden-Goal `0.500` | aliasされた現在観測を原理的に区別できない |
| Gaussian stochastic state | Adopt | RSSM test/training成功 | 観測ありposteriorと観測なしpriorの橋。今回はmean pathなので多峰性は未検証 |
| Prior-head ensemble | Adopt provisionally | Phase 04でOOD disagreement比1.538; integrated std計測可能 | risk score interfaceを作れる。shared backboneなのでepistemic近似は弱い |
| Latent overshooting | Adopt | Phase 05で実装、integrated loss finite | multi-step priorを直接拘束。今回単独ablationなし |
| Reward/value/continuation | Adopt | Phase 06および統合validation | planningに必要。valueはdistance potentialへ変更 |
| Goal auxiliary head | Adopt for experiment | hidden-Goalを直接監査可能 | true-state教師を使うtask-specific補助loss。汎用モデルでは外す候補 |
| Image decoder | Adopt for audit | reconstruction MSE 7.45e-5 | representation可視化用。planner経路には不要 |
| Random-shooting MPC | Adopt | integrated 40/40 success | 離散4-action・H6なら十分。大きなaction空間ではCEM等が必要 |
| Risk penalty | Keep, benefit unproven | risk/mean-onlyとも40/40 | OODやstochastic hazard課題で再評価が必要 |
| Imagined actor-critic | Reject for this integration | Phase 08 final distance 0.513, success false | actorがworld-model誤差を利用。MPCの再計画を優先 |
| Slot Attention | Reject pending repair | mean best IoU 0.271 | object bindingが崩れた表現を統合しない |
| Known-binding C-SWM / SlotFormer | Hold | 機構は動作したがordered slotsを前提 | perceptionが未知objectを安定抽出できるまで統合しない |
| Latent action discovery | Reject pending identifiability work | true-action permutation accuracy 0.270 | dynamics精度だけではaction semanticsを保証しない |
| VQ video tokens / UniSim adapter | Hold | individual smoke tests成功 | 現在の単一画像Grid課題には不要。video/multimodal domainで置換候補 |
| 3D occupancy | Hold | voxel rolloutは動作 | 2D Grid課題にrepresentation mismatch |
| Action-conditioned JEPA | Hold | physical transition probe成功 | physical/video observationへ移る際のencoder代替候補 |
| Robot safety boundary | Adopt concept | Phase 12 guard tests成功 | 統合版では離散action validationのみ。実機前にdead-man/workspace制約が必要 |

## Evidence strength

- Strong within this repository: Phase 90の3-seed memory alias accuracyとmemory ablation。
- Moderate: 各phaseのone-seed smoke testがmechanical correctnessを示す。
- Weak: 統合版のone-seed、二Goal、in-distribution成功率。
- Not established: real-world transfer、risk-aware superiority、uncertainty calibration、汎用object/video/multimodal性能。

## Decision rule

統合対象は、(1) 現在の課題で必要、(2) testとsmoke runが成功、(3) downstream interfaceが明確、の三条件を優先した。失敗した機構は履歴を消さず、修復条件とともに保留または不採用とした。
