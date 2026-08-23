# World Model研究ロードマップ

最終更新: 2026-08-23

状態: `Not Started`（未着手）、`Reading`（読書中）、`Implementing`（実装中）、`Evaluating`（評価中）、`Completed`（完了）、`Needs Re-evaluation`（再評価が必要）。

## このリポジトリの出発点

- **Completed** — 既存リポジトリ監査。詳しくは`REPOSITORY_AUDIT.md`。
- **Needs Re-evaluation** — 以前実装したという基本・画像latentコードは、このcheckoutには見つからなかった。後から回収できた場合は既存実験を上書きせず比較する。

## 01 Basic Dynamics

- **Not Started** — 完全観測の`(state, action) -> next state` baseline。
- **Not Started** — multi-step state rollout。

## 02 Visual Latent

- **Not Started** — image Encoder/Decoderとautoencoder baseline。
- **Not Started** — memoryなしlatent dynamics。
- **Not Started** — latent rollout。

## 03 Memory

- **Completed** — `01_gru`: 完全観測Grid Worldでの決定論的GRU latent dynamics。
- **Completed** — `02_partial_observation`: 同じ現在画像なのに隠れGoalが異なるpaired alias環境。
- **Completed** — `03_rssm`: deterministic/stochastic state、prior/posterior、reparameterization、KL、reconstruction、prior rollout。
- **Completed** — `04_transformer_memory`: latent/action token、position、causal attention、autoregressive rollout、attention可視化。
- **Completed in Phase 90** — No Memory / GRU / RSSM / Transformerの統一比較とmemory ablation。

最初のGRU仮説は「完全観測では現在画像がほぼMarkov stateなので、GRUは動いてもmemoryの優位性を示せない」である。Partial Observationで初めて、その仮説を比較できる情報欠落を作った。

## 04 Uncertainty

- **Completed** — `01_probabilistic_dynamics`: heteroscedastic diagonal Gaussian、NLL、coverage、sampled rollout。
- **Completed** — `02_ensemble`: bootstrap ensemble、epistemic/aleatoricの分解、OOD disagreement、TS1/TS-infinity rollout。

## 05 Long Horizon

- **Completed** — `01_latent_overshooting`: 複数開始点・最大5 stepのrecursive objective、30 step error。
- **Completed** — `02_temporal_abstraction`: 5 action chunkのmacro transition。

## 06 Reward / Value

- **Completed** — `01_prediction_heads`: immediate reward、value、continuation/termination prediction。

## 07 Planning

- **Completed** — `01_random_shooting`: 候補action列をランダム生成して評価。
- **Completed** — `02_cem`: 良い候補に分布を寄せるCross-Entropy Method。
- **Completed** — `03_mpc`: 先頭actionだけ実行して再計画するMPC。
- **Completed** — `04_latent_planning`: Decoderを使わないtask-oriented latent planning。

## 08 Imagination RL

- **Completed** — `01_actor_critic`: imagined trajectory、actor、EMA critic、lambda-return。model exploitationも失敗結果として保存。
- **Completed（資料範囲）** — Dreamer / V2 / V3の違い。完全なDreamer再現ではない。

## 09 Spatial Representation

- **Completed** — `01_cswm`: known-binding slot、relation graph、contrastive energy。
- **Completed（失敗結果）** — `02_slot_attention`: reconstructionは動いたがobject mask IoU 0.271。collapseとして保存。
- **Completed** — `03_slotformer`: ordered slotのtemporal Transformer。
- **Completed** — `04_occupancy_3d`: action-conditioned 8³ voxel rollout。

## 10 Video World Model

- **Completed** — `01_vq_video_tokenizer`: VQ visual token、straight-through training、codebook診断。
- **Completed（semantic alignment失敗）** — `02_latent_action_dynamics`: token dynamicsは動いたがtrue action対応accuracy 0.270。
- **Completed** — `03_heterogeneous_simulator`: motor/language/goal adapterを持つconditional simulator。

## 11 Multimodal

- **Completed** — `01_multimodal_fusion`: vision、proprioception、language、touch token、missingness、future grounding。

## 12 Physical AI

- **Completed** — `01_action_conditioned_jepa`: EMA target、anti-collapse、state probe、action ablation。
- **Completed** — `02_robot_interface`: bounded/dead-man/workspace guard、demonstration replay、offline imitation、simulator評価。

## 90 Evaluation

- **Completed** — `01_memory_benchmark`: memory modelの統一3-seed比較。
- **Completed** — one/5/10-step prediction、hidden Goal、ablation、training stability、parameter、CPU latency。
- **Completed** — `02_system_registry`: dataset version、seed、entry pointを持つ24件の評価artifact一覧。

## 99 Integrated World Model

- **Completed** — `01_evidence_selected`: CNN/RSSM belief、prior ensemble、overshooting、reward/value/continuation/Goal head、risk-aware discrete MPC。
- **Completed** — 部分観測filtering、prior imagination、guarded action、replanningを一つのloopへ接続。
- **Completed** — 採用・保留・不採用の判断理由を`COMPONENT_DECISIONS.md`へ記録。
- **Completed** — 全131 tests成功、24件のcross-phase評価artifact。

## 次に読む順序

初めて読むなら`03_memory/01_gru` → `02_partial_observation` → `03_rssm` → `90_evaluation/01_memory_benchmark` → `99_integrated_world_model`の順がよい。個別のPhase 04〜12は、そのテーマを記事化・深掘りするときに読む。
