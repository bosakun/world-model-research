# Evidence-selected Integrated World Model

## Purpose

Phase 03–12で個別に実装した機構を、Phase 90の比較結果と各実験の失敗例に基づいて選択し、部分観測画像から安全な離散行動までを一つの学習・推論経路に接続する。

これは「あらゆる機構を入れた最大モデル」ではない。今回のGrid World課題に必要で、実験上の根拠がある部品だけを採用した `Simplified educational implementation` である。採否の全記録は `COMPONENT_DECISIONS.md` に残す。

## Problem

個別のencoder、memory、dynamics、uncertainty、reward/value、plannerが動いても、それらの学習時の表現と利用時の表現が一致しなければ制御系として機能しない。特に学習時だけ観測を使うposterior featureへtask headを学習し、計画時のprior featureへそのまま適用するとdistribution mismatchが起こる。

統合課題では、初めに見えたGoalが2回の左移動後に視界外へ消える。現在画像は同一でもGoalは右または下にあり、memoryなしでは適切な行動を区別できない。

## Previous Model

Phase 90ではRSSMとTransformerがhidden-Goal accuracy `1.000 ± 0.000`、memory ablation後は双方`0.500`だった。RSSMはTransformerよりCPU latencyが小さかった（batch 16で4.48 ms対8.05 ms）ため、統合基盤にはRSSMを選んだ。

一方、各前段実験は単体だった。

- RSSMには制御用task headと不確実性-aware planningがなかった。
- ensembleはベクトル状態の確率的予測実験であり、部分観測画像RSSMと未統合だった。
- reward/value/continuation headは世界モデルrolloutと未接続だった。
- planningは別の連続Point World上で評価されていた。
- imagination actorはモデル誤差を利用し、実環境Goalへ到達しなかった。

## Hypothesis

部分観測履歴をRSSM stateへfilterし、prior ensemble上のreward/valueをrisk-aware MPCで評価すれば、Goalが視界外になった後でも履歴に応じた行動を選べるはずである。

## Architecture

```text
partial image o_t ── CNN encoder ── e_t
                                    │
previous z + action ── GRUCell ── h_t
                                    │
                  ┌─────────────────┴──────────────┐
                  │                                │
          posterior q(z_t|h_t,e_t)       ensemble prior p_k(z_t|h_t)
                  │                                │
               [h_t,z_t]                   imagined [h_t,z_t]
                  │                                │
       decoder + task heads         reward/value + disagreement
                                                   │
                                  random-shooting risk-aware MPC
                                                   │
                                      discrete action guard
                                                   │
                                             Grid World
```

統合した機構はCNN perception、連続Gaussian RSSM、3本のprior head、3-step latent overshooting、reward/value/continuation/goal head、離散random-shooting MPC、安全境界である。

## Data Flow

学習時:

```text
observation sequence + action sequence
  -> posterior filtering
  -> image reconstruction
  -> posterior/prior task prediction
  -> KL + overshooting + task losses
```

制御時:

```text
visible Goal history
  -> posterior filteringでbelief stateを作る
  -> 以後の候補action sequenceは各priorだけでrollout
  -> ensemble return mean - 0.5 * ensemble std
  -> best sequenceの先頭actionだけ実行
  -> 新観測でposterior stateを更新
  -> 再計画
```

外部機器へのaction送信は行わず、ローカルGrid World内だけで評価した。

## Tensor Shapes

`B`: batch、`T=12`: transition数、`E=3`: ensemble数、`H=64`、`Z=16`。

| Tensor | Shape | Meaning |
|---|---:|---|
| observations | `[B, T+1, 3, 20, 20]` | 部分観測画像 |
| actions | `[B, T, 4]` | one-hot離散action |
| true_states | `[B, T+1, 4]` | evaluation/教師用 `(agent_r, agent_c, goal_r, goal_c)` |
| embedding | `[B, T+1, 64]` | CNN観測埋め込み |
| deterministic state `h` | `[B, T+1, 64]` | recurrent history |
| posterior state `z` | `[B, T+1, 16]` | 観測で補正した状態 |
| prior mean/std | `[E, B, T+1, 16]` | 観測なし予測とensemble差 |
| feature `[h,z]` | `[B, T+1, 80]` | decoder/task head入力 |
| reward/value/continuation | `[B, T]` | 各transitionのtask target |
| goal logits | `[B, T, 2]` | hidden Goal identity |
| planner candidates | `[512, 6, 4]` | horizon 6の候補action列 |
| ensemble returns | `[3, 512]` | 候補ごとの予測return |

## Mathematics

Deterministic transition:

\[
h_t = \operatorname{GRU}(h_{t-1}, [z_{t-1}, a_{t-1}])
\]

Posteriorとensemble prior:

\[
q(z_t\mid h_t,e_t)=\mathcal N(\mu^q_t,(\sigma^q_t)^2),\qquad
p_k(z_t\mid h_t)=\mathcal N(\mu^p_{k,t},(\sigma^p_{k,t})^2)
\]

今回のsmall-scale planningではsampling noiseを避け、filteringとimaginationのstateにはmeanを使う。分布のstdはKL学習には保持する。

学習目的:

\[
\mathcal L = 3L_{recon}+2L_r+L_V+0.5L_c+0.5L_g+0.05L_{KL}+0.1L_{over}
\]

task lossはposterior featureとprior featureの平均である。これにより、計画時に実際に使うprior状態にもreward/valueを直接学習させる。

Risk-aware score:

\[
J(a_{t:t+H})=
\frac{1}{E}\sum_k R_k
-\beta\operatorname{Std}_k[R_k],\qquad \beta=0.5
\]

\[
R_k=\sum_{j=0}^{H-1}\gamma^j\hat r(s^k_{t+j})
+\gamma^H\hat V(s^k_{t+H})
\]

## Code Mapping

| Concept | File | Class/function |
|---|---|---|
| partial navigation sequences | `dataset.py` | `IntegratedNavigationDataset` |
| observation encoder | `model.py` | `IntegratedWorldModel.encoder`, `encode` |
| deterministic RSSM state | `model.py` | `cell`, `State.deterministic` |
| posterior filtering | `model.py` | `observe`, `posterior_step` |
| ensemble prior imagination | `model.py` | `priors`, `prior_step` |
| decoder/task heads | `model.py` | `decoder`, `reward`, `value`, `continuation`, `goal` |
| analytic KL and all objectives | `losses.py` | `diagonal_gaussian_kl`, `integrated_loss` |
| random-shooting MPC | `planner.py` | `RiskAwarePlanner.plan` |
| final action boundary | `planner.py` | `DiscreteActionGuard.filter` |
| training/checkpoint | `train.py` | `train` |
| closed-loop evaluation | `evaluate.py` | `run_episode`, `evaluate` |

## Training

- seed: 331
- dataset version: `integrated-partial-navigation-v1`
- train/validation sequences: 768/192
- sequence length: 12
- batch size: 32
- epochs: 60
- optimizer: Adam
- learning rate: `8e-4`
- training steps: 1,440
- parameter count: 456,037
- checkpoint format: dict with `format_version=1`, `model`, `config`, `optimizer`, `training_steps`
- checkpoint path: `outputs/checkpoint.pt`（Git対象外）

```bash
MPLCONFIGDIR=/tmp/world-model-mpl .venv/bin/python \
  99_integrated_world_model/01_evidence_selected/train.py
```

## Losses

- Reconstruction MSE: `[h_t,z_t]`が画像内容を保持するよう学習する。
- Reward MSE: imagined transitionが即時task utilityを予測する。
- Value MSE: terminal planning用の距離potential `-ManhattanDistance/4`を予測する。これは方策returnではない。
- Continuation BCE: episode継続可能性を予測する。
- Goal cross entropy: aliasされた現在観測を履歴で区別できるようにする補助loss。
- KL: 観測を使うposteriorと観測なしpriorを整合させる。free natsは0.5。
- Overshooting MSE: 3 stepまで再帰したprior meanをposterior stateへ近づける。

## Evaluation Interface

```bash
MPLCONFIGDIR=/tmp/world-model-mpl .venv/bin/python \
  99_integrated_world_model/01_evidence_selected/evaluate.py
```

`evaluation_metrics.json`にdataset version、seed、episode数、success、step、不確実性、entry pointを保存する。Phase 90 registryから再発見可能である。

## Smoke Test Results

2026-08-23のone-seed run:

- unit tests: 4 passed
- validation total: 0.04650
- reconstruction MSE: 0.0000745
- reward MSE: 0.00310
- value MSE: 0.00248
- goal loss: 0.0000360
- risk-aware MPC: 40/40 success、平均3.0 steps
- mean-only MPC: 40/40 success
- mean epistemic return std: 0.44966

`outputs/integrated_rollout.png`は、同一の視界外開始位置から右Goalと下Goalへ別の軌跡を選んだ例を示す。

## Failure Cases

初回実装はrisk-aware/mean-onlyとも20/40成功だった。右Goalだけ成功し、下Goalでも右へ進んだ。原因はtask headをposterior featureだけで学習し、計画時には未教師のprior featureへ適用したこと、およびbehavior-policy returnをterminal valueとして使ったことだった。

修正としてtask headを全prior headにも教師あり学習し、value targetを状態距離potentialへ変更した。その後40/40へ改善した。これは「各部品が単体で動く」ことと「利用時の表現上で学習されている」ことが別問題だと示す。

## Findings

- hidden Goalを区別できたmemory表現をplanningまで接続すると、異なる行動を選択できた。
- posterior/prior interfaceは統合時の主要なfailure boundaryだった。
- risk penaltyなしでも全成功したため、今回の簡単なin-distribution課題はuncertainty-aware planningの優位性を検証できない。
- decoderは表現監査には有用だが、plannerはdecoderを使わない。

## Limitations

- one seed、二つの固定Goal、決定論的5x5 Grid Worldだけである。
- 3本のpriorはencoderとGRUを共有するcorrelated headsであり、bootstrap ensembleより弱いepistemic近似である。
- RSSM stateにはmeanを使用し、確率sampleによる多峰性rolloutは行わない。
- valueは最適方策returnではなく、true state由来のdistance potential教師である。
- goal identity補助教師はtrue stateを使うため、純粋な自己教師ありworld modelではない。
- continuation headは学習したが、現在のplanner scoreでは明示的maskに使っていない。
- risk-awareのOOD、安全性、calibration優位性は未証明である。
- 実機actionは送信していない。

## Compare Later

- comparison: mean-only対risk-aware、shared-head対bootstrap ensemble、mean rollout対sampled rollout、RSSM対Transformer統合版
- metrics: OOD success、calibration、collision/unsafe proposal率、latency、memory、long-horizon error
- expected advantage: RSSMの固定サイズbelief、ensemble disagreementによる慎重な行動選択
- expected weakness: correlated ensembleとpotential-based valueによる過信・task特化
- ablations: KL、overshooting、goal loss、prior task supervision、value terminal、uncertainty penalty、decoder

## Final Model Candidate

```text
Candidate:
Yes, as a compact research baseline; not yet as a physical deployment model.

Reason:
Partial-observation memoryからprior imagination、task prediction、planning、guardまでが一つのclosed loopで動作した。

Advantages:
Evidence-selected components, explicit prior/posterior boundary, uncertainty interface,
decoder-free planning, reproducible metadata.

Disadvantages:
Small deterministic domain, correlated ensemble, one seed, task-shaped supervision,
no demonstrated risk-aware gain.

Possible conflicts:
Transformer memory requires a different state/cache interface; object slots and video tokens
would replace rather than simply append to the vector encoder; actor learning may exploit model error.
```

## Next Questions

- stochastic/OOD環境でrisk penaltyは成功率や安全性を改善するか。
- bootstrapされた独立RSSM ensembleはcorrelated prior headsよりcalibrationが良いか。
- true-state distance教師なしでvalueをpolicy returnまたはTD targetから学べるか。
- continuationをplanner return maskへ組み込むとterminal behaviorは改善するか。
- Physical AI接続前に、action guardへ速度・workspace・dead-man制約をどう統合するか。

## References

### Learning Latent Dynamics for Planning from Pixels (PlaNet)

- Authors: Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson
- Year: 2018
- Paper: https://arxiv.org/abs/1811.04551
- Used for: deterministic/stochastic recurrent state、prior/posterior、latent overshooting、latent planningの設計原理
- Implementation: `model.py`, `losses.py`
- Scope: continuous RSSM coreを用いた簡略教育実装。PlaNetの完全なCEM制御系・観測分布・訓練設定の再現ではない。

### Dream to Control: Learning Behaviors by Latent Imagination

- Authors: Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi
- Year: 2019
- Paper: https://arxiv.org/abs/1912.01603
- Used for: latent feature上のreward/value prediction、観測なしimaginationという考え方
- Implementation: `model.py`, `planner.py`
- Scope: actor-criticではなくMPCを採用した独自統合。Dreamer完全再現ではない。

### Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS)

- Authors: Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine
- Year: 2018
- Paper: https://arxiv.org/abs/1805.12114
- Used for: ensemble disagreement、trajectory prediction、uncertainty-aware model-based controlの動機
- Implementation: `model.py`のprior heads、`planner.py`
- Scope: bootstrap ensemble、probabilistic particles、continuous CEMを省いた簡略実装。

### DayDreamer: World Models for Physical Robot Learning

- Authors: Philipp Wu, Alejandro Escontrela, Danijar Hafner, Ken Goldberg, Pieter Abbeel
- Year: 2022
- Paper: https://arxiv.org/abs/2206.14176
- Used for: learned world modelとphysical action interfaceを分離し、deployment safety boundaryを明示する動機
- Implementation: `planner.py`の`DiscreteActionGuard`
- Scope: ローカルGrid Worldのみ。DayDreamerのonline robot learning再現ではない。

### Project-specific integration

- Classification: `Independent implementation` / `Experimental modification`
- Used for: paired hidden-Goal dataset、goal auxiliary loss、distance-potential value、correlated prior ensemble、risk-adjusted discrete random shooting
- Implementation: `dataset.py`, `losses.py`, `planner.py`, `evaluate.py`
