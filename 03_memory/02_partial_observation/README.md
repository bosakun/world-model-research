# 部分観測Grid World: Memoryが必要になる状況を作る

状態: 完了（2026-08-22）。PlaNet、Dreamer、World Modelsの再現ではなく、独自の教育用環境である。

## 目的

`01_gru`の完全観測画像では現在frameから世界状態の大半が分かる。そこでAgent中心の3x3視界だけを観測として与え、視界外に消えたGoalを現在画像だけでは特定できない状況を作る。

仮説は、現在観測が不十分なら、過去の観測・action履歴を`h_t`へ保持するGRUが、memoryなしの`(z_t,a_t)->z_{t+1}`より有利になり得る、である。本実験はその必要条件を作り検査する段階であり、性能比較はまだ行わない。

## 完全観測との違い

| 性質 | `01_gru` | 本実験 |
|---|---|---|
| model入力 | 5x5世界全体 | Agent中心3x3 view |
| 視界外Goal | 起きない | unknownとして描画 |
| Agent位置 | global map上 | local view中央 |
| 真値 | 画像から見える | `true_states` / `full_worlds`で別保持 |
| 現在画像だけでGoal位置を知れるか | ほぼ可能 | 意図的に不可能 |

3x3 patchは既存Encoder互換の固定5x5/20x20 canvas中央へ描く。青い外側セルはemptyではなく「未観測」である。

## なぜMemoryが必要か

二つのpaired sequenceを作る。初期Agent位置は同じで、Goalはright `(2,3)`またはdown `(3,2)`にあり、どちらも`left, left`を実行する。

```text
t=0: Goalはright または downに見える
t=1: Agentがleftへ移動
t=2: 両Goalは3x3 viewの外へ消える

o_2^right == o_2^down
true_goal^right != true_goal^down
```

テストと`outputs/aliasing_pair.png`で、`t=2`の部分観測がbitwise一致することを確認する。memoryなしモデルは同じ現在入力から二つの隠れGoalを区別できない。GRUなら以前のGoal cueとaction履歴を`h_t`へ保持できる可能性がある。

## POMDPの直感とデータフロー

環境内部には完全なtrue state `s_t`があるが、Agentには観測`o_t=O(s_t)`だけが渡る。異なる`s_t`が同じ`o_t`を作るなら、1枚の画像を世界全体とみなせない。GRU hidden stateは過去の証拠を圧縮したbelief-like表現として機能し得る。

```text
true state s_t -> observation function O -> partial image o_t
                                      -> Encoder -> z_t
action a_t + history ------------------------------> GRU hidden h_t
```

## Tensor Shapes

`B`: batch、`T=6`、`D_a=4`、`D_z=16`、`D_h=64`。

| Tensor | Shape | 意味 |
|---|---:|---|
| partial observations | `[B,T+1,3,20,20]` | modelへ渡す局所画像 |
| full worlds | `[B,T+1,3,20,20]` | 可視化・評価専用。入力禁止 |
| actions | `[B,T,4]` | one-hot action |
| true states | `[B,T+1,4]` | `(agent_row,agent_col,goal_row,goal_col)` |
| goal visibility | `[B,T+1]` | Goalがview内かのmetadata |
| encoded latents | `[B,T+1,16]` | Encoder出力 |
| Simple / GRU predictions | `[B,T,16]` | memoryなし / recurrent予測 |
| GRU hidden states | `[B,T,64]` | 履歴表現 |

## 数式

```text
s_t = (p_t, g)
p_{t+1} = clip(p_t + delta(a_t))
o_t = O(s_t) = local_3x3(p_t,g) + unknown_elsewhere
```

`p_t`はAgent位置、`g`はepisode内で固定のGoal座標である。Goalは`|g_row-p_row|<=1`かつ`|g_col-p_col|<=1`のときだけ画像へ描く。alias pairの条件は以下であり、`torch.equal`で厳密検査する。

```text
s_t^A != s_t^B,  O(s_t^A) = O(s_t^B)
```

## コード対応

| 概念 | 実装 |
|---|---|
| true state、遷移、局所観測 | `partial_env.py::PartialObservationGridWorld` |
| dataset | `partial_dataset.py::PartialObservationSequenceDataset` |
| full truth / visibility | `true_states`, `full_worlds`, `goal_visible` |
| 既存Simple/GRUとの互換 | `model_adapters.py` |
| 可視化 | `visualize.py` |

## 実行・結果

```bash
uv run pytest -q 03_memory/01_gru/tests 03_memory/02_partial_observation/tests
uv run python 03_memory/02_partial_observation/visualize.py
```

13 testsが成功した（既存GRU 6件 + partial observation 7件）。既存modelを変更せずに以下のtensorを通せた。

```text
encoded latents:     [2,7,16]
Simple predictions:  [2,6,16]
GRU predictions:     [2,6,16]
GRU hidden states:   [2,6,64]
```

`t=2`のalias pairは完全一致し、true Goal座標は異なった。`full_world.png`、`partial_observation.png`、`observation_sequence.png`、`aliasing_pair.png`を生成した。

## 失敗例・限界

- 局所観測だけでは、学習済みGRUがmemoryを使う証拠にならない。
- Goalが再び3x3 viewに入れば、現在frameだけで位置を解決できる。
- 障害物、stochastic dynamics、distractor、reward/valueはない。
- `true_states`や`full_worlds`をmodel入力へ渡すと答えを漏らす。

Candidate: **Yes（評価環境として） / Undecided（統合機構として）**。これはmemory architectureを置き換えるものではなく、memoryの因果的比較を可能にするdataset/environmentである。

## 次の問いと参考文献

次は同条件でNo Memory / Simple DynamicsとGRUを学習し、hidden-Goal prediction、one-step/rollout error、history shuffle、hidden reset、parameter数、runtimeを比較する。

### Planning and Acting in Partially Observable Stochastic Domains

- Authors: Leslie Pack Kaelbling, Michael L. Littman, Anthony R. Cassandra
- Year: 1998
- Paper: https://doi.org/10.1016/S0004-3702(98)00023-X
- 利用箇所: state/observationの区別、POMDP、finite-memory controllerの文脈。
- 差分: 本環境は論文のplanning algorithm再現ではない。

### World Models / PlaNet / Dreamer

- Papers: https://arxiv.org/abs/1803.10122 / https://arxiv.org/abs/1811.04551 / https://arxiv.org/abs/1912.01603
- 利用箇所: visual representation、recurrent latent state、imaginationへつながる文脈。
- 差分: stochastic RSSM、prior/posterior、KL、reward/value、actor、planningは本実験にない。
