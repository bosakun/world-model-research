# Partial ObservabilityとMemoryを理解する

## 解決する問題

GRUを追加しただけでは、memoryに使うべき情報がdataset内にあるとは限らない。完全観測Grid WorldではframeにGoalが写る。本実験ではGoalをlocal cameraの外へ隠し、現在画像が同じでも本当の世界が異なる状態を作る。

## stateとobservation

- true state `s_t`: simulatorが持つ完全情報。Agentのrow/colとGoalのrow/col。
- observation `o_t`: Agent/modelが受け取る3x3 agent-centred画像。視界外はunknown。

評価では`true_states`を見てよいが、model入力に渡してはいけない。そうするとPOMDPの問題設定が消える。

## Before / After

```text
完全観測: full image -> z_t -> f(z_t,a_t)
部分観測: true state -> local observation o_t -> z_t
          past z/action -> h_t -> GRU prediction
```

`t=2`ではright Goalとdown Goalの局所画像が同じになる。現在画像の意味は履歴に依存する。

## POMDPの直感

POMDPでは世界の本当のstateは隠れており、action後に得られるobservationだけで判断する。Bayesian beliefなら候補世界の確率を明示する。GRUの`h_t`は確率分布を明示しないが、「前にGoalが右にあった」「その後leftを2回した」といった予測に役立つ証拠を保持できる。

```text
s_t^A != s_t^B でも o_t^A = o_t^B は起こり得る。
```

## データフローと数式

1. simulatorが`(agent_row,agent_col,goal_row,goal_col)`を保持する。
2. actionでAgent位置だけを更新する。
3. `O(s_t)`がAgent中心3x3を固定20x20 canvas中央へ描く。
4. Goalがlocal range外なら描かない。青は未観測、黒は観測済みempty。
5. Encoderが`o_t -> z_t`、GRUが`[z_t;a_t],h_t -> h_{t+1}`を計算する。

```text
p_{t+1}=clip(p_t+delta(a_t)),  g_{t+1}=g_t
o_t=O(s_t)
h_{t+1}=GRUCell([z_t;a_t],h_t)
```

`T`個のactionは状態を`T`回遷移させるため、observation/stateは`T+1`個ある。

## 重要部品を外すと

| 外すもの | 結果 |
|---|---|
| local observation | Goalが現在frameへ現れ、memory問題が弱くなる |
| paired alias | memoryが必要なケースが偶然入らない可能性がある |
| true-state metadata | 同一観測が異なる世界を隠したと証明できない |
| action history | 記憶したGoalとの相対関係を更新できない |
| GRU hidden carry | recurrent modelがmemoryなし処理になる |
| sequence構造 | 過去のcueを利用できない |

## 説明できるようになる確認項目

- この環境の`state`と`observation`の違いを具体的に言えるか。
- `t=2`で同じ画像でもGoalが異なる理由を説明できるか。
- 青unknownと空セルの違いは何か。
- transition単位でなくsequence datasetが必要な理由は何か。
- `h_t`へ何を保持してほしいのか。
- GRU互換性テストがGRUの優位性を証明しない理由は何か。

## 次の問い

- 学習済みGRUは最初に見たGoal方向を実際に保持するか。
- matched Simple Dynamicsよりhidden-Goal predictionで勝つか。
- history shuffleや`h_t` resetで優位性は消えるか。
- moving Goal、複数Goal、observation noiseが入るとdeterministic GRUで足りるか。
