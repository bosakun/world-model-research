# GRU Memoryを理解する

## 何を解決するか

feed-forward dynamicsは今渡された変数だけを見る。観測が不完全なら、同じ`z_t`でも別の本当の状態に対応し、異なる未来を予測すべきことがある。GRUは過去の`(z,a)`を反映した`h_t`を持たせ、履歴に依存する予測を可能にする。

ただし現在の環境は完全観測なので、これは「機構の動作確認」であって「memoryの効果の証明」ではない。

## Before / After

```text
Before: predicted z_{t+1} = f(z_t, a_t)

After:  h_{t+1} = GRUCell([z_t;a_t], h_t)
        predicted z_{t+1} = g(h_{t+1})
```

`z_t`は現在画像だけのEncoder出力`[B,16]`、`h_t`は以前のupdateに依存する履歴state`[B,64]`である。同じ`z_t`でも異なる`h_t`になり得る。

## データフロー

1. 正解画像を`z_0...z_T`へencodeする。
2. `[z_t;a_t]`を作り、`h_t`とともにGRUCellへ渡す。
3. 更新直後の`h_{t+1}`から`predicted z_{t+1}`を得る。
4. trainingでは正解`z_t`を次stepへ渡す（teacher forcing）。
5. rolloutでは予測`z_{t+1}`を次stepへ戻し、hiddenも持ち運ぶ。
6. episode境界ではhiddenをzeroへresetする。別episodeへ持ち越すと情報leakになる。

## 数式

### 入力

```text
x_t = [z_t;a_t] in R^20
```

`a_t`をone-hotにすることで、left/rightなどに存在しない大小関係を与えない。

### GRU gates

```text
r_t = sigmoid(W_ir x_t + b_ir + W_hr h_t + b_hr)
u_t = sigmoid(W_iu x_t + b_iu + W_hu h_t + b_hu)
n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_t + b_hn))
h_{t+1} = (1-u_t) * n_t + u_t * h_t
```

- `r_t`: candidateを作るとき、過去の各成分をどれだけ使うか。
- `u_t`: 古いhiddenを残す比率。1に近い成分ほど保持される。
- `n_t`: 現在入力と選択した過去から作る候補memory。

gateにより、RNNが毎stepすべてを書き換えて長期情報を失う問題を緩和する。

### 予測とloss

```text
predicted z_{t+1} = g_theta(h_{t+1})
L = L_rec + 0.2 L_pos + 2 L_dyn
```

- `L_rec`: latentが画像を保持するようにする。なければconstant latent collapseが可能。
- `L_pos`: 小さいAgent位置を25分類で守る。pixel MSEのbackground shortcutを検出する。
- `L_dyn`: 次時刻latentを予測する。targetをdetachし、Encoderとdynamicsが簡単な表現へ共倒れするのを抑える。

## コード対応

| 知りたいこと | ファイル |
|---|---|
| 観測、action、境界処理 | `env.py` |
| 系列tensor | `dataset.py` |
| Encoder / Decoder | `model.py::VisualEncoder`, `VisualDecoder` |
| updateのタイミング | `model.py::GRUDynamics.step` |
| training sequence | `GRUDynamics.forward` |
| rollout時のstate更新 | `GRUDynamics.rollout` |
| lossとdetach | `losses.py` |
| memoryなし反実仮想 | `baseline.py::SimpleDynamics` |

## 重要部品を外すと

| 外す部品 | 起きること |
|---|---|
| hidden / GRU | 履歴は予測へ影響しない。完全観測では差が小さい場合もある |
| update gate | old stateを保つ経路が弱くなり、長期保持が難しい |
| reset gate | candidateが不要な過去を選別しにくい |
| action | 同じ状態で複数の次状態が曖昧になる |
| reconstruction loss | latent collapseが低loss解になる |
| dynamics loss | 画像復元しても未来を学ばない |
| episode reset | 別episodeの情報が漏れる |
| rollout中のhidden carry | 毎stepone-step modelをresetするのと同じになる |

## 説明できるようになる確認項目

- `z_t`と`h_t`の情報源・shape・役割を区別できるか。
- 観測が`T+1`枚でactionが`T`個なのはなぜか。
- `h_t`がいつ`h_{t+1}`へ更新されるか。
- GRUCellの入力と出力は何か。
- GRUが「記憶する」とは、どの情報がどの経路を通ることか。
- teacher forcingだけで長期rolloutを評価できない理由は何か。
- 完全観測Grid Worldでmemoryの効果を結論できない理由は何か。
- なぜこの実装はPlaNet、RSSM、Dreamerではないのか。

## 次に残る問い

- 観測を数step隠したとき、`h_t`はGoal情報を保持するか。
- 同程度のparameter数のMLPと比較して、改善はrecurrence由来だと示せるか。
- scheduled samplingやmulti-step lossはrollout driftを下げるか。
- hidden利用をloss以外にどう計測するか（history shuffle、probe、gate統計）。
