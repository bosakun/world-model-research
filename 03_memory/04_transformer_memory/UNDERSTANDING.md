# Transformer Memoryを理解する

## 解決する問題

recurrent modelは現在hiddenを通してしか過去を参照できない。Transformer memoryは過去tokenのwindowを残し、各予測がその履歴へweighted retrievalを行う。重要なcueが遠い過去にあるとき、直接accessが有利かが研究上の問いである。

## Before / After

```text
GRU/RSSM: h_0 -> h_1 -> ... -> h_t
Transformer: x_0, x_1, ..., x_t を保持し、token tが0...tへattention
```

Transformerも無限memoryではない。`max_context`を越えて切ったtokenは忘れる。recurrent compressionをなくすが、state表現やcontext管理の必要をなくすわけではない。

## Tokenとcausal attention

```text
x_t = W_x[z_t,a_t] + p_t
```

- `z_t`: 現在画像の16次元latent。
- `a_t`: その後に実行する4次元one-hot action。
- `p_t`: learned position。attentionだけでは順序を知らないため必要。

```text
Q=XW_Q, K=XW_K, V=XW_V
A=softmax(QK^T/sqrt(d_k)+M)
Y=AV
```

queryは「今必要な情報」、keyは「各tokenが持つ手掛かり」、valueは「選ばれたときに取り出す内容」と考えられる。`sqrt(d_k)`はwidthが大きいだけでsoftmaxが飽和するのを抑える。

```text
M_ij = 0 if j<=i, -infinity if j>i
```

future attentionを厳密に0にするmaskがないと、training時に`z_{t+1}`以降を読んで答えを漏らす。future tokenを大きく変えても過去outputが変わらないtestが重要である。

## Teacher forcingとrollout

```text
teacher forcing: true z_tを全tokenへ入れてz_hat_{t+1}を予測
rollout: z_0から開始し、z_hat_{t+1}を次tokenへ戻す
```

前者は全causal positionをparallelに処理できる。後者は自分の予測を消費するため逐次的で、compounding errorを示す。

## コード対応

| 概念 | 実装 |
|---|---|
| token / position | `TransformerMemoryDynamics.tokenize`, `position_embedding` |
| causal mask | `causal_mask` |
| Q/K/V、attention、FFN | `CausalTransformerBlock` |
| next latent | `prediction_head` |
| teacher-forced / rollout | `forward` / `rollout` |
| loss | `transformer_losses.py` |
| attention可視化 | `evaluate.py` |

## 外すとどうなるか

| 外す部品 | 結果 |
|---|---|
| action | 同じ画像から異なるcontrolの未来が曖昧になる |
| position | ordered trajectoryでなくbag of pairsになる |
| causal mask | future leakによりone-step metricが無効になる |
| history | memoryなしfeed-forward transitionへ戻る |
| residual/norm | 深いmodelの最適化が不安定になり得る |
| image/semantic target | jointly learned latentが意味を失う可能性 |
| autoregressive evaluation | compounding errorが隠れる |

## Recurrent memoryとの比較

| 問い | GRU/RSSM | Causal Transformer |
|---|---|---|
| 保存形式 | fixed-size hidden | bounded token list |
| 古いeventへのaccess | 全中間updateを通る | context内ならdirect edge |
| teacher-forced処理 | 逐次 | parallel |
| rollout state | hidden vector | token history/cache |
| scaling | 概ね線形 | dense attentionは概ね二乗 |

## 説明できるようになる確認項目

- 一つのtokenに何を入れるか。
- `z_t`と`a_t`を組にして`z_{t+1}`を予測する理由。
- positionがないと何が失われるか。
- causal maskのどの要素を禁止するか。
- future token変更testがなぜ強いか。
- attention mapのrow/columnは何を表すか。
- attention weightが因果的重要性の証明にならない理由。
- teacher forcingとautoregressive rolloutの違い。
- TransDreamerの完全なTSSMと異なる点。

## 次の問い

- どのhistory長でattentionが64次元recurrent stateを上回るか。
- 特定headは初期Goal tokenを担当するか。
- relative/rotary positionはcontext外へより一般化するか。
- Transformer contextとstochastic prior/posteriorを組み合わせるべきか。
- KV cacheはrollout costをどう変えるか。
