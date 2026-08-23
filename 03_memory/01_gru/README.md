# 完全観測Grid WorldにおけるGRU潜在ダイナミクス

状態: 完了（2026-08-22）。

## 目的

action-conditioned latent dynamicsへ再帰的hidden stateを導入し、one-step予測、hidden stateの伝播、autoregressiveなmulti-step rolloutが正しく動くことを、独立したPyTorch実験として確認する。

## 問題と仮説

memoryなしの遷移は現在だけから未来を予測する。

```text
(z_t, a_t) -> predicted z_{t+1}
```

同じ現在観測でも、過去によって本当の状態や必要な予測が異なるなら、この写像は曖昧になる。GRUは履歴を`h_t`へ圧縮し、予測を履歴にも条件付けられるようにする。

ただし本実験のGrid Worldは決定論的かつ完全観測であり、現在画像はほぼMarkov stateである。そのためGRUが動くことは確認できても、memoryが本当に必要・有利だとはまだ示せない。この因果的な検証は`02_partial_observation`以降で行う。

## 以前のモデル

監査時、このcheckoutには既存実装が存在しなかった。そのため`baseline.py::SimpleDynamics`をmemoryなしの参照として新規作成した。復元された既存コードと誤って扱わない。既存latent dimensionもなかったため、ここでは`latent_dim=16`を実験上の選択として明記する。

## アーキテクチャ

```text
observation sequence [B,T+1,3,20,20]
       -> CNN Encoder -> z_0 ... z_T [B,T+1,16]

z_t [16] + one-hot action a_t [4] + h_t [64]
       -> GRUCell(20 -> 64)
       -> h_{t+1} [64]
       -> prediction MLP
       -> predicted z_{t+1} [16]
       -> shared Decoder
       -> predicted observation [3,20,20]
```

学習ではteacher forcingを使い、各stepへ正解画像からencodeした`z_t`を渡す。評価rolloutでは`z_0`だけを正解から得て、以後は予測latentを次入力に戻しながら`h_t`を持ち運ぶ。

これは決定論的recurrent latent modelである。stochastic state、prior/posterior、KL、reward/value、actor、plannerは含まない。RSSM、PlaNet、Dreamerの再現ではない。

## Tensor Shapes

`B`: batch、`T=8`、`C=3`、`H=W=20`、`D_z=16`、`D_a=4`、`D_h=64`。

| Tensor | Shape | 意味 |
|---|---:|---|
| observations | `[B,T+1,3,20,20]` | 完全観測RGB系列 |
| actions | `[B,T,4]` | one-hot action |
| encoded latents | `[B,T+1,16]` | 現在画像の表現`z_t` |
| GRU input | `[B,20]` | `[z_t;a_t]` |
| hidden | `[B,64]` | 履歴を圧縮する`h_t` |
| hidden sequence | `[B,T,64]` | 更新後hidden列 |
| predicted latents | `[B,T,16]` | 次時刻latent予測 |
| decoded predictions | `[B,T,3,20,20]` | 未来画像予測 |

## 数式

`x_t=[z_t;a_t]`として、実装しているPyTorch `GRUCell`の更新は次である。

```text
r_t = sigmoid(W_ir x_t + b_ir + W_hr h_t + b_hr)
u_t = sigmoid(W_iu x_t + b_iu + W_hu h_t + b_hu)
n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_t + b_hn))
h_{t+1} = (1-u_t) * n_t + u_t * h_t
predicted z_{t+1} = g_theta(h_{t+1})
```

`r_t`はcandidateを作る際に過去をどれだけ使うか、`u_t`はcandidateと既存memoryのどちらを残すかを学習する。記号名は論文によって異なるが、ここではPyTorchの規約に合わせる。

学習目的は以下である。

```text
L = L_rec + 0.2 L_pos + 2 L_dyn
L_rec = MSE(Decoder(Encoder(o_t)), o_t)
L_pos = CE(cell_logits(Decoder(Encoder(o_t))), agent_cell_t)
L_dyn = MSE(predicted z_{t+1}, stopgrad(Encoder(o_{t+1})))
```

小さなAgentはpixel MSEだけでは無視され得るため、25セルのagent-position cross entropyを補助lossとして追加した。`stopgrad`はdynamics lossがEncoderを容易なtargetへ崩壊させるのを抑える。これはこの教育用rendererに対する独自変更であり、PlaNet/DreamerのELBOではない。

## コード対応

| 概念 | 実装 |
|---|---|
| `z_t=Encoder(o_t)` | `model.py::VisualEncoder` |
| `Decoder(z_t)` | `model.py::VisualDecoder` |
| GRU update | `model.py::GRUDynamics.step` / `torch.nn.GRUCell` |
| teacher-forced sequence | `model.py::GRUDynamics.forward` |
| autoregressive rollout | `model.py::GRUDynamics.rollout` |
| prediction head | `model.py::GRUDynamics.prediction_head` |
| 各loss | `losses.py::world_model_loss` |
| memoryなし参照 | `baseline.py::SimpleDynamics` |
| 環境・系列 | `env.py`, `dataset.py` |

## 学習と評価

```bash
uv run pytest -q 03_memory/01_gru/tests
uv run python 03_memory/01_gru/train.py
uv run python 03_memory/01_gru/evaluate.py
```

既定smoke runはtrain/validation `256/64`系列、長さ8、batch 32、40 epoch、Adam、learning rate `3e-3`、seed 7である。Encoder、Decoder、GRUをjoint trainingする。

評価はteacher-forced one-step latent/pixel error、agent-cell accuracy、8-step autoregressive rollout、horizon別error、shape、finite value、gradient、parameter数、推論時間を確認する。Simple Dynamicsはinterface確認に残しているが、まだ公平な性能比較として訓練していない。

## 結果

最終40 epoch smoke runの結果:

| 指標 | 結果 |
|---|---:|
| tests | 6 passed |
| 全parameter数 | 343,336 |
| GRU dynamics parameter数 | 21,712 |
| initial/final train loss | 0.803508 / 0.252070 |
| final validation loss | 0.278304 |
| validation reconstruction MSE | 0.003758 |
| held-out one-step latent MSE | 0.141365 |
| one-step agent-cell accuracy | 83.59% |
| 8-step rollout mean pixel MSE | 0.008376 |
| 8-step rollout mean agent-cell accuracy | 55.86% |
| GRU rollout time / sequence | 6.98 microseconds |

horizon別position accuracyは`75.0, 76.6, 64.1, 54.7, 51.6, 50.0, 37.5, 37.5%`。one-stepでは動いても長期rolloutで誤差が蓄積することを確認した。

## 失敗例・知見

- 初期のplain MSEはAgentを落としても低errorになった。
- active pixel重み付けではchance-levelのposition accuracy、Goalだけの高重みでは「全セルを赤くする」shortcutが出た。
- 小さなtransposed-convolution decoderでは位置分類が学べず、MLP decoderへ変更した。
- 最終モデルもhorizon 7–8でposition accuracy 37.5%まで低下した。

pixel MSEだけでは意味的な状態予測を保証しない。one-step成功もstable imaginationを意味しない。そして完全観測の結果はGRU実装の妥当性を示すだけで、memoryの必要性は示さない。

## 比較・限界・次の問い

| 性質 | Simple Dynamics | GRU Dynamics |
|---|---|---|
| 入力 | `z_t,a_t` | `z_t,a_t,h_t` |
| 履歴 | なし | `h_t`へ圧縮 |
| rolloutで持つstate | `z_t` | `z_t`と`h_t` |
| 完全観測での必要性 | 概ね十分 | 概ね冗長 |

Candidate: **Undecided**。固定長の履歴summaryと明示的rollout stateは利点だが、逐次計算、忘却、追加parameter、uncertainty不在という弱点がある。次は`02_partial_observation`で現在frameだけでは区別できない状態を作り、その後にmatched comparisonを行う。

## 参考文献

### Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation

- Authors: Kyunghyun Cho et al.
- Year: 2014
- Paper: https://aclanthology.org/D14-1179/ / https://doi.org/10.3115/v1/D14-1179
- 利用箇所: GRUのgating。`model.py::GRUDynamics`は`torch.nn.GRUCell`を使う。
- 差分: 機械翻訳用の原論文に対するaction-conditioned latent transitionへの教育用適用。

### World Models / Recurrent World Models Facilitate Policy Evolution

- Authors: David Ha, Jürgen Schmidhuber
- Year: 2018
- Paper: https://arxiv.org/abs/1803.10122 / https://arxiv.org/abs/1809.01999
- 利用箇所: 視覚表現の後にrecurrent temporal modelを置く文脈。
- 差分: MDN-RNN、VAE、controllerは実装していない。

### Learning Latent Dynamics for Planning from Pixels (PlaNet)

- Authors: Danijar Hafner et al.
- Year: 2018 / ICML 2019
- Paper: https://arxiv.org/abs/1811.04551
- 利用箇所: 後続RSSMとの概念的な比較。ここはprior/posterior/KLを含まない。
