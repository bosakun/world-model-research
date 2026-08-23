# Transformer Memory: latent/action履歴へのcausal attention

状態: 完了（2026-08-22）。TransDreamerとTransformerを参考にした小規模教育用causal world modelであり、TransDreamerの完全再現ではない。

## 目的・問題・仮説

GRU/RSSMは過去を一つのrecurrent vectorへ圧縮する。効率的だが、古い出来事を使うには情報が全intermediate updateを生き残る必要がある。Transformerは過去のlatent/action token列を明示的に保持し、現在の予測がcontent-dependentなweightで履歴へアクセスできる。

仮説は、positionとcausal maskを持つTransformerなら、GRU hiddenなしでlatent/action履歴から未来latentを予測できる、である。長期履歴でrecurrent memoryを上回るかはPhase 90以前には結論しない。

## アーキテクチャ

```text
o_t -> CNN Encoder -> z_t [16]
a_t [4] -> concat[z_t,a_t] [20]
       -> linear projection [64] + learned position p_t
       -> token x_t [64]
       -> 2 x causal Transformer block
          (4-head attention + feed-forward)
       -> context c_t [64]
       -> next-latent head -> z_hat_{t+1} [16]
       -> Decoder / Goal-state head
```

teacher forcingでは正解画像からencodeした`z_t`で各tokenを作る。rolloutでは`z_hat_{t+1}`を次tokenに追加し、future imageはencodeしない。

## データフローとTensor Shapes

```text
o_0...o_T + a_0...a_{T-1}
-> z_0...z_T
-> ([z_0,a_0]+p_0) ... ([z_{T-1},a_{T-1}]+p_{T-1})
-> causal self-attention
-> z_hat_1...z_hat_T -> decoded future image / Goal state
```

`B=32`、`T=6`、model width `D=64`、head数`H=4`、layer数`L=2`。

| Tensor | Shape | 意味 |
|---|---:|---|
| observations | `[B,T+1,3,20,20]` | partial image sequence |
| actions | `[B,T,4]` | one-hot actions |
| latents | `[B,T+1,16]` | frame representation |
| content tokens | `[B,T,20]` | `[z_t,a_t]` |
| input/context tokens | `[B,T,64]` | content projection + position |
| attention | `[L,B,H,T,T]` | query-to-key weights |
| predicted latents | `[B,T,16]` | `z_hat_{t+1}` |
| predicted images | `[B,T,3,20,20]` | decoded future |
| Goal logits | `[B,T,10]` | 9位置 + not-visible |

## 数式

tokenは次で作る。

```text
x_t = W_x [z_t,a_t] + p_t
Q=XW_Q, K=XW_K, V=XW_V
A = softmax(QK^T / sqrt(d_k) + M)
Attention(X) = AV
```

causal maskは`j<=i`なら0、`j>i`なら`-infinity`である。したがってsoftmax後、query `i`は未来key `j>i`へ正確に0のweightしか置けない。

```text
Y = X + MultiHeadAttention(LayerNorm(X), M)
C = Y + FFN(LayerNorm(Y))
z_hat_{t+1} = W_o c_t
L = L_reconstruct + L_future_image
  + 0.5 MSE(z_hat_{t+1}, stopgrad(z_{t+1})) + 0.1 L_goal
```

image channel weight `[1,20,1]`とGoal headは小さいGoalを背景に埋もれさせないための独自変更であり、TransDreamer本体の機構ではない。確率的stateを含まないためKLはない。

## コード対応

| 概念 | 実装 |
|---|---|
| Encoder / Decoder | `transformer_memory.py::VisualEncoder`, `VisualDecoder` |
| latent/action tokenization | `TransformerMemoryDynamics.tokenize` |
| positional information | `position_embedding` |
| causal mask | `TransformerMemoryDynamics.causal_mask` |
| attention + residual FFN | `CausalTransformerBlock.forward` |
| teacher-forced sequence | `TransformerMemoryDynamics.forward` |
| autoregressive rollout | `TransformerMemoryDynamics.rollout` |
| loss | `transformer_losses.py` |
| attention plot | `evaluate.py::evaluate` |

## 学習・評価

```bash
.venv/bin/python 03_memory/04_transformer_memory/train.py
.venv/bin/python 03_memory/04_transformer_memory/evaluate.py
.venv/bin/python -m pytest -q 03_memory
```

seed 29、`partial-observation-v1`、train/validation `128/32`、6 action/7 image、width 64、4 heads、2 layers、FFN 128、max context 16、dropout 0、batch 32、40 epoch、Adam `3e-3`、160 steps、406,794 parameters。checkpointはGit対象外。

`evaluate.py`はautoencoder reconstruction、teacher-forced one-step、autoregressive rollout、Goal state-head accuracy、latent/token/attention shape、parameter数、plotsを別々に出す。

## Smoke Test結果

Memory全4実験を合わせて29 testsが成功した。shape、finite forward/backward、Encoder/Decoder/token projection/position/attention/headへのgradient、future attentionが厳密に0であること、future token変更へのcausal invariance、position差、context境界、autoregressive rollout、dataset互換を検査した。

| 指標 | 結果 |
|---|---:|
| train total epoch 1 -> 40 | 1.794921 -> 0.043907 |
| validation total | 0.035018 |
| validation weighted future image | 0.011991 |
| validation latent prediction | 0.000094 |
| autoencoder pixel MSE | 0.001481 |
| teacher-forced one-step pixel MSE | 0.000964 |
| 6-step autoregressive pixel MSE | 0.000964 |
| one-step / rollout Goal accuracy | 1.000 / 1.000 |

最終layerのattention図では未来側upper triangleが厳密にmaskされ、利用可能な履歴へnonzero weightがある。これはcausalityとaccessを示すが、attention weightだけで因果的重要性を証明するものではない。

## 失敗例・限界・比較

- Goalがview外へ消えた後、rollout画像には薄いgreen traceが残る。semantic state headとpixel-perfectなDecoderを区別する必要がある。
- 6 tokenの小さなdatasetでは長期Transformer優位を示せない。
- learned absolute positionは`max_context=16`まで。長い推論には設計変更またはsliding windowが必要。
- dense attentionはcontext lengthに対しtime/memoryが概ね二乗で増える。

Candidate: **Undecided**。明示的履歴accessと並列teacher-forced処理は利点だが、固定context、autoregressive再計算、data要求、RSSM uncertainty semantics不在が弱点である。後でNo Memory/GRU/RSSM/Transformerをmatched split/seedで比較し、one/5/10-step、hidden Goal、long-context recall、latency、peak memory、stabilityを測る。

## 参考文献

### TransDreamer: Reinforcement Learning with Transformer World Models

- Authors: Chang Chen, Yi-Fu Wu, Jaesik Yoon, Sungjin Ahn
- Year: 2022
- Paper: https://arxiv.org/abs/2202.09481
- 利用箇所: Transformer world modelの動機。
- 差分: 本実験はdeterministic causal Transformerで、TSSM、prior/posterior、reward、actor/valueはない。

### Attention Is All You Need

- Authors: Ashish Vaswani et al.
- Year: 2017
- Paper: https://arxiv.org/abs/1706.03762
- 利用箇所: multi-head attention、position、causal mask、residual FFN。
