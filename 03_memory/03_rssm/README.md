# RSSM: 観測で補正し、観測なしで想像する潜在ダイナミクス

状態: 完了（2026-08-22）。PlaNetを主要な参考にした教育用の簡略実装であり、完全再現ではない。

## 目的

GRUは履歴を保持できるが、観測を見る前の予測と、観測を見た後のstate推論を区別しない。RSSMは決定論的state `h_t`、stochastic state `z_t`、予測用prior、観測で補正するposteriorを持つ。観測系列で学び、未来画像なしでもpriorだけでrolloutすることが目的である。

## アーキテクチャ

```text
o_t -> CNN Encoder -> e_t
(h_{t-1}, z_{t-1}, a_{t-1}) -> GRU -> h_t
h_t -> prior p(z_t|h_t)
(h_t,e_t) -> posterior q(z_t|h_t,e_t) -> z_t
[h_t,z_t] -> Decoder / Goal state head

posterior seed + future actions -> prior only -> imagined future images
```

`h_t`はactionと過去stateから決定論的に更新される履歴、`z_t`は現在の確率的stateである。学習時はposterior、future imaginationではpriorだけを使う。評価用`true_states`と`full_worlds`はmodel入力ではない。

## Tensor Shapes

`B=32`、action horizon `T=6`、embedding `E=64`、`D_h=64`、`D_z=16`。

| Tensor | Shape | 意味 |
|---|---:|---|
| observations | `[B,T+1,3,20,20]` | partial image sequence |
| actions | `[B,T,4]` | one-hot action |
| embeddings | `[B,T+1,64]` | `e_t` |
| deterministic state | `[B,T+1,64]` | `h_t` |
| stochastic state | `[B,T+1,16]` | posterior sample `z_t` |
| prior/posterior mean,std | `[B,T+1,16]` | Gaussian parameters |
| reconstructions | `[B,T+1,3,20,20]` | posterior stateからの画像 |
| imagined observations | `[B,T,3,20,20]` | prior-only rollout |

## 数式

```text
h_t = GRU(h_{t-1}, [z_{t-1},a_{t-1}])
p(z_t|h_t) = N(mu_p, diag(sigma_p^2))
q(z_t|h_t,o_t) = N(mu_q, diag(sigma_q^2))
epsilon ~ N(0,I)
z_t = mu_q + sigma_q * epsilon
o_hat_t = Decoder([h_t,z_t])
```

reparameterizationにより乱数を`epsilon`へ移し、`mu`と`sigma`へgradientを流す。analytic KLは次である。

```text
KL(q||p) = log(sigma_p/sigma_q)
         + (sigma_q^2+(mu_q-mu_p)^2)/(2 sigma_p^2) - 1/2
L = L_weighted_image + 0.1 L_goal + 0.001 max(KL(q||p), 1 nat)
```

KLは観測を見たposterior stateを、画像なしで使うpriorが予測するようにする。free natsは小さなKLをさらに潰す圧力を抑える。green channelのweight 20とGoal cross entropyは、小さいGoalをpixel MSEが無視しないための独自教育用変更である。

## コード対応

| 概念 | 実装 |
|---|---|
| `o_t -> e_t` | `rssm.py::ObservationEncoder` |
| recurrent transition | `RecurrentStateSpaceModel.transition` |
| prior/posterior | `GaussianHead`, `prior`, `posterior` |
| reparameterization | `DiagonalGaussian.sample` |
| 観測付きfiltering | `RecurrentStateSpaceModel.observe` |
| 観測なしimagination | `RecurrentStateSpaceModel.imagine` |
| Decoder / Goal head | `ObservationDecoder`, `goal_head` |
| analytic KL / objective | `rssm_losses.py` |

## 学習・評価

```bash
uv run python 03_memory/03_rssm/train.py
uv run python 03_memory/03_rssm/evaluate.py
uv run pytest -q 03_memory/01_gru/tests 03_memory/02_partial_observation/tests 03_memory/03_rssm/tests
```

seed 23、`partial-observation-v1`、train/validation `128/32`系列、6 action/7 observation、batch 32、40 epoch、Adam `3e-3`、160 steps、428,330 parameters。checkpointは`outputs/checkpoint.pt`（format version 1、Git対象外）。

## Smoke Test結果

GRU・partial observation・RSSMを合わせて21 testsが成功した。shape/finite value、positive std、reparameterization gradient、KL、Encoder/GRU/prior/posterior/Decoder/headへのgradient、posteriorの観測依存性、priorの未来観測非依存性、prior-only imaginationを検査した。

| 指標 | 結果 |
|---|---:|
| initial/final train loss | 1.036060 / 0.016743 |
| validation total | 0.010783 |
| validation weighted reconstruction | 0.008861 |
| raw KL | 1.756577 nats |
| posterior pixel MSE | 0.000746 |
| one-step prior pixel MSE | 0.000618 |
| 6-step prior rollout pixel MSE | 0.000628 |
| posterior/prior Goal-head accuracy | 1.000 / 1.000 |

## 失敗例・知見

曖昧な初期frameではDecoderが正しいright Goalに加え、下方向へ薄い第二候補を描く。pixel MSEは平均化を許すため、100%のstate-head accuracyを「完全な画像」と誤解してはいけない。

初期のplain RGB MSEは小さいGoalを無視した。decoded RGBからGoal classを取る方法はred artifactでmetricを攻略されたため、`[h_t,z_t]`上の独立headへ変更した。deterministic mean rolloutはsample diversityやcalibrationを示さず、Phase 04で評価する。

## 限界・比較・採否

Candidate: **Undecided**。prior/posterior、recurrent deterministic state、differentiable samplingは有望だが、tiny deterministic worldとone seedではGRUより良い・calibratedだとは示せない。

- 連続diagonal Gaussianのみで、DreamerV2のdiscrete stateではない。
- reward、continuation、actor/critic、planning、本格的overshootingは未実装。
- 後でNo Memory / GRU / RSSM / Transformerを同一split、複数seed、one/5/10-step error、hidden Goal、calibration、parameter、latencyで比較する。

## 参考文献

### Learning Latent Dynamics for Planning from Pixels (PlaNet)

- Authors: Danijar Hafner et al.
- Year: 2018 / ICML 2019
- Paper: https://arxiv.org/abs/1811.04551
- 利用箇所: deterministic/stochastic state、prior/posterior、latent imagination。
- 差分: synthetic dataset、weighted reconstruction、Goal head、訓練設定は独自。PlaNet完全再現ではない。

### Auto-Encoding Variational Bayes

- Authors: Diederik P. Kingma, Max Welling
- Year: 2013
- Paper: https://arxiv.org/abs/1312.6114
- 利用箇所: reparameterizationとdiagonal Gaussian KL。`rssm.py`、`rssm_losses.py`。
