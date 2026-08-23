# RSSMを理解する

## 解決する問題

deterministic GRUは履歴を持てるが、「観測を見る前の予測」と「観測を見た後の推論」を区別しない。RSSMはpriorとposteriorを分け、観測系列で学びながら未来を観測なしで想像できるようにする。

## Before / After

```text
Before: h_{t+1}=GRU([z_t,a_t],h_t), z_hat_{t+1}=head(h_{t+1})
After:  h_t=GRU(h_{t-1},[z_{t-1},a_{t-1}])
        p(z_t|h_t): 観測前の予測
        q(z_t|h_t,o_t): 観測後の補正
```

`h_t`は決定論的な履歴、`z_t`は現在の確率的仮説である。Encoder output `e_t`は観測特徴であり、`z_t`そのものではない。

## 数式と必要性

```text
h_t = GRU(h_{t-1},[z_{t-1},a_{t-1}])
p(z_t|h_t) = N(mu_p, diag(sigma_p^2))
q(z_t|h_t,o_t) = N(mu_q, diag(sigma_q^2))
z_t = mu_q + sigma_q * epsilon, epsilon~N(0,I)
```

- `h_t`: actionと順序を運ぶ。ないとpriorは履歴を失う。
- prior: future observationなしで使う分布。imaginationに必須。
- posterior: 実画像が与える追加情報でstateを補正する。
- reparameterization: randomnessを`epsilon`へ移し、`mu/sigma`へgradientを流す。
- positive std: `softplus(raw_std)+0.1`でKLの不安定を避ける。

```text
KL(q||p) = log(sigma_p/sigma_q)
         + (sigma_q^2+(mu_q-mu_p)^2)/(2*sigma_p^2) - 1/2
```

reconstructionだけならposteriorだけが良くなり得る。KLはpriorがposteriorのstateを予測するようにし、rollout時にDecoderが未学習latentへ行くのを防ぐ。free natsは小さなKLをさらに潰す圧力を止める。

## 観測あり学習と観測なしrollout

```text
observed sequence:
previous state/action -> h_t -> prior
o_t -> Encoder -> e_t -> posterior -> z_t -> decode

future rollout:
posterior seed -> action -> h_{t+1} -> prior -> z_{t+1} -> decode
```

future imageをrolloutへ渡すとfilteringになり、predictionではなくなる。

## コード対応

| 概念 | 実装 |
|---|---|
| `h_t` transition | `rssm.py::RecurrentStateSpaceModel.transition` |
| prior | `prior`, `GaussianHead.forward` |
| posterior | `infer_posterior` |
| sample | `DiagonalGaussian.sample` |
| filtering / imagination | `observe` / `imagine` |
| image model | `ObservationEncoder`, `ObservationDecoder` |
| KL / objective | `rssm_losses.py` |

## 外すとどうなるか

| 外す部品 | 結果 |
|---|---|
| `h_t` | priorが再帰的履歴を失う |
| `z_t` | deterministic recurrent modelへ戻る |
| prior | 観測なし未来stateを作れない |
| posterior | 現在観測によるbelief修正ができない |
| KL | posterior再構成が良くてもprior rolloutが壊れ得る |
| reparameterization | sampleからGaussian parameterへgradientが流れない |
| Decoder/head | stateに視覚的・意味的情報を残す理由が弱い |
| future-image遮断 | evaluationがground truth leakする |

## 説明できるようになる確認項目

- `h_t`、`z_t`、`e_t`の違いを説明できるか。
- priorとposteriorが知っている情報の差は何か。
- なぜtrainingはposterior、imaginationはpriorを使うか。
- KLの両側に何の分布があり、なぜ必要か。
- reparameterizationはどのgradient問題を解くか。
- 画像再構成が良くてもprior rolloutが悪くなり得る理由は何か。
- stochastic stateがcalibrated uncertaintyを保証しない理由は何か。

## 次の問い

- prior stdは小さな決定論的taskで意味のあるuncertaintyか。
- sampled rolloutやcategorical likelihoodは平均化artifactを減らすか。
- Goal headをprobeだけにしてもmemory情報は残るか。
