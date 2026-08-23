# 研究ノート: RSSM

日付: 2026-08-22。将来の記事の材料であり、記事本文ではない。

## 実装前の疑問

- stochastic stateは追加memoryか、それともdeterministic memoryに条件付く現在の不確実stateか。
- 学習ではposteriorが画像を見るのに、rolloutのpriorを何が学習させるのか。
- deterministicなtwo-case Grid Worldでdiagonal Gaussianは意味あるuncertaintyを学ぶのか。

## 修正された理解

- `h_t`と`z_t`は交換可能なlatentではない。前者は決定論的履歴、後者は分布を持つ現在state。
- priorはregularizer専用networkではなく、imagination中に未来stateを作るnetworkである。
- stochastic RSSMがcalibrated uncertaintyを自動的に意味するわけではない。

## 失敗と修正

- image MSE+KLだけでは初期Goalがright/downの平均になった。
- decoded RGBからGoal classを取るとred artifactでmetricを攻略された。
- classifyingを`[h_t,z_t]`上の独立headへ移し、channel weight `[1,20,1]`を使った。主Goalは改善したが、初期frameには薄い第二candidateが残った。

## 保存すべき結果と記事材料

- 21 combined tests、428,330 parameters、seed 23、160 Adam steps。
- validation total 0.010783、raw KL 1.756577 nats。
- posterior / one-step prior / six-step prior pixel MSE: 0.000746 / 0.000618 / 0.000628。
- Goal state headはposterior、prior、six horizonで100%。ただしone-seed smoke result。
- 「posteriorは学習時の答え合わせ、priorは想像時の閉本予測。KLが前者から後者へ知識を移す」。
- 図: `reconstruction.png`、`latent_rollout.png`、`loss_curve.png`、`rollout_error.png`。

## 次に比較したいこと

multiple stochastic sampleとcalibration/NLL、history shuffle/reset、No Memory/GRU/RSSM/Transformerのmatched data、image qualityとstate qualityの分離、parameter/latency/memory比較。
