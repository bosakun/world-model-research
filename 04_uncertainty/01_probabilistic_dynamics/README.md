# 確率的Dynamics: 世界にある偶然を予測する

状態: 2026-08-22に完了。これはPETSの考え方を参考にした独立した小規模実装であり、PETSの完全再現ではありません。

英語の技術記録（数式の原文、全実験条件、詳細な出典）は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) に保存しています。このREADMEでは、同じ実装内容を日本語で説明します。

## Purpose

次stateを一つの点として予測するmodelを、「中心」と「ばらつき」を持つGaussian分布として予測するmodelへ変えます。ここで扱うのは aleatoric uncertainty、つまり十分なデータがあっても残る世界そのものの偶然です。

## Problem

MSEで学習したmodelは、複数の結果があり得るとき平均付近を一つ出します。しかし、予測が安定している場所と、結果が大きく揺れる場所を区別できません。平均だけの未来を使ってplanningすると、危険な広がりを見落として自信過剰になり得ます。

## Previous Model

03 Memoryまでのmodelは、画像またはlatentを決定的に予測していました。RSSMにはstochastic latentがありますが、「予測した標準偏差が実際の遷移ノイズと合うか」は評価していませんでした。今回は正解のノイズ量が分かる小さな連続状態環境で確認します。

## Hypothesis

Gaussian NLLで学習すれば、modelは平均的な移動と、入力により変化する既知のノイズ量の両方を学ぶはずです。sampleしたrolloutは時間とともに広がり、平均だけのrolloutは毎回同じ軌跡になるはずです。

## Architecture

    state s_t [2] + one-hot action a_t [4]
                         |
                        MLP
                    /         \
         mean delta [2]    raw log variance [2]
                    \         /
              Gaussian N(mu, diag(sigma^2))
                         |
               mean またはsampleした次state

環境側ではactionごとの移動量に、場所とactionで大きさが変わるGaussian noiseを加えます。真のnoise standard deviationは評価の答え合わせにだけ使います。

## Tensor Shapes

| Tensor | Shape | 意味 |
|---|---|---|
| states | [B, 2] | 連続座標 (x, y) |
| actions | [B, 4] | left/right/down/upのone-hot |
| next states | [B, 2] | noiseを含む次state |
| mean / log variance / std | [B, 2] | 予測Gaussianのパラメータ |
| sequence states | [B, T+1, 2] | 正解rollout |
| sequence actions | [B, T, 4] | action列 |
| rollout states / means / stds | [B, T, 2] | modelの未来予測 |

## Mathematics

環境の遷移は次です。

    s_{t+1} = s_t + delta(action_t) + epsilon_t
    epsilon_t ~ N(0, diag(sigma_true(s_t, action_t)^2))

modelは次の分布を予測します。

    p_theta(s_{t+1} | s_t, action_t)
    = N(mu_theta, diag(sigma_theta^2))

Gaussian NLLには二つの役割があります。

    正解と平均の差が大きい -> lossが増える
    varianceを大きく出しすぎる -> lossが増える

よってmodelは、当てられる場所では狭く、偶然が大きい場所では広く予測する必要があります。log varianceには学習可能な上下限を置き、極端に小さい/大きいvarianceによる数値不安定を防ぎます。

sampleは次の形です。

    s_hat = mu + sigma * epsilon
    epsilon ~ N(0, I)

## Code Mapping

| 概念 | コード |
|---|---|
| ノイズを持つ環境 | stochastic_dataset.py の stochastic_transition |
| 真のノイズ量 | transition_noise_std |
| Gaussian model | probabilistic_dynamics.py の ProbabilisticDynamics |
| varianceの制限 | ProbabilisticDynamics.forward |
| reparameterized sampling | GaussianPrediction.sample |
| Gaussian NLL | probabilistic_losses.py の diagonal_gaussian_nll |
| sample rollout | ProbabilisticDynamics.rollout |
| coverageとグラフ | evaluate.py の evaluate |

## Training

    .venv/bin/python 04_uncertainty/01_probabilistic_dynamics/train.py
    .venv/bin/python 04_uncertainty/01_probabilistic_dynamics/evaluate.py
    .venv/bin/python -m pytest -q 04_uncertainty/01_probabilistic_dynamics/tests

| 項目 | 値 |
|---|---|
| seed / dataset | 37 / heteroscedastic-point-v1 |
| train / validation | 1024 / 256 transitions |
| model | 64 unitのSiLU層2つ、diagonal Gaussian |
| optimizer / learning rate | Adam / 1e-3 |
| batch / epochs / steps | 64 / 80 / 1280 |
| parameter数 | 4,872 |
| checkpoint | outputs/checkpoint.pt（git管理外） |

## Losses

- Gaussian NLL: 平均とaleatoric varianceを同時に学ぶ。
- bound regularizer（1e-4）: log varianceの上下限が不必要に広がるのを抑える。
- ensemble disagreementはここでは扱わない。epistemic uncertaintyは次の実験の担当です。

## Evaluation

評価では、next-state RMSE、Gaussian NLL、1σ/2σ coverage、予測stdと真のstdの相関、sample rolloutを確認します。

coverageは「正解が予測範囲に入った割合」です。校正されたGaussianなら、1σは約68.3%、2σは約95.4%になります。ただし範囲を無限に広げてもcoverageだけは高くできるため、NLLやsharpnessも必要です。

## Smoke Test Results

8件のtestが成功しました。datasetの時系列対応、既知のheteroscedastic noise、正で有限なvariance、NLL、reparameterizationのgradient、rollout shape、sampleとmeanの違いを確認しています。

| 指標 | 結果 |
|---|---:|
| train NLL（epoch 1 -> 80） | 1.29975 -> -3.65436 |
| validation NLL | -3.73237 |
| held-out RMSE | 0.05686 |
| held-out Gaussian NLL | -3.59307 |
| 1σ / 2σ coverage | 0.6953 / 0.9531 |
| predicted/true std correlation | 0.9376 |
| mean predicted std (x, y) | (0.0532, 0.0453) |
| mean true std (x, y) | (0.0539, 0.0438) |

連続分布のNLLが負になることは正常です。狭い分布では確率密度が1を超えられるためで、負の確率という意味ではありません。

## Failure Cases / Limitations

- training範囲の外では、予測stdが真のcurveから外れます。一つのmodelだけでは「データがない」を独立して表せず、model errorがvarianceへ混ざります。
- diagonal Gaussianは、相関したnoiseや二つに分かれた結果を表せません。
- in-distributionでの校正は、未知の分布でも正しいことを保証しません。
- 小さな2次元合成環境であり、画像Dynamicsの比較ではありません。

## Findings

入力ごとに変わるnoise量を、データがある範囲ではよく回復できました。平均の正しさと、不確実さの校正は別々に測る必要があります。training範囲の外での問題が、次のensemble実験の動機です。

## Compare Later

deterministic MSE modelとの比較、single Gaussianとensembleの比較を行います。見る指標はRMSE、NLL、coverage、calibration、rolloutの広がり、parameter数、推論時間です。

## References

- Chua et al., Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS), 2018. https://arxiv.org/abs/1805.12114
- Kendall and Gal, What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, 2017. https://arxiv.org/abs/1703.04977

この環境のnoise lawは独立した教育用実装です。PETS由来なのはprobabilistic head、variance bounds、trajectory samplingの考え方です。
