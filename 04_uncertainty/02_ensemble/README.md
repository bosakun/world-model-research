# Probabilistic Ensemble: 知識不足と世界の偶然を分ける

状態: 2026-08-22に完了。これはPETSに着想を得た小規模bootstrap ensembleであり、PETSの完全再現ではありません。

英語の全技術記録は [README_TECHNICAL_EN.md](README_TECHNICAL_EN.md) に保存しています。

## Purpose

前の確率的Dynamics Modelへmodel同士の不一致を追加し、二種類の不確実さを分けます。

    aleatoric: 各modelの中での世界の偶然
    epistemic: model同士の予測の食い違い

さらにPETSで使われるparticle rolloutのTS∞とTS1を実装します。

## Problem

一つの確率modelは世界のnoiseを出せますが、training範囲の外で「自分は知らない」とは言いにくいです。variance headが知識不足による誤差まで世界の偶然として吸収する可能性があります。

## Previous Model / Hypothesis

01_probabilistic_dynamicsはin-distributionで校正されたvarianceを学べましたが、training範囲外でstdが真のcurveから外れました。bootstrapで学習した複数modelは、データが多い場所では一致し、範囲外ではより不一致になるはずです。

## Architecture

    同じ state + action
      |     |     |     |     |
    model0 model1 model2 model3 model4
      |     |     |     |     |
    (mu0,var0) ...          (mu4,var4)
                 |
    ensemble mean      = member meanの平均
    aleatoric variance = member varianceの平均
    epistemic variance = member meanの分散
    total variance     = 二つの和

各memberは前実験と同じdiagonal Gaussian modelです。既存実験を上書きしていません。

## Data Flow

    transition dataset
    -> 5個のbootstrap dataset
    -> 5個の独立modelをGaussian NLLで学習
    -> 同じ入力を全memberへ渡す
    -> varianceを分解し、ID/OODとrolloutを評価

TS∞では粒子ごとに最初に選んだmemberを最後まで使います。TS1では毎stepでmemberを選び直します。

## Tensor Shapes

ensemble E=5、batch B、particle P=128、horizon T=12、state dimension=2です。

| Tensor | Shape | 意味 |
|---|---|---|
| states / actions | [B, 2] / [B, 4] | 共通入力 |
| member means / variances | [E, B, 2] | member別Gaussian |
| ensemble mean | [B, 2] | member meanの平均 |
| aleatoric / epistemic / total variance | [B, 2] | variance分解 |
| rollout particles | [B, P, T, 2] | sample trajectory |
| model IDs | [B, P, T] | 粒子が使ったmember |
| bootstrap indices | [E, N] | 重複ありsampleの行番号 |

## Mathematics

member mは次を予測します。

    p_m(y | x) = N(mu_m(x), Sigma_m(x))

全体のvarianceはlaw of total varianceで分けます。

    mean = member meanの平均
    aleatoric = member varianceの平均
    epistemic = member meanの分散
    total = aleatoric + epistemic

これは有限個のensembleによる近似であり、厳密なBayesian posteriorそのものではありません。

## Code Mapping

| 概念 | コード |
|---|---|
| bootstrap sampling | ensemble_dataset.py の bootstrap_indices |
| member集合 | probabilistic_ensemble.py の ProbabilisticEnsemble |
| variance分解 | ProbabilisticEnsemble.decompose |
| TS∞ / TS1 rollout | ProbabilisticEnsemble.rollout |
| 独立optimizer | train.py の train |
| ID/OOD評価 | evaluate.py の evaluate |

## Training

    .venv/bin/python 04_uncertainty/02_ensemble/train.py
    .venv/bin/python 04_uncertainty/02_ensemble/evaluate.py
    .venv/bin/python -m pytest -q 04_uncertainty/01_probabilistic_dynamics/tests 04_uncertainty/02_ensemble/tests

| 項目 | 値 |
|---|---|
| seed / bootstrap seed | 41 / 42 |
| dataset | heteroscedastic-point-v1、train 1024 / validation 256 |
| ensemble | 4,872 parameterのGaussian MLPを5個 |
| total parameter数 | 24,360 |
| optimizer / learning rate | memberごとのAdam / 1e-3 |
| epochs / steps | 60 / memberごとに960 |

## Losses

各memberは前実験と同じGaussian NLLとvariance-bound regularizerを使います。memberを無理に違わせるlossは使いません。bootstrap dataと初期値の違いからdiversityを作ります。

## Evaluation

IDとOODで、RMSE、NLL、total varianceのcoverage、aleatoric stdと真のnoiseの相関、epistemic stdを測ります。OODはtraining範囲外の x の領域です。

## Smoke Test Results

15件のuncertainty testが成功しました。shape、varianceの分解式、同じmeanならepistemicが0になること、bootstrap diversity、全memberへのgradient、TS∞/TS1のmodel IDを確認しています。

| 指標 | 結果 |
|---|---:|
| train member-mean NLL（epoch 1 -> 60） | 1.30675 -> -3.56282 |
| validation member-mean NLL | -3.68912 |
| held-out RMSE | 0.06051 |
| 1σ / 2σ coverage | 0.7324 / 0.9668 |
| aleatoric std correlation | 0.9526 |
| ID epistemic std | 0.01066 |
| OOD epistemic std | 0.01640 |
| OOD / ID ratio | 1.5383 |

## Failure Cases / Limitations

- OODでepistemicは上がりましたが、memberのaleatoric varianceの増加の方が大きく、二つは完全には分離できませんでした。
- bootstrap ensembleがすべての未知入力を検出する保証はありません。全memberが同じ偏りを共有すれば、そろって間違えます。
- Gaussian mixtureを一つのGaussianへmoment matchすると、二つに分かれた未来を隠すことがあります。
- TS1は一つの一貫したworld hypothesisに存在しないtrajectoryを作る可能性があり、TS∞は悪いmemberを長く信じる可能性があります。
- 小さな合成環境、member数5、一つのseedでの確認です。

## Findings / Compare Later

ensembleはtraining範囲の外で別のepistemic signalを出しましたが、それを万能な安全指標とは扱えません。後ではmember数、data量、single model、TS∞/TS1、planning時のrisk penaltyを比較します。

## References

- Chua et al., PETS, 2018. https://arxiv.org/abs/1805.12114
- Kendall and Gal, 2017. https://arxiv.org/abs/1703.04977

PETSのうち、この実験が扱うのはprobabilistic ensemble、bootstrap、particle propagationです。CEM control、elite selection、PETS benchmarkはここには含みません。
