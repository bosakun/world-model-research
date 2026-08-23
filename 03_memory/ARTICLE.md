# 世界モデルにMemoryはなぜ必要なのか

この記事では、5×5のGrid Worldを使って、世界モデルにMemory（記憶）が必要になる理由を順番に説明します。

最初の世界モデルは、次の処理だけです。

    今の画像 + Agentのaction -> 次に起きることを予測

画像をEncoderで短い数字の並びへ変換し、Dynamics Modelで次の並びを予測し、Decoderで次の画像へ戻します。大切なのは数式の暗記ではなく、「何を入力し、何を予測しているか」を説明できることです。

## 1. GRUは過去をメモする

Memoryなしモデルは、現在の潜在状態 z_t と action a_t だけを使います。

    z_t + a_t -> 次のzを予測

GRUでは、過去の出来事をまとめた h_t も入力します。

    z_t + a_t + h_t -> GRUCell -> h_{t+1} -> 次のzを予測

| 名前 | たとえ | 役割 |
|---|---|---|
| z_t | 今の写真のメモ | 現在画像の要約 |
| a_t | 押したボタン | Agentの行動 |
| h_t | 出来事をまとめたメモ帳 | 過去の画像とactionの要約 |

GRUは、古いメモのどこを残し、現在の情報をどこへ書き足すかを学習します。人間のように文章を保存するのではなく、未来の予測に役立つ数字を h_t に残します。

学習中は正解画像を使えます（teacher forcing）。しかし未来を想像するときは正解画像がないため、モデル自身の予測を次の入力へ戻します。

    z_0 -> 予測z_1 -> 予測z_2 -> 予測z_3

そのため、長いrolloutほど小さな間違いが積み重なります。

## 2. 部分観測でMemoryの必要性を作る

世界全体が毎回見えているなら、過去を覚えなくても現在画像だけで答えられます。そこでAgentの周囲3×3だけが見える環境を作りました。

    世界A: 最初にGoalが右に見えた
    世界B: 最初にGoalが下に見えた
    その後、Agentが同じように移動する

Goalが視界の外へ消えると、現在画像は同じなのに本当のGoal位置が違う状態になります。

    現在画像A == 現在画像B
    本当の世界A != 本当の世界B

これをaliasと呼びます。「同じ写真なのに、過去によって正しい答えが変わる」という意味です。

| 言葉 | 意味 |
|---|---|
| true state | AgentとGoalの本当の座標。環境だけが知る |
| observation | Agentに見せる局所的な画像 |

true stateをモデルへ渡すと答えを先に教えることになります。true stateは答え合わせだけに使います。

一枚の画像だけでは過去が分からないため、GRUには系列を渡します。

    o_0, a_0, o_1, a_1, o_2

これを読むことで、GRUは「前にGoalが右にあった」「その後leftを押した」という手掛かりを h_t に残せます。

## 3. なぜGRUが有利だとまだ言えないのか

Partial Observationで確認できたのは、Memoryが必要になりそうな問題を作れたことまでです。まだ「GRUが実際に過去を使い、Memoryなしより良い」と証明したわけではありません。

GRUを入れても、次の可能性があります。

- h_tをほとんど使わない。
- 学習が足りず、Goal情報を覚えられない。
- データの偏りでMemoryなしでも偶然当たる。
- 改善がMemory以外の理由かもしれない。

本当に比較するには、同じデータ、同じ学習回数、同じ評価方法で、MemoryなしモデルとGRUを比べます。特にGoalが視界外になった後の予測を比較します。この比較は03_memory/05_comparisonで行う予定です。

## 4. RSSMのpriorとposterior

RSSMでは、過去のメモ h_t、画像特徴 e_t、確率的な状態 z_t を分けます。

    h_t : 過去の出来事メモ
    e_t : 現在画像をEncoderに通した特徴
    z_t : 現在の世界についての仮説

画像を見る前の予想がpriorです。

    h_t -> prior -> z_t

現在画像を見た後の修正がposteriorです。

    h_t + e_t -> posterior -> z_t

posteriorを挟むのは、z_tを一つの固定値ではなく、複数の可能性を含む確率分布から作るためです。画像だけではGoalが左か右か分からないとき、その不確実さを表せます。

学習中は画像があるのでposteriorを使えます。しかし未来には正解画像がありません。未来rolloutではpriorだけで進む必要があります。

    posterior: 画像を見た後の推測
    prior:     画像を見る前の予測
    KL:        priorとposteriorのズレを小さくする

KL divergenceは、画像を見ているときに得た知識を、画像なしの未来予測にも移すために使います。RSSMはPlaNet等を参考にしていますが、このリポジトリは小さな教育用の簡略実装です。

## 5. Transformer Memoryとの違い

GRUは過去を一つの h_t へまとめます。Transformerは過去のtokenを並べ、必要なページをattentionで参照します。

    GRU:         過去を一冊の要約メモにする
    Transformer: 過去のページを並べ、必要なページを見る

時刻2は時刻0と時刻1を見られますが、まだ存在しない時刻3は見られません。これがcausal maskです。この実装の目的は、Transformerが必ずGRUより良いと示すことではなく、過去の要約方法の違いを理解することです。

## 6. Pixel MSEとshortcut

MSEはMean Squared Error（平均二乗誤差）です。正解画像と予測画像の各pixelの差を二乗し、全pixelで平均します。

    正解pixel 1.0、予測pixel 0.8 -> 差の二乗は0.04

5×5の画像で、背景が23マス、Agentが1マス、Goalが1マスだとします。Goalは画像のごく一部です。モデルが背景をきれいに描き、Goalを消しても、画像の大部分は一致するのでPixel MSEは低くなることがあります。

このように、重要な構造を学ぶ代わりに、簡単な手掛かりだけでlossを下げる方法をshortcutと呼びます。モデルは「Goalは重要」と理解せず、lossが下がる方法を選んでいるだけです。

## 7. Goal lossとposition loss

画像全体のMSEだけでは、小さなAgentやGoalを無視できます。そこで重要な位置を別に採点します。

    画像全体      -> Pixel MSE
    Agentの位置   -> Position loss
    Goalの位置    -> Goal loss（追加する場合）

現在の01_gruで実装されているのは、Goal lossではなくAgentのposition lossです。Agentが25マスのどこにいるかを、25クラスの分類問題として学習します。

このlossは、背景だけを再現するshortcutを取りにくくし、Agentの位置を直接学ばせます。Goal lossを追加する場合も、目的は「Goalという意味を、画像全体の平均とは別に評価すること」です。

ただしPartial ObservationでGoalが視界外なら、現在画像だけではGoal位置を確定できません。過去の観測とactionをGRUへ渡し、hidden stateから推測させる必要があります。

## 8. 人間の脳との関係

脳がRSSMをそのまま計算していると証明されたわけではありません。ただ、似た直感はあります。人間は過去の経験から「この位置に机があるはず」と予想し、視覚情報でその予想を修正します。

    過去の経験 -> 次の予想 -> 新しい感覚 -> 予想を修正

priorは予想、posteriorは感覚を見た後の推測に近いものです。大切なのは、現在の感覚だけでなく、過去の記憶と予測を使って見えないものを推測している、という点です。

## 9. まとめ

1. 世界モデルは、現在の情報とactionから未来を予測する。
2. 現在画像だけで世界が分かるなら、Memoryは必要ない。
3. 部分観測では、同じ画像でも過去によって本当の状態が変わる。
4. GRUは過去の情報を h_t へまとめる。
5. RSSMはprior（画像を見る前）とposterior（画像を見た後）を分ける。
6. Pixel MSEが低いだけでは、Goalを覚えた証拠にならない。
7. position lossやGoal lossは、重要な意味を別に評価する。
8. GRUがMemoryなしより有利かどうかは、同じ条件の比較で初めて分かる。

## 関連する実装

- [01_gru](01_gru/README.md)
- [02_partial_observation](02_partial_observation/README.md)
- [03_rssm](03_rssm/README.md)
- [04_transformer_memory](04_transformer_memory/README.md)
- [Memory比較の計画](../90_evaluation/01_memory_benchmark/README.md)

## 参考文献

- Ha and Schmidhuber, *World Models*, 2018. https://arxiv.org/abs/1803.10122
- Hafner et al., *Learning Latent Dynamics for Planning from Pixels*（PlaNet）, 2019. https://arxiv.org/abs/1811.04551
- Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination*, 2020. https://arxiv.org/abs/1912.01603

この実装は論文の考え方を参考にした、独立した小規模・教育用実装です。論文の完全再現ではありません。
