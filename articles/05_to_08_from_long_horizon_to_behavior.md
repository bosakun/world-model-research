# 長い未来を予測し、良い行動を選ぶまで

## この記事でつなぐもの

世界モデルが次の状態を予測できても、それだけではAgentは行動できません。

    長い未来を想像する
    -> その未来の良さを評価する
    -> action候補を比べる
    -> 方策そのものを学ぶ

この記事は、05 Long Horizon、06 Reward / Value、07 Planning、08 Imagination RLを一つの流れとして扱います。

## 1. 1stepで当たっても、遠い未来では崩れる

学習中は正解stateを入力にできます。しかしrolloutでは、自分の予測を次の入力へ戻します。

    正解state -> 予測1step
    予測state -> 予測2step
    予測state -> 予測3step

小さな誤差が積み重なる現象をcompounding errorと呼びます。

latent overshootingは、1step後だけでなく数step先のlatentも直接近づけます。

    時刻tからk step想像したlatent
    と
    時刻t+kの画像から推論したlatent
    を近づける

未来画像は学習時の答え合わせだけに使い、rolloutの入力には入れません。これが「未来を見ずに想像する」条件を守るために重要です。

temporal abstractionは、複数のprimitive actionをmacro actionへまとめます。

    right, right, right, right
    -> 「右へ4step進む」

予測をつなぐ回数を減らせますが、途中の接触や障害物のような細かい変化を飛ばす弱点があります。

詳細: [05 Long Horizon](../05_long_horizon/START_HERE.md)

## 2. 未来に点数を付ける

未来を予測しても、「どの未来が良いか」は別に学ぶ必要があります。

    reward:       この一歩の得点
    value:        この先を含む得点
    continuation: episodeが次stepも続く確率

rewardだけでは目先の得しか見えません。valueがあると、遠回りして後でGoalへ着くactionを評価できます。continuationがないと、終了後の存在しないrewardまで数える危険があります。

    latent state
    -> reward head
    -> value head
    -> continuation head

詳細: [06 Reward / Value](../06_reward_value/01_prediction_heads/README.md)

## 3. modelの中でaction候補を試す

Planningは、本物の環境で試す前にworld model内で未来を試すことです。

Random Shootingは、ランダムなaction列を多数作り、最も高い予測scoreの列を選びます。

CEMは、良かった候補の近くを重点的にsampleします。

    sample -> elite選択 -> 分布更新 -> 再sample

MPCは、計画した列の最初のactionだけを実行し、次の本物の観測を得て計画を作り直します。

    計画 -> 最初の一歩だけ実行 -> 観測 -> 再計画

latent planningは画像を毎step復元せず、latent内のreward/valueで候補を採点します。速くなり得ますが、latentにcontrolに必要な情報がなければ失敗します。

詳細: [07 Planning](../07_planning/START_HERE.md)

## 4. 想像した未来で方策を学ぶ

Planningは行動ごとに候補を探します。Imagination Actor-Criticは、「このlatentならこのaction」というActorをworld model内部で学びます。

    latent -> Actor -> action
    latent + action -> imagined next latent
    predicted reward + Critic value -> Actor/Critic更新

Actorはactionを出し、Criticは将来の良さを評価します。実環境で何千回も試さず、imagined trajectoryから学べることが利点です。

ただしworld modelが間違っていれば、Actorはmodelの中でだけ得するactionを学ぶ可能性があります。imagined returnだけでなく、実環境returnで確認する必要があります。

詳細: [08 Imagination RL](../08_imagination_rl/ARTICLE.md)

## まとめ

- 長い未来では、1step精度だけでは足りない。
- reward、value、continuationが未来を意思決定可能な形にする。
- Planningは候補列をworld model内で比べる。
- Imagination RLは、その探索をActor/Criticへ学習させる。
- 予測modelが間違えば、planningもpolicyも間違った方向へ最適化される。
