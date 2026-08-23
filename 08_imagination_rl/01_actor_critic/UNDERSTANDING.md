# 想像した未来でActorとCriticを学ぶ

## 何が困るのか

Planningは行動のたびに多数の候補を探します。毎回の探索を減らしたいなら、「このstateならこのaction」を出すActorを学習させます。

## 二つのnetwork

| 名前 | 役割 |
|---|---|
| Actor | latent stateからactionを出す |
| Critic | そのstateから先の良さを予測する |

## imaginationの流れ

    現在のlatent state
    -> Actorがactionを選ぶ
    -> world modelが次latentを想像する
    -> rewardとcontinuationを予測する
    -> Criticが将来の価値を予測する

この想像上のtrajectoryでActorとCriticを更新します。

## 注意点

本物の環境で試さない分、学習をたくさん進められます。しかしworld modelが間違っていれば、Actorはmodelの中でだけ得するactionを学ぶことがあります。実環境データでworld modelを更新し続ける必要があります。

## 自分で説明できるか

- ActorとCriticは何が違うか。
- imagined trajectoryとは何か。
- world modelの誤りがActorへどう影響するか。
