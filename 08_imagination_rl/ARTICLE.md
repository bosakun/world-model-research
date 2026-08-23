# 想像した未来からAgentはどう学ぶのか

## 1. PlanningとActor-Criticの違い

Planningは、行動するたびに候補のaction列を探します。Actor-Criticは、過去の経験から「この状態ならこのactionを選ぶ」というActorを育てます。

    Actor: 状態を受け取りactionを出す
    Critic: その状態から先の良さを予測する

## 2. Imagination

本物の環境で毎回試す代わりに、world modelの中でlatent trajectoryを作ります。

    latent state
    -> Actorがactionを出す
    -> world modelが次のlatentを想像
    -> reward / continuationを予測
    -> Criticが未来の価値を予測

この想像上の経験でActorとCriticを更新します。

## 3. なぜ便利か

本物のロボットや環境で失敗を何千回も試すのは高価で危険な場合があります。world modelがある程度正しければ、内部の想像で多くの学習を進められます。

ただしworld modelが間違っていれば、Actorは「モデルの中でだけ通用するずるいaction」を学ぶ危険があります。実環境のデータでworld modelを更新し続ける必要があります。

## 4. まとめ

- Actorはactionを選ぶ。
- Criticは将来の良さを予測する。
- world modelの内部で未来を作って二つを学習する。
- 想像の質はworld modelの質に依存する。
