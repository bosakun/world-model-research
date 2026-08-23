# 世界モデルは「次に何が起きるか」だけでは行動できない

世界モデルが未来の画像を予測できても、Agentはまだ「どのactionを選ぶべきか」を知りません。そこで、状態から良さを予測するheadを追加します。

## reward

rewardは一歩ぶんの得点です。

    Goalへ近づく -> 小さなプラス
    Goalへ到着する -> 大きなプラス

ただし、一歩のrewardだけを見ると、すぐの得だけを選びやすくなります。

## value

valueは、現在から先に得られるrewardの合計を予測します。

    今の状態は、将来も含めるとどれくらい良いか

遠回りしても後でGoalへ着くactionを選ぶために必要です。

## continuation

episodeには終わりがあります。Goalへ着いた、壁にぶつかった、時間切れになった、などです。continuationは「次stepが続く確率」を予測します。

これがないと、終了後もrewardが続くように数えてしまうことがあります。

## 三つの関係

    latent state
    -> reward head: 今回の得点
    -> value head: 未来を含む得点
    -> continuation head: 未来が続くか

この実装はDreamer系のheadの考え方を小さな環境へ持ち込んだ教育用実装です。次のPlanningでは、これらの予測を使ってaction候補を比べます。
