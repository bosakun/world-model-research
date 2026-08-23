# Reward、Value、Continuationを世界モデルへ足す

## 何が困るのか

次の画像を予測できても、Agentはどの未来が良いか分かりません。行動を選ぶには、未来の見た目だけでなく、得点と終了も予測する必要があります。

## 三つのhead

| head | 質問 | 例 |
|---|---|---|
| reward | この一歩で何点？ | Goalへ着いたので+1 |
| value | この先の合計は何点？ | 遠回りでも後でGoalへ着く |
| continuation | 次stepは続く？ | Goal到達でepisode終了 |

同じlatent stateから三つのheadを出します。

    latent state
    -> reward
    -> value
    -> continuation

## なぜ分けるのか

rewardは一歩の得点です。valueは将来の得点まで含みます。continuationは、終了後の存在しない未来を数えないために必要です。三つを混ぜると、何を予測しているか分かりにくくなります。

## 外すとどうなるか

- reward headを外す: action候補の直近の良さを採点できない。
- value headを外す: 長い未来の良さを見積もりにくい。
- continuation headを外す: episode終了後もrewardが続くように扱う危険がある。

## 自分で説明できるか

- rewardとvalueの違いは何か。
- continuationはなぜ必要か。
- 画像予測だけでは行動を選びにくい理由は何か。
