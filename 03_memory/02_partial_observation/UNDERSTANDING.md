# Lesson 2: なぜ過去を覚えないといけないのか

Lesson 1ではGRUというメモ帳を作りました。しかし、今の画像だけで答えが分かるなら、メモ帳は必要ありません。このLessonでは、**過去を覚えないと答えられない問題**を作ります。

## 目で見て確かめる

まず`outputs/aliasing_pair.png`を開いてください。

二つの世界があります。

```text
世界A: 最初にGoalが右に見えた
世界B: 最初にGoalが下に見えた
```

Agentはどちらでも同じようにleftを2回押します。するとGoalは視界の外へ消えます。

```text
t=0: Goalが見える
t=1: Agentがleftへ動く
t=2: Goalが見えない
```

`t=2`の画像はAとBで完全に同じです。しかし、本当のGoalの位置は違います。

```text
画像A == 画像B
でも
本当の世界A != 本当の世界B
```

この状態をaliasと呼びます。名前を覚えるより、「同じ写真なのに正しい答えが違う」と考えてください。

## stateとobservationの違い

ここで二つの言葉を分けます。

| 言葉 | 意味 | 誰が知っているか |
|---|---|---|
| true state | AgentとGoalの本当の座標すべて | 環境（simulator） |
| observation | Agentが今見られる3x3の画像 | Agent / model |

環境はGoalが右か下かを知っています。しかしmodelには局所画像しか渡しません。`true_states`は「あとで答え合わせをするため」に持っているだけで、modelへ入力してはいけません。

## なぜ現在画像だけでは足りないか

memoryなしmodelが受け取るものは、今の画像とactionだけです。

```text
同じ画像 + 同じaction -> 同じ予測
```

しかし世界AとBは、同じ現在画像でも本当は違います。memoryなしmodelは二つの答えを区別できないので、どちらかを当ててももう一方を外します。均等な二択なら最高でも50%です。

GRUなら、`t=0`で見えたGoalの方向を`h_t`に残せます。

```text
最初に右Goalを見た -> h_tに右の手掛かり
最初に下Goalを見た -> h_tに下の手掛かり
```

`t=2`の画像が同じでも、`h_t`が違えば違う行動や予測を選べます。

## POMDPは何か

POMDPは「世界の全情報を見られない問題」の名前です。難しい数式を先に覚えなくて大丈夫です。

```text
本当の世界 -> カメラ -> 見える一部分
```

カメラが見せない場所には情報があっても、Agentには分かりません。だから複数の時刻の観測とactionを合わせて考える必要があります。

## なぜsequence datasetが必要か

`t=2`の画像だけを一枚渡しても、Goalが以前どこにあったかは分かりません。

GRUへ渡すのは一枚ではなく、時系列です。

```text
o_0, a_0, o_1, a_1, o_2
```

この列があって初めて、GRUは「前に見えたGoal」と「その後に押したaction」をメモできます。

## コードを読む順番

```text
partial_env.py      : true stateと局所カメラ
partial_dataset.py  : paired aliasの系列を作る
visualize.py        : 同じ画像・違うGoalを図にする
model_adapters.py   : 既存GRU/Simple Dynamicsへつなぐ
tests/              : Goalが本当に漏れていないかを確認する
```

まず`partial_env.py`で「3x3の外は青いunknownになる」ことだけ確認してください。次に`partial_dataset.py`で、right Goalとdown Goalのpairを作っている部分を探します。

## このLessonでできるようになること

- stateとobservationの違いを言える。
- 同じ現在画像でも過去によって本当の世界が違う例を説明できる。
- 画像が一枚ではなくsequenceで必要な理由を言える。
- `true_states`をmodel入力へ渡すと何が悪いかを説明できる。
- まだGRUが有利だと証明できていない理由を言える。

次のLessonでは、過去を覚えるだけでなく、**未来の画像を見る前にどう予測するか**を考えます。
