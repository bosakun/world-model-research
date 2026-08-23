# Lesson 4: 過去を一つに要約せず、必要な場面を見返す

GRUは過去を`h_t`という一つのメモ帳にまとめます。Transformerは別の考え方をします。過去の出来事をページとして残し、今必要なページを見返します。

## GRUとの違いをたとえで考える

```text
GRU:
毎日の日記を、1枚の要約メモへ書き直す

Transformer:
過去の日記ページを何枚か残し、必要なページを開く
```

GRUでは、最初に見たGoalの情報が何stepも`h_t`に残り続ける必要があります。Transformerでは、過去のGoalが入ったtokenがcontext内に残っていれば、今の予測がそこを参照できます。

ただしTransformerも無限に覚えられるわけではありません。`max_context`より古いページは捨てられます。

## tokenとは何か

この実験では、一時刻を一つのtokenにします。

```text
token t = 今の画像のメモ z_t + その後に押すaction a_t + 時刻の印 p_t
```

| 部品 | 必要な理由 |
|---|---|
| `z_t` | 今見えているものを表す |
| `a_t` | 同じ画像でも押すボタンで未来が変わる |
| `p_t` | 何番目の出来事かを知らせる。これがないと順序が分からない |

## attentionとは何か

attentionは、今のtokenが過去tokenへ「どのページが役立ちそうか」と重みを付ける仕組みです。

```text
今の予測
  -> 過去のtoken 0を少し見る
  -> token 1を多く見る
  -> token 2も少し見る
  -> それらを混ぜて次の予測を作る
```

重要なのは、attentionが「必ず一つの正しいページを選ぶ」仕組みではないことです。複数のページに少しずつ注意を向けることもあります。

## causal maskはなぜ必要か

trainingでは、系列全体が手元にあります。もしmodelが未来のtokenを見られると、次の状態を当てる問題で未来の答えを読めてしまいます。

```text
token 0は token 0だけを見る
token 1は token 0,1を見る
token 2は token 0,1,2を見る
token 2は token 3以降を見てはいけない
```

この「未来を見るのを禁止する壁」がcausal maskです。testでは、未来tokenを大きく変えても過去のoutputが変わらないことを確認しています。

## 学習とrollout

### teacher forcing

学習中は、本物の`z_t`を全部のtokenへ入れます。多くの時刻をまとめて計算できるのがTransformerの利点です。

### rollout

未来を想像するときは、予測した`z_hat`を次のtokenへ追加します。

```text
z_0 + a_0 -> z_hat_1
z_hat_1 + a_1 -> z_hat_2
z_hat_2 + a_2 -> z_hat_3
```

このときは予測を一つずつ追加するので、GRUと同じく誤差が蓄積します。teacher forcingがparallelでも、未来imaginationまで自動的に簡単になるわけではありません。

## GRUとTransformerを比べる

| 質問 | GRU | Transformer |
|---|---|---|
| 過去の保存方法 | 1本のhiddenメモ | tokenのリスト |
| 古い出来事の利用 | hiddenに残っていれば使える | context内なら直接参照できる |
| 学習中の計算 | stepごとに進む | 多くのtokenをまとめて処理できる |
| 長い系列のコスト | 比較的増えにくい | token数が増えると重くなりやすい |

どちらが常に上ではありません。短い系列ではGRU/RSSMの方が軽く、長い履歴が必要ならTransformerが有利かもしれません。だから同じ条件で比較します。

## コードを読む順番

```text
transformer_memory.py の tokenize
-> position_embedding
-> causal_mask
-> CausalTransformerBlock
-> prediction_head
-> rollout
-> evaluate.py の attention_map
```

最初にQ/K/Vの行列計算を追う必要はありません。`causal_mask`が未来を隠し、`tokenize`が`z_t,a_t,p_t`を一つにしていることを確認してください。

## 自分で答えてみる

- GRUのメモ帳とTransformerのページ棚はどう違うか。
- tokenにactionが必要な理由は何か。
- positionがないと何が分からなくなるか。
- causal maskがないと、なぜ予測の評価がずるになるか。
- attention mapの明るい場所は何を表すか。
- attention weightだけで「この情報が原因だった」と言えないのはなぜか。

## ここまでで十分な理解

「Transformerは過去を一つのhiddenへ圧縮せず、順序付きの過去tokenを残し、未来を見ない制約の下で必要なtokenを参照して次を予測する」と説明できれば十分です。
