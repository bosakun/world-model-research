# はじめにここから: 03 Memoryを自分で作って理解するための道案内

このページは`03_memory/`を学ぶための入口です。目的は、**自分で小さな世界モデルを作りながら、「過去を覚えると何が変わるのか」を理解すること**です。

最初から論文、数式、巨大なmodelを理解する必要はありません。まずは、次の一文を自分の言葉で言えれば十分です。

> 世界モデルは、「今見えているもの」と「自分がした行動」から、次に何が起きるかを予想するプログラムである。

## このリポジトリで使う小さな世界

5x5のマス目に、AgentとGoalがあります。

```text
Agentが右へ動く
↓
画像がどう変わるかを予測する
```

これが一番小さいWorld Modelです。難しい名前が出てきても、最初は「Agentが動いたら次の絵を当てる」という話に戻ってください。

## 先に覚えなくてよい言葉

次の言葉は、最初は名前だけで構いません。

- latent: 画像を小さな数字の並びにしたもの
- hidden state / memory: 過去を忘れないためのメモ
- prior: 次の画像を見る前の予想
- posterior: 実際に画像を見た後の修正
- KL divergence: priorとposteriorのズレを小さくするための量
- planning: 未来を何通りか予想して、良いactionを選ぶこと

名前を暗記するより、「なぜそれが必要になったのか」を順番に見る方が大切です。

## おすすめの順番

### Lesson 1: 未来を予測するとは何か

[01_gru/README.md](01_gru/README.md)を読みます。

ここでは「今の画像」と「右へ行くaction」から「次の画像」を当てます。最初の目標は、`z`やGRUを理解することではなく、**inputとoutputが何か**を言えるようになることです。

```text
今の画像 + action -> 次の画像
```

次に[01_gruのUNDERSTANDING.md](01_gru/UNDERSTANDING.md)を読みます。

### Lesson 2: なぜ過去を覚える必要があるか

[02_partial_observation/README.md](02_partial_observation/README.md)を読み、`outputs/aliasing_pair.png`を見ます。

ここで初めて、**同じ現在画像なのに、正しい答えが二通りある**状況が出ます。これがmemoryを導入する理由です。

### Lesson 3: 画像を見る前と見た後を分ける

[03_rssm/README.md](03_rssm/README.md)を読みます。

RSSMで重要なのは一つです。

```text
未来の画像はまだ見えない
↓
見えないまま未来を予測する仕組みが必要
```

この「画像を見る前の予想」がpriorです。posteriorやKLの数式は、その必要性が分かってから読んでください。

### Lesson 4: 過去を1個のメモにするか、全部並べて参照するか

[04_transformer_memory/README.md](04_transformer_memory/README.md)を読みます。

- GRU: 過去を1冊のメモ帳へ要約する。
- Transformer: 過去のページを並べ、必要なページを探す。

この比喩で違いが分かれば、attentionの式は後回しで大丈夫です。

## 03 Memoryを学び終えたら

### 次: 本当に良くなったかを調べる

[90_evaluation/01_memory_benchmark/README.md](../90_evaluation/01_memory_benchmark/README.md)を読みます。

ここでは「複雑なmodelだから良い」と決めず、同じ条件で比べます。

### Lesson 6: 部品をつないでAgentが行動するまで

[99_integrated_world_model/01_evidence_selected/README.md](../99_integrated_world_model/01_evidence_selected/README.md)を読みます。

最後に、画像を見て、過去を覚え、未来を想像し、actionを選ぶところまでつなげます。

## コードを読むときの方法

一度に全ファイルを読まないでください。各Lessonで次の順に見ます。

1. `env.py`: 世界で何が起きるか。
2. `dataset.py`: 学習用の問題をどう作るか。
3. `model.py`: inputからoutputまで何を計算するか。
4. `losses.py`: modelに何を正解として学ばせるか。
5. `tests/`: 「正しく動く」の意味。
6. `evaluate.py`と`outputs/`: 実際にできたか。

分からない行があれば、まずその行を飛ばして「inputは何で、outputは何か」だけを追ってください。数式やPyTorchの細部は、必要になった場所で戻れば十分です。

## 毎回、自分に聞く3つの質問

1. 前のmodelでは、何ができなかったか？
2. 今回足した部品は、その問題のどこを解決するか？
3. その部品を外すと、何が起きるか？

この3つに答えられれば、論文の式を完全に暗記していなくても、仕組みを理解し始めています。

## 実際に動かしてみる

最初のLessonは、リポジトリのルートで次を実行して確認できます。

```bash
uv run pytest -q 03_memory/01_gru/tests
uv run python 03_memory/01_gru/train.py
uv run python 03_memory/01_gru/evaluate.py
```

実行後は、まず`03_memory/01_gru/outputs/rollout_comparison.png`を見てください。「予測画像がstepを重ねるとどう崩れるか」を見ることが、数字より先です。
