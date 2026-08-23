# 研究ノート: Memory統一比較

## 比較前の問題

個別実験はそれぞれ動いたが、Encoder、loss、seed、学習量が違った。数値を横に並べるだけでは「memory architectureの差」なのか「条件の差」なのか分からない。

## 重要な設計

- even/odd pairで、最初に見えるGoalだけを変え、actionとGoal消失後の現在画像を同一にした。
- modelが本当に持つmemory stateからGoalを読む。視覚latentに無理にすべてを押し込む不公平を避けた。
- reset/truncate ablationを最初から入れた。

## 結果と解釈

- RSSM/Transformerは3/3 seedで1.0、GRUは平均0.833だが1 seed失敗。
- ablation後は全memory modelが0.5。履歴情報が分類へ使われた。
- h10 image MSEは似通い、No Memoryが悪くない場合もある。見えないGoalを忘れても、背景が大部分を占める画像は当てられる。
- RSSMを採用したのは「最高の画像MSE」ではなく、stable memory、Transformerより低latency、prior/posteriorの下流interfaceを合わせて評価したため。

## 記事材料

- `memory_comparison.png`。
- 「同じ現在画像で二択なら、memoryなしの理論上限は50%」という説明。
- 「複雑なmodelの採否は一つのlossで決めない」。

## 次に調べること

more seeds、longer delay、Goal supervisionなし、noise/OOD、planning success、peak memory。
