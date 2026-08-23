# はじめにここから: 06 Reward / Value

未来を予測できても、「どの未来が良いか」が分からなければAgentは行動を選べません。このフェーズではworld modelに三つの予測headを足します。

| head | 質問 |
|---|---|
| reward | この一歩でどれくらい良いことが起きる？ |
| value | この先ずっと合計するとどれくらい良い？ |
| continuation | このepisodeは次stepも続く？ |

まずは[01_prediction_heads](01_prediction_heads/README.md)、次に[ARTICLE.md](ARTICLE.md)を読んでください。
