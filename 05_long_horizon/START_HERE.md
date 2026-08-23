# はじめにここから: 05 Long Horizon

ここでは、未来を何stepも予測すると誤差がなぜ大きくなるかを学びます。

    1step目の小さな誤差
    -> 次stepでは誤った予測を入力にする
    -> 誤差が積み重なる

この現象をcompounding errorと呼びます。

## 学ぶ順番

1. [01_latent_overshooting](01_latent_overshooting/README.md)

   1stepだけでなく、数step先のlatentも当てるよう学習する理由を学びます。

2. [02_temporal_abstraction](02_temporal_abstraction/README.md)

   一歩ずつ予測する代わりに、数step分をまとめたmacro actionで予測する考え方を学びます。

3. [ARTICLE.md](ARTICLE.md)

## ここで説明できれば十分

- 1step予測が良くても、長いrolloutが良いとは限らない。
- overshootingは、数step後の予測にも直接lossを与える方法である。
- temporal abstractionは、細かいactionをまとめて長い変化を見る方法である。
