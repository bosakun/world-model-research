# 長い未来を予測すると、なぜ世界モデルは崩れるのか

## 1. 一歩ずつなら当たるのに、十歩先では外れる

学習中のworld modelは、正解の現在状態を入力にして次の状態を当てることが多いです。

    正解の状態 s_t -> 予測 s_{t+1}

しかし実際に未来を想像するrolloutでは、前の予測を次の入力にします。

    s_0 -> 予測s_1 -> 予測s_2 -> 予測s_3

1step目の小さなズレが次stepの入力になるため、ズレは積み重なります。これがcompounding errorです。

## 2. latent overshooting

RSSMのようなlatent world modelでは、1step後だけを当てるlossだと、遠い未来のlatentが正しい場所に行くとは限りません。

latent overshootingは、現在から1step、2step、3step先まで想像し、それぞれを実際の観測から推論した状態に近づけます。

    現在から想像した3step先
    と
    3step後の画像を見て推論した状態
    を近づける

目的は、長いrolloutでも使えるDynamicsを学ばせることです。未来の画像をrollout中に見せるのではなく、学習時の答え合わせにだけ使います。

## 3. temporal abstraction

長い未来を一歩ずつ予測すると、計算も誤差の機会も増えます。

そこで複数のprimitive actionを一つのmacro actionとして扱います。

    右、右、右、右
    -> 「右へ4マス進む」というまとまり

temporal abstractionは、短い時間の物理的な細部より、数step単位の変化を予測する方法です。いつも有利とは限りません。細かい制御が必要な場面では、途中を飛ばすことが弱点になります。

## 4. このフェーズの要点

- 長期予測は、1step精度だけでは評価できない。
- overshootingは遠い未来にも直接学習信号を送る。
- temporal abstractionは、時間を粗く見て長い変化を扱う。
- どちらも誤差を魔法のように消すものではなく、長い未来を学ぶための設計である。

## 次に読むもの

- [Reward / Value](../06_reward_value/01_prediction_heads/README.md)
- [01_latent_overshooting](01_latent_overshooting/README.md)
- [02_temporal_abstraction](02_temporal_abstraction/README.md)
